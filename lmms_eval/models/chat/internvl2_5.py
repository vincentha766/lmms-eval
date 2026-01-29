from __future__ import annotations

import time
from typing import List, Optional, Union

import cv2
import numpy as np
import torch
from accelerate import Accelerator, DistributedType
from accelerate.state import AcceleratorState
from loguru import logger as eval_logger
from PIL import Image
from tqdm import tqdm
from transformers import (
    AutoImageProcessor,
    AutoModelForCausalLM,
    AutoTokenizer,
)

from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from lmms_eval.models.model_utils.gen_metrics import log_metrics
from lmms_eval.protocol import ChatMessages


@register_model("internvl2_5")
class InternVL2_5(lmms):
    """
    InternVL2.5 Model for vision-language understanding tasks.
    Supports both image and video inputs.

    Example usage:
        python -m lmms_eval \\
            --model internvl2_5 \\
            --model_args pretrained=/root/autodl-tmp/InternVL2_5-8B,num_frames=8 \\
            --tasks mme,lvbench \\
            --batch_size 1 \\
            --device cuda:0
    """

    is_simple = False  # Use chat model type (recommended)

    def __init__(
        self,
        pretrained: str = "OpenGVLab/InternVL2_5-8B",
        device: str = "cuda",
        device_map: str = "",
        batch_size: int = 1,
        num_frames: int = 8,
        use_cache: bool = True,
        trust_remote_code: bool = True,
        dtype: Optional[Union[str, torch.dtype]] = "auto",
        **kwargs,
    ) -> None:
        super().__init__()
        # Do not use kwargs for now
        assert kwargs == {}, f"Unexpected kwargs: {kwargs}"

        # Initialize accelerator
        accelerator = Accelerator()
        if accelerator.num_processes > 1 and device_map == "":
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"
        else:
            self._device = torch.device(device)
            self.device_map = device_map if device_map else "auto"

        self.pretrained = pretrained
        self.batch_size_per_gpu = int(batch_size)
        self.num_frames = num_frames
        self.use_cache = use_cache
        self.trust_remote_code = trust_remote_code

        # Process dtype
        if isinstance(dtype, str) and dtype != "auto":
            dtype = getattr(torch, dtype)
        elif dtype == "auto":
            dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        self.dtype = dtype

        # Initialize image processor
        eval_logger.info(f"Loading image processor from {pretrained}")
        self.image_processor = AutoImageProcessor.from_pretrained(
            pretrained, trust_remote_code=trust_remote_code
        )

        # Initialize tokenizer
        eval_logger.info(f"Loading tokenizer from {pretrained}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained, trust_remote_code=trust_remote_code
        )
        if (
            getattr(self.tokenizer, "pad_token_id", None) is None
            and getattr(self.tokenizer, "eos_token_id", None) is not None
        ):
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # Initialize model
        eval_logger.info(f"Loading model from {pretrained}")
        try:
            self._model = AutoModelForCausalLM.from_pretrained(
                pretrained,
                trust_remote_code=trust_remote_code,
                dtype=dtype,
                device_map=self.device_map,
                low_cpu_mem_usage=True,
            ).eval()
        except TypeError:
            # Fallback for older transformers versions
            self._model = AutoModelForCausalLM.from_pretrained(
                pretrained,
                trust_remote_code=trust_remote_code,
                torch_dtype=dtype,
                device_map=self.device_map,
                low_cpu_mem_usage=True,
            ).eval()

        # Set image context token id
        img_ctx_id = self.tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>")
        self._model.img_context_token_id = img_ctx_id

        # Setup distributed training
        if accelerator.num_processes > 1 and device_map == "":
            assert accelerator.distributed_type in [
                DistributedType.FSDP,
                DistributedType.MULTI_GPU,
                DistributedType.DEEPSPEED,
            ], "Unsupported distributed type provided. Only DDP and FSDP are supported."
            # If you want to use DistributedType.DEEPSPEED, you have to run accelerate config before using the model
            # Also, you have to select zero stage 0 (equivalent to DDP) in order to make the prepare model works
            if accelerator.distributed_type == DistributedType.DEEPSPEED:
                kwargs = {
                    "train_micro_batch_size_per_gpu": self.batch_size_per_gpu,
                    "train_batch_size": self.batch_size_per_gpu
                    * accelerator.num_processes,
                }
                AcceleratorState().deepspeed_plugin.deepspeed_config_process(
                    must_match=True, **kwargs
                )
                eval_logger.info(
                    "Detected that you are using DistributedType.DEEPSPEED. Make sure you run `accelerate config` and set zero stage to 0"
                )
            if (
                accelerator.distributed_type == DistributedType.FSDP
                or accelerator.distributed_type == DistributedType.DEEPSPEED
            ):
                self._model = accelerator.prepare(self.model)
            else:
                self._model = accelerator.prepare_model(
                    self.model, evaluation_mode=True
                )
            self.accelerator = accelerator
            if self.accelerator.is_local_main_process:
                eval_logger.info(
                    f"Using {accelerator.num_processes} devices with data parallelism"
                )
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        elif accelerator.num_processes == 1 and device_map == "auto":
            eval_logger.info(
                f"Using {accelerator.num_processes} devices with pipeline parallelism"
            )
            self._rank = 0
            self._world_size = 1
        else:
            eval_logger.info(f"Using single device: {self._device}")
            self._model.to(self._device)
            self._rank = 0
            self._world_size = 1
        self.accelerator = accelerator

        eval_logger.info(
            f"InternVL2.5 model loaded successfully on {self.model.device}"
        )

    @property
    def model(self):
        # returns the model, unwrapping it if using Accelerate
        if hasattr(self, "accelerator"):
            return self.accelerator.unwrap_model(self._model)
        else:
            return self._model

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    @property
    def device(self):
        return self._device

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    def load_video_frames(
        self, video_path: str, num_frames: int
    ) -> List[Image.Image]:
        """
        Load frames from a video file.

        Args:
            video_path: Path to the video file
            num_frames: Number of frames to extract (evenly spaced)

        Returns:
            List of PIL Images
        """
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total_frames == 0:
            raise ValueError(f"Could not read video from {video_path}")

        # Calculate frame indices to extract (evenly spaced)
        frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)

        frames = []
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # Convert to PIL Image
                pil_image = Image.fromarray(frame_rgb)
                frames.append(pil_image)

        cap.release()

        if len(frames) == 0:
            raise ValueError(f"Could not extract any frames from {video_path}")

        return frames

    def get_conv_template(self, template_name: str):
        """Get conversation template for InternVL2.5 with ChatML format."""

        class ConvTemplate:
            def __init__(self, template_name):
                self.name = template_name
                self.system_message = "You are a helpful assistant."
                self.roles = ["user", "assistant"]
                self.messages = []
                self.sep = "<|im_end|>"

            def append_message(self, role, message):
                self.messages.append([role, message])

            def get_prompt(self):
                # Use ChatML format for InternVL2.5
                ret = f"<|im_start|>system\n{self.system_message}<|im_end|>\n"
                for role, message in self.messages:
                    if message:
                        ret += f"<|im_start|>{role}\n{message}<|im_end|>\n"
                    else:
                        ret += f"<|im_start|>{role}\n"
                return ret

        return ConvTemplate(template_name)

    def build_prompt(
        self,
        question: str,
        num_patches_list: List[int],
        system_message: Optional[str] = None,
    ) -> str:
        """Build prompt for InternVL model with image placeholders."""
        template_name = getattr(
            getattr(self.model, "config", None), "template", "internvl2_5"
        )
        num_image_token = int(getattr(self.model, "num_image_token", 256))

        if system_message is None:
            system_message = getattr(self.model, "system_message", None) or getattr(
                getattr(self.model, "conv_template", None),
                "system_message",
                "You are a helpful assistant.",
            )

        # Build the image placeholder
        image_tokens = ""
        for num_patches in num_patches_list:
            image_tokens += (
                "<img>" + "<IMG_CONTEXT>" * num_patches * num_image_token + "</img>\n"
            )

        # Build the full prompt
        template = self.get_conv_template(template_name)
        if system_message:
            template.system_message = system_message

        query = image_tokens + question
        template.append_message(template.roles[0], query)
        template.append_message(template.roles[1], None)
        prompt = template.get_prompt()

        return prompt

    def generate_until(self, requests: List[Instance]) -> List[str]:
        res = []

        def _collate(x):
            return x[2], x[2]

        # Group requests by their generation_kwargs
        re_ords = utils.Collator(
            [reg.args for reg in requests],
            _collate,
            group_fn=lambda x: x[2],
            grouping=True,
        )
        chunks = re_ords.get_batched(n=self.batch_size, batch_fn=None)
        num_iters = (
            len(requests) // self.batch_size
            if len(requests) % self.batch_size == 0
            else len(requests) // self.batch_size + 1
        )
        pbar = tqdm(
            total=num_iters, disable=(self.rank != 0), desc="Model Responding"
        )
        e2e_latency = 0
        total_tokens = 0

        for chunk in chunks:
            ctx, doc_to_messages, all_gen_kwargs, doc_id, task, split = zip(*chunk)
            task = task[0]
            split = split[0]
            chat_messages = [
                doc_to_messages[0](self.task_dict[task][split][ids]) for ids in doc_id
            ]
            chat_messages: List[ChatMessages] = [
                ChatMessages(**{"messages": message}) for message in chat_messages
            ]

            # Extract media from messages
            visuals_list = []
            videos_list = []
            text_prompts = []

            for chat_msg in chat_messages:
                images, videos, audios = chat_msg.extract_media()
                visuals_list.append(images)
                videos_list.append(videos)

                # Extract text content
                text_prompt = ""
                for msg in chat_msg.messages:
                    for content in msg.content:
                        if content.type == "text":
                            text_prompt += content.text
                text_prompts.append(text_prompt)

            # Flatten media lists
            visuals = self.flatten(visuals_list)
            videos = self.flatten(videos_list)

            gen_kwargs = all_gen_kwargs[0]

            # Set default generation kwargs
            default_gen_kwargs = {
                "max_new_tokens": 512,
                "temperature": 0.0,
                "do_sample": False,
            }
            current_gen_kwargs = {**default_gen_kwargs, **gen_kwargs}

            # Adjust sampling parameters
            if current_gen_kwargs["temperature"] > 0:
                current_gen_kwargs["do_sample"] = True
            else:
                current_gen_kwargs["do_sample"] = False
                current_gen_kwargs["temperature"] = None

            # Process inputs for each item in the batch
            batch_outputs = []
            for idx, text_prompt in enumerate(text_prompts):
                if self.accelerator.is_main_process and doc_id[idx] % 100 == 0:
                    eval_logger.debug(
                        f"Prompt for doc ID {doc_id[idx]}:\n\n{text_prompt}\n"
                    )

                # Process media based on modality
                pixel_values_list = []
                num_patches_list = []

                try:
                    if videos and len(videos) > idx and videos[idx]:
                        # Video input - extract frames
                        frames = self.load_video_frames(videos[idx], self.num_frames)
                        for frame in frames:
                            pv = self.image_processor(
                                images=frame, return_tensors="pt"
                            )["pixel_values"]
                            pixel_values_list.append(pv)
                        # Each frame is treated as a separate image patch
                        num_patches_list = [pv.shape[0] for pv in pixel_values_list]
                    elif visuals and len(visuals) > idx and visuals[idx]:
                        # Image input
                        images_to_process = (
                            visuals[idx]
                            if isinstance(visuals[idx], list)
                            else [visuals[idx]]
                        )
                        for img in images_to_process:
                            if isinstance(img, str):
                                img = Image.open(img).convert("RGB")
                            pv = self.image_processor(
                                images=img, return_tensors="pt"
                            )["pixel_values"]
                            pixel_values_list.append(pv)
                        # For images, each processed image may have multiple patches
                        num_patches_list = [pv.shape[0] for pv in pixel_values_list]

                    # Build prompt with image placeholders
                    if pixel_values_list:
                        pixel_values = torch.cat(pixel_values_list, dim=0)
                        pixel_values = pixel_values.to(
                            device=self._device, dtype=self.dtype
                        )
                        prompt = self.build_prompt(
                            question=text_prompt,
                            num_patches_list=num_patches_list,
                        )
                    else:
                        # Text-only input (should not happen in vision tasks)
                        pixel_values = None
                        prompt = text_prompt

                    # Tokenize prompt
                    model_inputs = self.tokenizer(prompt, return_tensors="pt")
                    input_ids = model_inputs["input_ids"].to(self._device)
                    attention_mask = model_inputs["attention_mask"].to(self._device)

                    # Prepare generation kwargs
                    eos_token_id = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
                    gen_config = {
                        "max_new_tokens": current_gen_kwargs["max_new_tokens"],
                        "do_sample": current_gen_kwargs["do_sample"],
                        "eos_token_id": eos_token_id,
                        "pad_token_id": self.tokenizer.pad_token_id,
                    }

                    if current_gen_kwargs["do_sample"] and current_gen_kwargs.get(
                        "temperature"
                    ):
                        gen_config["temperature"] = current_gen_kwargs["temperature"]

                    # Generate
                    start_time = time.time()
                    with torch.inference_mode():
                        if pixel_values is not None:
                            output_ids = self.model.generate(
                                pixel_values=pixel_values,
                                input_ids=input_ids,
                                attention_mask=attention_mask,
                                **gen_config,
                            )
                        else:
                            output_ids = self.model.generate(
                                input_ids=input_ids,
                                attention_mask=attention_mask,
                                **gen_config,
                            )
                    end_time = time.time()

                    # Decode output
                    # Note: InternVL2.5's generate() returns only the newly generated tokens,
                    # not the full sequence including input tokens. This is different from
                    # some other models that return the complete sequence.
                    input_len = int(input_ids.shape[1])

                    if output_ids.shape[1] > input_len:
                        # Model returned full sequence (input + generated)
                        gen_ids = output_ids[0, input_len:]
                    else:
                        # Model returned only generated tokens
                        gen_ids = output_ids[0]

                    answer = self.tokenizer.decode(
                        gen_ids, skip_special_tokens=True
                    ).strip()

                    # Track metrics
                    e2e_latency += end_time - start_time
                    total_tokens += len(gen_ids)

                except Exception as e:
                    eval_logger.error(f"Error {e} in generating")
                    answer = ""
                    e2e_latency += 0
                    total_tokens += 0

                if self.accelerator.is_main_process and doc_id[idx] % 100 == 0:
                    eval_logger.debug(
                        f"Generated text for doc ID {doc_id[idx]}:\n\n{answer}\n"
                    )

                batch_outputs.append(answer)
                self.cache_hook.add_partial(
                    "generate_until", (text_prompt, gen_kwargs), answer
                )

            res.extend(batch_outputs)
            pbar.update(1)

        # Reorder this group of results back to original unsorted form
        res = re_ords.get_original(res)

        pbar.close()

        # Log metrics
        metric_dict = {
            "total_tokens": total_tokens,
            "e2e_latency": e2e_latency,
            "avg_speed": total_tokens / e2e_latency if e2e_latency > 0 else 0,
            "additional_metrics": {
                "rank": self.rank,
            },
        }
        log_metrics(**metric_dict)

        return res

    def generate_until_multi_round(self, requests: List[Instance]) -> List[str]:
        """
        Generate responses for multi-round conversations.

        Note: This method is not yet implemented for InternVL2.5.

        Args:
            requests: List of Instance objects containing multi-round requests

        Raises:
            NotImplementedError: Multi-round generation is not yet implemented
        """
        raise NotImplementedError(
            "Multi-round generation is not yet implemented for InternVL2.5"
        )

    def loglikelihood(
        self, requests: List[Instance]
    ) -> List[tuple[float, bool]]:
        """
        Compute log-likelihood for multiple choice tasks.
        """
        eval_logger.warning(
            "loglikelihood is not fully implemented for InternVL2.5. "
            "Returning dummy values."
        )
        # Return dummy values for now
        return [(0.0, False) for _ in requests]

    def flatten(self, input_list):
        """Flatten a nested list."""
        output = []
        for item in input_list:
            if isinstance(item, list):
                output.extend(self.flatten(item))
            else:
                output.append(item)
        return output
