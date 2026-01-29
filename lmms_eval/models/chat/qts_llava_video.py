"""
QTSplus-LLaVA-Video Model Implementation

This module implements the QTSplus-LLaVA-Video-7B-Qwen2 model for video understanding tasks.
The model uses a standard HuggingFace transformers interface and supports video inputs
through frame extraction and processing.

Example usage:
```
# Single GPU
python -m lmms_eval \
    --model qts_llava_video \
    --model_args pretrained=AlpachinoNLP/QTSplus-LLaVA-Video-7B-Qwen2,fps=1.0,max_frames=15 \
    --tasks videomme,mlvu \
    --batch_size 1 \
    --device cuda:0

# Multi-GPU with accelerate
accelerate launch --num_processes=8 -m lmms_eval \
    --model qts_llava_video \
    --model_args pretrained=AlpachinoNLP/QTSplus-LLaVA-Video-7B-Qwen2,fps=1.0,max_frames=15 \
    --tasks videomme,mlvu \
    --batch_size 1
```
"""

from __future__ import annotations

import json
import time
from typing import List, Optional, Tuple, Union

import torch
from accelerate import Accelerator, DistributedType
from accelerate.state import AcceleratorState
from loguru import logger as eval_logger
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoProcessor

from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from lmms_eval.models.model_utils.gen_metrics import log_metrics
from lmms_eval.protocol import ChatMessages


@register_model("qts_llava_video")
class QTSLlavaVideo(lmms):
    """
    QTSplus-LLaVA-Video-7B-Qwen2 Model

    A video understanding model based on LLaVA architecture with Qwen2 backbone.
    Supports video inputs through frame extraction at configurable FPS and frame limits.

    The model uses a chat-based interface where videos are represented with <image> tokens
    in the prompt (following LLaVA-Video conventions).

    Args:
        pretrained: Path or name of the pretrained model
        device: Device to run the model on (default: "cuda")
        device_map: Device map for model parallelism (default: "" for auto-assignment in multi-GPU)
        batch_size: Batch size for inference (default: 1)
        fps: Frames per second for video sampling (default: 1.0)
        max_frames: Maximum number of frames to extract from video (default: 15)
        max_new_tokens: Maximum number of new tokens to generate (default: 256)
        use_cache: Whether to use KV cache during generation (default: True)
    """

    is_simple = False  # Using chat model type

    def __init__(
        self,
        pretrained: str = "AlpachinoNLP/QTSplus-LLaVA-Video-7B-Qwen2",
        device: Optional[str] = "cuda",
        device_map: Optional[str] = "",
        batch_size: Optional[Union[int, str]] = 1,
        fps: float = 1.0,
        max_frames: int = 15,
        max_new_tokens: int = 256,
        use_cache: bool = True,
        trust_remote_code: bool = True,
        **kwargs,
    ) -> None:
        """Initialize the QTSplus-LLaVA-Video model."""
        super().__init__()

        # Initialize accelerator for distributed training support
        accelerator = Accelerator()

        # Determine device and device_map based on distributed setup
        if accelerator.num_processes > 1 and device_map == "":
            # Multi-process mode: assign each process to its own GPU
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"
        else:
            # Single process mode or custom device_map
            self._device = torch.device(device)
            self.device_map = device_map

        # Determine dtype based on device availability
        load_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

        eval_logger.info(f"Loading QTSplus-LLaVA-Video model from {pretrained}")
        eval_logger.info(f"Using dtype: {load_dtype}, device_map: {self.device_map}")

        # Load model with device_map
        try:
            self._model = AutoModelForCausalLM.from_pretrained(
                pretrained,
                trust_remote_code=trust_remote_code,
                dtype=load_dtype,
                device_map=self.device_map,
                low_cpu_mem_usage=True,
            ).eval()
        except TypeError:
            # Fallback to torch_dtype if dtype is not supported
            self._model = AutoModelForCausalLM.from_pretrained(
                pretrained,
                trust_remote_code=trust_remote_code,
                torch_dtype=load_dtype,
                device_map=self.device_map,
                low_cpu_mem_usage=True,
            ).eval()

        # Load processor
        self.processor = AutoProcessor.from_pretrained(
            pretrained,
            trust_remote_code=trust_remote_code,
        )

        # Set model parameters
        self.fps = fps
        self.max_frames = max_frames
        self.max_new_tokens = max_new_tokens
        self.batch_size_per_gpu = int(batch_size)
        self.use_cache = use_cache
        self.trust_remote_code = trust_remote_code

        # Handle distributed setup
        if accelerator.num_processes > 1 and device_map == "":
            assert accelerator.distributed_type in [
                DistributedType.FSDP,
                DistributedType.MULTI_GPU,
                DistributedType.DEEPSPEED,
            ], "Unsupported distributed type provided. Only DDP, FSDP and DEEPSPEED are supported."

            if accelerator.distributed_type == DistributedType.DEEPSPEED:
                kwargs = {
                    "train_micro_batch_size_per_gpu": self.batch_size_per_gpu,
                    "train_batch_size": self.batch_size_per_gpu * accelerator.num_processes,
                }
                AcceleratorState().deepspeed_plugin.deepspeed_config_process(must_match=True, **kwargs)
                eval_logger.info("Detected DistributedType.DEEPSPEED. Make sure you run `accelerate config` and set zero stage to 0")

            if accelerator.distributed_type == DistributedType.FSDP or accelerator.distributed_type == DistributedType.DEEPSPEED:
                self._model = accelerator.prepare(self.model)
            else:
                self._model = accelerator.prepare_model(self.model, evaluation_mode=True)

            self.accelerator = accelerator
            if self.accelerator.is_local_main_process:
                eval_logger.info(f"Using {accelerator.num_processes} devices with data parallelism")
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        elif accelerator.num_processes == 1 and device_map == "auto":
            eval_logger.info(f"Using {accelerator.num_processes} devices with pipeline parallelism")
            self._rank = 0
            self._world_size = 1
        else:
            eval_logger.info(f"Using single device: {self._device}")
            self.model.to(self._device)
            self._rank = 0
            self._world_size = 1

        self.accelerator = accelerator

        eval_logger.info(
            f"Model initialized: fps={fps}, max_frames={max_frames}, "
            f"rank={self._rank}, world_size={self._world_size}"
        )

    @property
    def model(self):
        """Return the underlying model, unwrapping it if using Accelerate."""
        if hasattr(self, "accelerator"):
            return self.accelerator.unwrap_model(self._model)
        else:
            return self._model

    @property
    def tokenizer(self):
        """Return the tokenizer from the processor."""
        return self.processor.tokenizer

    @property
    def eot_token_id(self):
        """Return the end-of-text token ID."""
        return self.tokenizer.eos_token_id

    @property
    def batch_size(self):
        """Return the batch size."""
        return self.batch_size_per_gpu

    @property
    def device(self):
        """Return the device."""
        return self._device

    @property
    def rank(self):
        """Return the rank (process rank in distributed mode)."""
        return self._rank

    @property
    def world_size(self):
        """Return the world size (number of processes)."""
        return self._world_size

    def flatten(self, input_list: List[List]) -> List:
        """
        Flatten a nested list into a single list.

        Args:
            input_list: Nested list to flatten

        Returns:
            Flattened list
        """
        new_list = []
        for sublist in input_list:
            for item in sublist:
                new_list.append(item)
        return new_list

    def generate_until(self, requests: List[Instance]) -> List[str]:
        """
        Generate responses for the given requests.

        This method processes video inputs by extracting frames at the specified FPS
        and max_frames limit, then generates text responses using the model.

        Args:
            requests: List of Instance objects containing generation requests

        Returns:
            List of generated text responses
        """
        res = []

        def _collate(x):
            """Collate function for sorting by doc id."""
            return x[0], x[0]

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
        pbar = tqdm(total=num_iters, disable=(self.rank != 0), desc="Model Responding")

        e2e_latency = 0
        total_tokens = 0

        for chunk in chunks:
            (
                ctx,
                doc_to_messages,
                all_gen_kwargs,
                doc_id,
                task,
                split,
            ) = zip(*chunk)

            # Process chat messages
            chat_messages = [
                doc_to_messages[idx](self.task_dict[task_name][split_name][doc_idx])
                for idx, (doc_idx, task_name, split_name) in enumerate(
                    zip(doc_id, task, split)
                )
            ]
            chat_messages: List[ChatMessages] = [
                ChatMessages(**{"messages": message}) for message in chat_messages
            ]

            # Extract media from messages
            visuals = []
            videos = []
            for messages in chat_messages:
                visual, video, _ = messages.extract_media()
                visuals.append(visual)
                videos.append(video)
            visuals = self.flatten(visuals)
            videos = self.flatten(videos)

            # Get generation kwargs (assume all in batch are the same)
            gen_kwargs = all_gen_kwargs[0]

            # Convert messages to HuggingFace format and apply chat template
            batched_messages = [
                chat_message.to_hf_messages() for chat_message in chat_messages
            ]

            # Convert content from list format to string format for this model's chat template
            # The model expects content as string with <image> tokens, not as a list
            for msg_list in batched_messages:
                for msg in msg_list:
                    if isinstance(msg.get("content"), list):
                        # Convert list of content items to a single string
                        content_parts = []
                        for item in msg["content"]:
                            if item.get("type") == "text":
                                content_parts.append(item["text"])
                            elif item.get("type") in ["image", "video"]:
                                # Use <image> token for both images and videos (LLaVA-Video convention)
                                content_parts.append("<image>")
                        msg["content"] = "\n".join(content_parts)

            texts = [
                self.processor.tokenizer.apply_chat_template(
                    msg, tokenize=False, add_generation_prompt=True
                )
                for msg in batched_messages
            ]

            # Log first prompt for debugging
            if self.accelerator.is_main_process and doc_id[0] % 100 == 0:
                eval_logger.debug(f"Prompt for doc ID {doc_id[0]}:\n{texts[0]}\n")

            # Process inputs with processor
            # The processor expects a single video path string (not a list)
            # when processing video files. If we have a list, take the first video.
            video_input = None
            if len(videos) > 0:
                # For batch_size=1, we should only have one video
                # The processor expects a string path for video files
                video_input = videos[0] if isinstance(videos, list) else videos

            inputs = self.processor(
                text=texts,
                videos=video_input,
                fps=self.fps,
                max_frames=self.max_frames,
                return_tensors="pt",
                padding=False,
            )

            # Move inputs to device and dtype
            inputs = inputs.to(self._device, self.model.dtype)

            # Set up generation parameters
            default_gen_kwargs = {
                "max_new_tokens": self.max_new_tokens,
                "temperature": 0.0,
                "top_p": None,
                "num_beams": 1,
            }
            current_gen_kwargs = {**default_gen_kwargs, **gen_kwargs}

            # Configure sampling based on temperature
            if current_gen_kwargs["temperature"] > 0:
                current_gen_kwargs["do_sample"] = True
            else:
                current_gen_kwargs["do_sample"] = False
                current_gen_kwargs["temperature"] = None
                current_gen_kwargs["top_p"] = None

            # Generate response
            try:
                start_time = time.time()
                with torch.inference_mode():
                    output_ids = self.model.generate(
                        **inputs,
                        max_new_tokens=current_gen_kwargs["max_new_tokens"],
                        do_sample=current_gen_kwargs["do_sample"],
                        temperature=current_gen_kwargs["temperature"],
                        top_p=current_gen_kwargs["top_p"],
                        num_beams=current_gen_kwargs["num_beams"],
                        use_cache=self.use_cache,
                        pad_token_id=self.eot_token_id,
                        eos_token_id=self.eot_token_id,
                    )
                end_time = time.time()

                # Extract generated tokens (remove input)
                input_len = int(inputs["input_ids"].shape[1])
                generated_ids_trimmed = [
                    out_ids[input_len:] for out_ids in output_ids
                ]

                # Decode responses
                answers = self.processor.tokenizer.batch_decode(
                    generated_ids_trimmed,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )

                # Calculate timing metrics
                e2e_latency += end_time - start_time
                total_tokens += sum(len(ids) for ids in generated_ids_trimmed)

            except Exception as e:
                eval_logger.error(f"Error during generation: {e}")
                answers = [""] * len(texts)
                e2e_latency += 0
                total_tokens += 0

            # Process and store results
            for ans, text in zip(answers, texts):
                # Strip whitespace from answer
                clean_ans = ans.strip()
                res.append(clean_ans)
                self.cache_hook.add_partial("generate_until", (text, gen_kwargs), clean_ans)

                # Log for debugging
                if self.accelerator.is_main_process:
                    eval_logger.debug(f"Question: {text}")
                    eval_logger.debug(f"Model Response: {clean_ans}")

            pbar.update(1)

        # Reorder results back to original form
        res = re_ords.get_original(res)

        # Log metrics
        avg_speed = total_tokens / e2e_latency if e2e_latency > 0 else 0
        metric_dict = {
            "total_tokens": total_tokens,
            "e2e_latency": e2e_latency,
            "avg_speed": avg_speed,
            "additional_metrics": {
                "rank": self.rank,
            },
        }
        log_metrics(**metric_dict)

        pbar.close()
        return res

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        """
        Compute log-likelihood for the given requests.

        Note: This method is not implemented for QTSLlavaVideo as the model
        is primarily designed for generation tasks.

        Args:
            requests: List of Instance objects containing loglikelihood requests

        Raises:
            NotImplementedError: This method is not implemented
        """
        raise NotImplementedError(
            "Loglikelihood is not implemented for QTSLlavaVideo. "
            "This model is designed for generation tasks only."
        )

    def generate_until_multi_round(self, requests: List[Instance]) -> List[str]:
        """
        Generate responses for multi-round conversations.

        Note: This method is not yet implemented.

        Args:
            requests: List of Instance objects containing multi-round requests

        Raises:
            NotImplementedError: This method is not yet implemented
        """
        raise NotImplementedError("Multi-round generation is not yet implemented")
