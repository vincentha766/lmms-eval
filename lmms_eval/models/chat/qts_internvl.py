from __future__ import annotations

import time
from typing import List, Optional, Union

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


@register_model("qts_internvl")
class QTSInternVL(lmms):
    """
    QTS-InternVL2.5 Model for video and image understanding tasks.
    Based on InternVL2.5 with QTS (Query Token Selection) optimization.
    """

    is_simple = False  # Use chat model type (recommended)

    def __init__(
        self,
        pretrained: str = "AlpachinoNLP/QTSplus-InternVL2.5-8B",
        device: str = "cuda",
        device_map: str = "",
        batch_size: int = 1,
        num_frames: int = 15,
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

        # Initialize processor
        eval_logger.info(f"Loading processor from {pretrained}")
        self.processor = AutoProcessor.from_pretrained(
            pretrained, trust_remote_code=trust_remote_code
        )
        self.tokenizer = self.processor.tokenizer

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
            f"QTS-InternVL model loaded successfully on {self.model.device}"
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

                # Prepare inputs based on modality
                if videos and len(videos) > idx and videos[idx]:
                    # Video input
                    inputs = self.processor(
                        text=text_prompt,
                        videos=videos[idx],
                        num_frames=self.num_frames,
                        return_tensors="pt",
                    )
                elif visuals and len(visuals) > idx and visuals[idx]:
                    # Image input
                    inputs = self.processor(
                        text=text_prompt,
                        images=visuals[idx],
                        return_tensors="pt",
                    )
                else:
                    # Text only input
                    inputs = self.processor(
                        text=text_prompt,
                        return_tensors="pt",
                    )

                # Move inputs to device
                for k, v in list(inputs.items()):
                    if isinstance(v, torch.Tensor):
                        inputs[k] = v.to(self._device)

                # Generate
                start_time = time.time()
                try:
                    with torch.inference_mode():
                        output_ids = self.model.generate(
                            **inputs,
                            max_new_tokens=current_gen_kwargs["max_new_tokens"],
                            do_sample=current_gen_kwargs["do_sample"],
                            temperature=current_gen_kwargs.get("temperature"),
                            eos_token_id=self.tokenizer.eos_token_id,
                            pad_token_id=self.tokenizer.pad_token_id,
                        )
                    end_time = time.time()

                    # Decode output
                    input_len = int(inputs["input_ids"].shape[1])
                    gen_ids = output_ids[0, input_len:]
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

        Note: This method is not yet implemented for QTS-InternVL.

        Args:
            requests: List of Instance objects containing multi-round requests

        Raises:
            NotImplementedError: Multi-round generation is not yet implemented
        """
        raise NotImplementedError(
            "Multi-round generation is not yet implemented for QTS-InternVL"
        )

    def loglikelihood(
        self, requests: List[Instance]
    ) -> List[tuple[float, bool]]:
        """
        Compute log-likelihood for multiple choice tasks.
        """
        eval_logger.warning(
            "loglikelihood is not fully implemented for QTS-InternVL. "
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
