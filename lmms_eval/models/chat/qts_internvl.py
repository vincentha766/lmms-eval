from __future__ import annotations

import time
from typing import List

import torch
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
        device: str = "cuda:0",
        device_map: str = "auto",
        batch_size: int = 1,
        num_frames: int = 15,
        use_cache: bool = True,
        trust_remote_code: bool = True,
        **kwargs,
    ) -> None:
        super().__init__()

        self.pretrained = pretrained
        self.device = device
        self.device_map = device_map
        self.batch_size = int(batch_size)
        self.num_frames = num_frames
        self.use_cache = use_cache
        self.trust_remote_code = trust_remote_code

        # Initialize processor
        eval_logger.info(f"Loading processor from {pretrained}")
        self.processor = AutoProcessor.from_pretrained(
            pretrained, trust_remote_code=trust_remote_code
        )

        # Determine dtype
        self.dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

        # Initialize model
        eval_logger.info(f"Loading model from {pretrained}")
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                pretrained,
                trust_remote_code=trust_remote_code,
                dtype=self.dtype,
                device_map=device_map if torch.cuda.is_available() else None,
                low_cpu_mem_usage=True,
            ).eval()
        except TypeError:
            # Fallback for older transformers versions
            self.model = AutoModelForCausalLM.from_pretrained(
                pretrained,
                trust_remote_code=trust_remote_code,
                torch_dtype=self.dtype,
                device_map=device_map if torch.cuda.is_available() else None,
                low_cpu_mem_usage=True,
            ).eval()

        self.tokenizer = self.processor.tokenizer

        eval_logger.info(
            f"QTS-InternVL model loaded successfully on {self.model.device}"
        )

    def generate_until(self, requests: List[Instance]) -> List[str]:
        res = []

        def _collate(x):
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
        pbar = tqdm(total=num_iters, disable=False, desc="Model Responding")
        e2e_latency = 0
        total_tokens = 0

        for chunk in chunks:
            ctx, doc_to_messages, all_gen_kwargs, doc_id, task, split = zip(*chunk)
            chat_messages = [
                doc_to_messages[idx](self.task_dict[task][split][ids])
                for idx, (ids, task, split) in enumerate(zip(doc_id, task, split))
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
                        inputs[k] = v.to(self.model.device)

                # Generate
                start_time = time.time()
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

                batch_outputs.append(answer)

                # Track metrics
                e2e_latency += end_time - start_time
                total_tokens += len(gen_ids)

            res.extend(batch_outputs)
            pbar.update(1)

        pbar.close()

        # Log metrics
        if len(res) > 0 and total_tokens > 0:
            avg_speed = total_tokens / e2e_latency if e2e_latency > 0 else 0.0
            log_metrics(
                e2e_latency=e2e_latency,
                total_tokens=total_tokens,
                avg_speed=avg_speed,
            )

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
