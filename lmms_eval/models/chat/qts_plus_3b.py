import time
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from loguru import logger as eval_logger
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoProcessor, AutoConfig, AutoTokenizer

from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from lmms_eval.models.model_utils.gen_metrics import log_metrics
from lmms_eval.models.model_utils.reasoning_model_utils import (
    parse_reasoning_model_answer,
)
from lmms_eval.models.simple.qwen2_5_vl import Qwen2_5_VL as Qwen2_5_VLSimple
from lmms_eval.protocol import ChatMessages

try:
    from qwen_vl_utils import process_vision_info
except ImportError:
    eval_logger.warning("Failed to import qwen_vl_utils; Please install it via `pip install qwen-vl-utils`")


@register_model("qts_plus_3b")
class QTSPlus3B(lmms):
    """
    QTS+ 3B Model
    Based on Qwen2.5-VL-3B-Instruct architecture with custom video processing

    This model extends the Qwen2.5-VL base model with improved video processing capabilities.
    It supports image, video and text inputs in a chat-like interface.

    Example usage:
    ```
    python -m lmms_eval --model qts_plus_3b --model_args pretrained=/path/to/model,max_pixels=12845056 --tasks mmmu,mme --batch_size 8 --limit 10
    ```

    For video evaluation tasks, the model processes video frames at a configurable rate and
    handles multi-modal messages following the chat model interface pattern.
    """
    is_simple = False  # Using chat model type

    def __init__(
        self,
        pretrained: str = "AlpachinoNLP/QTSplusQwenVL2_5-3B",
        device: Optional[str] = "cuda",
        device_map: Optional[str] = "auto",
        batch_size: Optional[Union[int, str]] = 1,
        min_pixels: int = 256 * 28 * 28,
        max_pixels: int = 1605632,
        max_num_frames: int = 32,
        fps: Optional[float] = None,  # Frames per second for video processing
        system_prompt: Optional[str] = "You are a helpful assistant.",
        use_cache: bool = True,
    ) -> None:
        super().__init__()

        self._device = torch.device(device)
        self.device_map = device_map if device_map else device

        # Load the model
        self._model = AutoModelForCausalLM.from_pretrained(
            pretrained,
            trust_remote_code=True,
            local_files_only=True,
        ).to(dtype=torch.bfloat16, device=device)
        self._model.eval()

        # Load processor and tokenizer
        self.processor = AutoProcessor.from_pretrained(
            pretrained,
            trust_remote_code=True,
            local_files_only=True,
            use_fast=True,
        )
        self._tokenizer = AutoTokenizer.from_pretrained(pretrained, trust_remote_code=True)

        # Set model parameters
        self.max_pixels = max_pixels
        self.min_pixels = min_pixels
        self.max_num_frames = max_num_frames
        self.fps = fps  # Added fps parameter
        self.system_prompt = system_prompt
        self.batch_size_per_gpu = int(batch_size)
        self.use_cache = use_cache

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def model(self):
        return self._model

    @property
    def eot_token_id(self):
        return self.tokenizer.eos_token_id

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    @property
    def device(self):
        return self._device

    @property
    def rank(self):
        return 0  # Single-process implementation

    def flatten(self, input):
        """Flatten a nested list into a single list."""
        new_list = []
        for i in input:
            for j in i:
                new_list.append(j)
        return new_list

    def generate_until(self, requests: List[Instance]) -> List[str]:
        """
        Generate responses for the given requests using QTS+ model.

        This implementation is based on demo.py and supports both image and video inputs.
        The method handles QTS+ specific video processing with features like:
        - Frame extraction and sampling for videos
        - Custom video grid formatting
        - Separate question_input_ids for improved context understanding
        """
        res = []

        # A dummy collate here to sort by doc id
        def _collate(x):
            return x[0], x[0]

        # we group requests by their generation_kwargs,
        # so that we don't try to execute e.g. greedy sampling and temp=0.8 sampling
        # in the same batch.
        re_ords = utils.Collator([reg.args for reg in requests], _collate, group_fn=lambda x: x[2], grouping=True)
        chunks = re_ords.get_batched(n=self.batch_size, batch_fn=None)
        num_iters = len(requests) // self.batch_size if len(requests) % self.batch_size == 0 else len(requests) // self.batch_size + 1
        pbar = tqdm(total=num_iters, disable=(self.rank != 0), desc="Model Responding")
        e2e_latency = 0
        total_tokens = 0
        for chunk in chunks:
            ctx, doc_to_messages, all_gen_kwargs, doc_id, task, split = zip(*chunk)
            chat_messages = [doc_to_messages[idx](self.task_dict[task][split][ids]) for idx, (ids, task, split) in enumerate(zip(doc_id, task, split))]
            chat_messages: List[ChatMessages] = [ChatMessages(**{"messages": message}) for message in chat_messages]
            visuals = []
            videos = []
            for messages in chat_messages:
                visual, video, _ = messages.extract_media()
                visuals.append(visual)
                videos.append(video)
            visuals = self.flatten(visuals)
            videos = self.flatten(videos)
            gen_kwargs = all_gen_kwargs[0]

            # Apply chat template
            video_kwargs = {
                "max_pixels": self.max_pixels,
                "min_pixels": self.min_pixels,
            }
            if self.fps is not None:
                video_kwargs["fps"] = self.fps
            else:
                video_kwargs["nframes"] = self.max_num_frames
            batched_messages = [chat_message.to_hf_messages(video_kwargs=video_kwargs) for chat_message in chat_messages]
            texts = [self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in batched_messages]
            # Extract vision information from the message with additional video kwargs
            image_inputs, video_inputs, video_kwargs = process_vision_info(batched_messages, return_video_kwargs=True)
            if video_inputs is not None:
                total_frames = video_inputs[0].shape[0]
                indices = np.linspace(0, total_frames - 1, self.max_num_frames, dtype=int)
                # Ensure unique indices if linspace produces duplicates for few frames
                indices = np.unique(indices)
                # Append the last frame index if not already included
                if total_frames - 1 not in indices:
                    indices = np.append(indices, total_frames - 1)
                    indices = np.unique(indices)  # Ensure uniqueness again
                video_inputs[0] = video_inputs[0][indices]

            # Create model inputs
            inputs = self.processor(
                text=texts,
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
                **video_kwargs
            )

            if self.device_map == "auto":
                inputs = inputs.to("cuda")
            else:
                inputs = inputs.to(self.device)

            # Set default generation kwargs
            default_gen_kwargs = {
                "max_new_tokens": 128,
                "temperature": 0.0,  # Set to 0 for greedy default
                "top_p": None,
                "num_beams": 1,
            }
            # Update with provided kwargs
            current_gen_kwargs = {**default_gen_kwargs, **gen_kwargs}
            pad_token_id = self.tokenizer.pad_token_id

            if current_gen_kwargs["temperature"] > 0:
                current_gen_kwargs["do_sample"] = True
            else:
                current_gen_kwargs["do_sample"] = False
                current_gen_kwargs["temperature"] = None
                current_gen_kwargs["top_p"] = None
                current_gen_kwargs["top_k"] = None

            # Extract and format video-specific inputs for QTS+ model
            pixel_values_videos = inputs.pop('pixel_values_videos', None)
            video_grid_thw = inputs.pop('video_grid_thw', None)
            inputs.pop('second_per_grid_ts', None)  # Remove unused parameter

            # Format vision input as expected by QTS+ model
            vision_input = None
            if pixel_values_videos is not None and video_grid_thw is not None:
                vision_input = {
                    'pixel_values_videos': pixel_values_videos,
                    'video_grid_thw': video_grid_thw
                }

            # Build question_input_ids from the textual question only
            question_input_ids = None
            if self.tokenizer is not None:
                question_texts = []
                for batch_msg in batched_messages:
                    for msg in batch_msg:
                        if msg.get("role") == "user" and isinstance(msg.get("content"), list):
                            for c in msg["content"]:
                                if isinstance(c, dict) and c.get("type") == "text" and isinstance(c.get("text"), str):
                                    question_texts.append(c["text"])
                if question_texts:
                    qt = "\n".join(question_texts)
                    enc = self.tokenizer(qt, add_special_tokens=False, return_tensors="pt")
                    question_input_ids = enc.input_ids.to(self.device)

            start_time = time.time()
            # Generate with QTS+ model-specific parameters
            cont = self.model.generate(
                vision_input=vision_input,
                input_ids=inputs.input_ids,
                question_input_ids=question_input_ids if question_input_ids is not None else inputs.input_ids,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=pad_token_id,
                do_sample=current_gen_kwargs["do_sample"],
                temperature=current_gen_kwargs["temperature"],
                top_p=current_gen_kwargs["top_p"],
                num_beams=current_gen_kwargs["num_beams"],
                max_new_tokens=current_gen_kwargs["max_new_tokens"],
                top_k=current_gen_kwargs.get("top_k", None),
                use_cache=self.use_cache,
            )
            end_time = time.time()

            generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, cont)]
            answers = self.processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)

            # Calculate timing metrics for batch
            e2e_latency += end_time - start_time
            total_tokens += sum(len(ids) for ids in generated_ids_trimmed)

            for ans, context in zip(answers, texts):
                clean_ans = parse_reasoning_model_answer(ans)
                res.append(clean_ans)
                self.cache_hook.add_partial("generate_until", (context, gen_kwargs), clean_ans)
                pbar.update(1)

                eval_logger.debug(f"Question: {context}")
                eval_logger.debug(f"Model Raw Response: {ans}")
                eval_logger.debug(f"Model Clean Response: {clean_ans}")
            # reorder this group of results back to original unsorted form
        res = re_ords.get_original(res)

        # Calculate average speed
        avg_speed = total_tokens / e2e_latency if e2e_latency > 0 else 0
        # Log metrics
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
        raise NotImplementedError("Loglikelihood is not implemented for QTSPlus3B")

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("TODO: Implement multi-round generation")

