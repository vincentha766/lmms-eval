accelerate launch --main_process_port=12346 -m lmms_eval \
  --model qts_plus_3b \
  --model_args "pretrained=/home/huanan/work/QTSplusQwenVL2_5-3B/QTSplusQwenVL2_5,max_pixels=376320" \
	--tasks megabench_open \
  --batch_size 1 \
  --use_cache cache
