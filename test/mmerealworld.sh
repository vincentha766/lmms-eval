accelerate launch \
	--main_process_port=12346 \
	-m lmms_eval \
	--model qwen2_5_vl \
	--model_args=pretrained=Qwen/Qwen2.5-VL-3B-Instruct,max_pixels=376320,interleave_visuals=False \
	--tasks mmerealworld \
	--batch_size 1
