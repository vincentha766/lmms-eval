accelerate launch \
	--num_processes=8 \
	--main_process_port=12346 \
	-m lmms_eval \
	--model qwen2_5_vl \
	--model_args=pretrained=Qwen/Qwen2.5-VL-3B-Instruct,max_pixels=12845056,attn_implementation=flash_attention_2,interleave_visuals=False \
	--tasks mme \
	--batch_size 1
