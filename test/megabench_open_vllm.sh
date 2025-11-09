# accelerate launch \
# 	-m lmms_eval \
# 	--model qwen2_5_vl \
# 	--model_args=pretrained=Qwen/Qwen2.5-VL-3B-Instruct,max_pixels=12845056,interleave_visuals=False \
# 	--tasks megabench_open \
# 	--log_samples \
# 	--log_samples_suffix llava_ov_megabench_open \
# 	--output_path ./logs/ \
# 	--batch_size 1
# 
# 	#--num_processes=8 \
# 	#--model_args=pretrained=Qwen/Qwen2.5-VL-3B-Instruct,max_pixels=12845056,attn_implementation=flash_attention_2,interleave_visuals=False \

accelerate launch \
    -m lmms_eval \
    --model vllm \
    --model_args=model_version=Qwen/Qwen2.5-VL-3B-Instruct,data_parallel_size=4,max_pixels=12845056,interleave_visuals=False \
    --tasks megabench_open \
    --batch_size 300 \
    --log_samples \
    --log_samples_suffix vllm \
    --output_path ./logs/
