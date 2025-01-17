# accelerate launch train.py \
#         --model_id "HuggingFaceTB/SmolLM2-1.7B-Instruct" \
#         --dataset_name "bigcode/the-stack-smol" \
#         --subset "data/python" \
#         --dataset_text_field "content" \
#         --split "train" \
#         --max_seq_length 2048 \
#         --max_steps 5000 \
#         --micro_batch_size 1 \
#         --gradient_accumulation_steps 8 \
#         --learning_rate 3e-4 \
#         --warmup_steps 100 \
#         --num_proc "$(nproc)"

ACCELERATE_LOG_LEVEL=info accelerate launch --config_file /work/benyamin/smollm/finetuning/finetuning/deepspeed_zero3.yaml train.py experiments/test.yaml