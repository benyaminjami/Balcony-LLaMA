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
export WANDB_API_KEY="44735889af01972b7a789b2a6b9d6f95ca6b9615"
export WANDB_PROJECT="Nested_FT"
ACCELERATE_LOG_LEVEL=info accelerate launch \
    --config_file /work/benyamin/smollm/finetuning/finetuning/ddp.yaml \
    train.py experiments/balcony.yaml > /work/benyamin/smollm/finetuning/logs/balcony.log 2>&1 &