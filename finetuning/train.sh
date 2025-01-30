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
# CONFIG_FILE=$1
export WANDB_API_KEY="44735889af01972b7a789b2a6b9d6f95ca6b9615"
export WANDB_PROJECT="Nested_FT"
CUDA_VISIBLE_DEVICES=0,1,2,3 ACCELERATE_LOG_LEVEL=info accelerate launch --main_process_port 29000 --config_file /home/parsa/projects/Nested-Llama/finetuning/ddp.yaml \
    train.py experiments/balcony_mlp.yaml > /home/parsa/projects/Nested-Llama/finetuning/logs/balcony_mlp.log 2>&1 &
