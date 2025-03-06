export WANDB_API_KEY="44735889af01972b7a789b2a6b9d6f95ca6b9615"
export WANDB_PROJECT="Nested_FT"

# Get the directory where the script is located
SCRIPT_DIR="$(dirname "$(realpath "$0")")"

# Extract experiment name from the YAML file path
EXPERIMENT_YAML=$1
EXPERIMENT_NAME=$(basename "$EXPERIMENT_YAML" .yaml)

ACCELERATE_LOG_LEVEL=info accelerate launch \
    --main_process_port 30000 \
    --config_file "$SCRIPT_DIR/deepspeed_zero3.yaml" \
    "$SCRIPT_DIR/finetuning/train.py" "$EXPERIMENT_YAML" > \
    "$SCRIPT_DIR/logs/${EXPERIMENT_NAME}.log" 2>&1 &
