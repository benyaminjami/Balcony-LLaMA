#!/bin/bash

# Define the base directory for checkpoints
CHECKPOINT_DIR="/work/benyamin/smollm/finetuning/checkpoints"

# Define the list of checkpoints
# CHECKPOINTS=("Smollm_1B_Frozen_Sorted_Balcony" "Smollm_1B_Sorted" "Smollm_1B_Sorted_SFT_Tie" "Smollm_1B_Sorted_SFT_Untie" "Smollm_1B_Unfrozen_Sorted_Balcony") #"Smollm_1B" )

CHECKPOINTS=("Smollm_1B_balcony" "Smollm_1B_balcony_unfrozen") 

# Define the output_exit_layer_index value
EXIT_LAYER_INDICES=(4 8 12 16)

# Create logs directory if it doesn't exist
mkdir -p logs

# Initialize GPU counter
GPU=0

for EXIT_LAYER_INDEX in "${EXIT_LAYER_INDICES[@]}"; do
# Iterate over each checkpoint
    for CHECKPOINT in "${CHECKPOINTS[@]}"; do
        # Construct the output path
        OUTPUT_PATH="${CHECKPOINT}_exit_${EXIT_LAYER_INDEX}"
        
        # Run the lm_eval command in the background and log the output
        lm_eval --model hf \
                --model_args pretrained="${CHECKPOINT_DIR}/${CHECKPOINT}/checkpoint-30000,output_full_model=False,output_exit_layers=${EXIT_LAYER_INDEX},trust_remote_code=True" \
                --num_fewshot 0 \
                --tasks hellaswag,winogrande,ai2_arc,boolq,piqa,openbookqa,mmlu \
                --device cuda:${GPU} \
                --batch_size 64 \
                --output_path "results/exit_${EXIT_LAYER_INDEX}/" > logs/${OUTPUT_PATH}.log 2>&1 &
        
        # Increment GPU counter
        GPU=$((GPU + 1))
        
        # Reset GPU counter if it exceeds 7
        if [ ${GPU} -gt 7 ]; then
            GPU=0
        fi
    done
done