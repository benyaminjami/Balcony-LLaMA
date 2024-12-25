#!/bin/bash

# Load configuration variables
CONFIG_FILE_PATH=$1
if [ -f "$CONFIG_FILE_PATH" ]; then
    source "$CONFIG_FILE_PATH"
else
    echo "Error: Configuration file $CONFIG_FILE_PATH not found!"
    exit 1
fi


# Path to the sync script
SYNC_SCRIPT="/work/benyamin/smollm/scripts/sync.sh"
CONDA_SCRIPT="/work/benyamin/smollm/scripts/conda.sh"
ENV_SCRIPT="/work/benyamin/smollm/scripts/env.sh; pip install wandb /work/benyamin/smollm/nanotron;"

LOG_BASE_PATH="/work/benyamin/smollm/logs"
RUN_PATH="$LOG_BASE_PATH/$RUN_NAME"

CONDA_COMMAND="source ~/.bashrc || true; source ~/miniconda3/etc/profile.d/conda.sh || true;"


# Function to sync folders with all specified nodes
sync_folders() {
    echo "============================ Syncing folders to all nodes ============================"
    for i in "${!NODES[@]}"; do
        echo "---------------------------- Syncing Node ${NODES[$i]} ----------------------------"
        if [ "${NODES[$i]}" -eq "$MASTER_NODE" ] && [ "$SYNC_MASTER" = false ]; then
            echo "Skipping sync on master node lux-3-node-0$MASTER_NODE."
            continue
        fi
        NODE_ADDRESS="${NODES_ADDRESS[$i]}"
        echo "Ensuring target directory exists on $USER@$NODE_ADDRESS..."
        ssh -i /work/benyamin/smollm/ssh/id_ed25519 $USER@$NODE_ADDRESS "mkdir -p /work/benyamin/smollm"
        if [ $? -ne 0 ]; then
            echo "Error: Failed to create directory on $USER@$NODE_ADDRESS!"
            exit 1
        fi
        echo "Syncing with $USER@$NODE_ADDRESS (Node Number: ${NODES[$i]})..."
        bash "$SYNC_SCRIPT" "${NODES[$i]}"
        if [ $? -ne 0 ]; then
            echo "Error: Syncing with $USER@$NODE_ADDRESS failed!"
            exit 1
        fi
    done
    echo "============================ Folder sync complete for all nodes ============================"
}

# Function to check if conda is installed and install it if not
check_conda() {
    echo "============================ Checking if conda is installed ============================"
    for i in "${!NODES[@]}"; do
        echo "---------------------------- Checking Node ${NODES[$i]} ----------------------------"
        if [ "${NODES[$i]}" -eq "$MASTER_NODE" ] && [ "$SYNC_MASTER" = false ]; then
            echo "Skipping conda check on master node lux-3-node-0$MASTER_NODE."
            continue
        fi
        NODE_ADDRESS="${NODES_ADDRESS[$i]}"
        echo "Checking conda on $USER@$NODE_ADDRESS..."
        ssh -i /work/benyamin/smollm/ssh/id_ed25519 $USER@$NODE_ADDRESS "$CONDA_COMMAND conda --version" > /dev/null 2>&1
        if [ $? -ne 0 ]; then
            echo "Conda not found on $USER@$NODE_ADDRESS. Running conda.sh..."
            ssh -i /work/benyamin/smollm/ssh/id_ed25519 $USER@$NODE_ADDRESS "bash -s" < "$CONDA_SCRIPT"
            if [ $? -ne 0 ]; then
                echo "Error: Conda installation failed on $USER@$NODE_ADDRESS!"
                exit 1
            fi
        else
            echo "Conda is already installed on $USER@$NODE_ADDRESS."
        fi
    done
    echo "============================ Conda check complete for all nodes ============================"
}

# Function to check if the conda environment exists and create it if not
check_conda_env() {
    echo "============================ Checking conda environment 'smol' ============================"
    for i in "${!NODES[@]}"; do
        echo "---------------------------- Checking Node ${NODES[$i]} ----------------------------"
        if [ "${NODES[$i]}" -eq "$MASTER_NODE" ] && [ "$SYNC_MASTER" = false ]; then
            echo "Skipping conda environment check on master node lux-3-node-0$MASTER_NODE."
            continue
        fi
        NODE_ADDRESS="${NODES_ADDRESS[$i]}"
        echo "Checking conda environment on $USER@$NODE_ADDRESS..."
        ssh -i /work/benyamin/smollm/ssh/id_ed25519 $USER@$NODE_ADDRESS "$CONDA_COMMAND conda env list | grep -w smol" > /dev/null 2>&1
        if [ $? -ne 0 ]; then
            echo "Conda environment 'smol' not found on $USER@$NODE_ADDRESS. Running env.sh..."
            ssh -i /work/benyamin/smollm/ssh/id_ed25519 $USER@$NODE_ADDRESS "bash -s" < "$ENV_SCRIPT"
            if [ $? -ne 0 ]; then
                echo "Error: Conda environment setup failed on $USER@$NODE_ADDRESS!"
                exit 1
            fi
        else
            echo "Conda environment 'smol' exists on $USER@$NODE_ADDRESS."
        fi
    done
    echo "============================ Conda environment check complete ============================"
}


# Function to start training on all nodes
start_training() {
    echo "============================ Starting training on all nodes ============================"
    for i in "${!NODES[@]}"; do
        echo "---------------------------- Starting Training on Node ${NODES[$i]} ----------------------------"
        NODE_ADDRESS="${NODES_ADDRESS[$i]}"

        echo "Creating log directories on $USER@$NODE_ADDRESS..."
        ssh -i /work/benyamin/smollm/ssh/id_ed25519 $USER@$NODE_ADDRESS "mkdir -p $RUN_PATH"
        if [ $? -ne 0 ]; then
            echo "Error: Failed to create log directory on $USER@$NODE_ADDRESS!"
            exit 1
        fi

            # TORCH_DISTRIBUTED_DEBUG=DETAIL \
            # NCCL_DEBUG=INFO \
        LOG_PATH="$RUN_PATH/node_${NODES[$i]}.log"
            # export NCCL_BLOCKING_WAIT=1;
            # export NCCL_ASYNC_ERROR_HANDLING=1;
        LAUNCH_COMMAND="
            export CUDA_DEVICE_MAX_CONNECTIONS="1";
            torchrun \
            --nproc_per_node=8 \
            --nnodes=${#NODES[@]} \
            --master_addr=\"${NODES_ADDRESS[0]}\" \
            --node_rank=$i \
            --master_port=29502"
            # --max_restarts=0"

        
        TRAIN_COMMAND="
            $CONDA_COMMAND \
            conda activate smol; \
            export WANDB_API_KEY="44735889af01972b7a789b2a6b9d6f95ca6b9615"; \
            $LAUNCH_COMMAND \
            /work/benyamin/smollm/nanotron/run_train.py \
            --config-file $CONFIG_FILE " #--batch-accumulation-per-replica $GRAD_ACCUMULATION"

        FINAL_TRAIN_COMMAND="$TRAIN_COMMAND > $LOG_PATH 2>&1 &"
        echo $FINAL_TRAIN_COMMAND

        echo "Starting training on $USER@$NODE_ADDRESS with logs at $LOG_PATH..."
        ssh -i /work/benyamin/smollm/ssh/id_ed25519 $USER@$NODE_ADDRESS "$FINAL_TRAIN_COMMAND"
        if [ $? -ne 0 ]; then
            echo "Error: Training failed to start on $USER@$NODE_ADDRESS!"
            exit 1
        fi
    done
    echo "============================ Training started on all nodes ============================"
}


# Main script logic
main() {
    echo "============================ Starting multi-node setup ============================"

    # Sync folders across all nodes
    sync_folders

    # Check and install conda on all nodes if necessary
    check_conda

    # Check and create conda environment if necessary
    check_conda_env

    echo "============================ All nodes are prepared ============================"

    # Add additional logic for multi-node training setup and execution here

    start_training

    echo "============================ Multi-node training setup complete ============================"
}

# Execute the main function
main
