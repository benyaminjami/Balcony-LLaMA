#!/bin/bash

# Load configuration variables
CONFIG_FILE_PATH=$1
if [ -f "$CONFIG_FILE_PATH" ]; then
    source "$CONFIG_FILE_PATH"
else
    echo "Error: Configuration file $CONFIG_FILE_PATH not found!"
    exit 1
fi

# Function to stop training on all nodes
stop_training() {
    echo "============================ Stopping training on all nodes ============================"
    for i in "${!NODES[@]}"; do
        NODE_ADDRESS="${NODES_ADDRESS[$i]}"
        echo "Attempting to stop training on $USER@$NODE_ADDRESS..."
        ssh -i /work/benyamin/smollm/ssh/id_ed25519 $USER@$NODE_ADDRESS "pkill -f torchrun" 2>/dev/null
        if [ $? -ne 0 ]; then
            echo "Warning: No training process found or failed to stop on $USER@$NODE_ADDRESS. Skipping this node."
            continue
        fi
        echo "Training successfully stopped on $USER@$NODE_ADDRESS."
    done
    echo "============================ Training stopped on all nodes ============================"
}

# Stop training on all nodes
stop_training
