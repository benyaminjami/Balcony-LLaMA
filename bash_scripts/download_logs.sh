#!/bin/bash

# Define the node numbers
NODES=(4 5 7 8 9 10)  # List of node numbers

# Define user and node addresses
USER="benyamin"
NODES_ADDRESS=()
for NODE_NUMBER in "${NODES[@]}"; do
    if [ "$NODE_NUMBER" -lt 10 ]; then
        NODES_ADDRESS+=("lux-3-node-0$NODE_NUMBER")
    else
        NODES_ADDRESS+=("lux-3-node-$NODE_NUMBER")
    fi
done

# Log directory path
LOG_BASE_PATH="/work/benyamin/smollm/logs"

# Function to download logs from all nodes
download_logs() {
    echo "============================ Downloading logs from all nodes ============================"
    for i in "${!NODES[@]}"; do
        NODE_ADDRESS="${NODES_ADDRESS[$i]}"
        echo "Attempting to download logs from $USER@$NODE_ADDRESS..."
        scp -i /work/benyamin/smollm/ssh/id_ed25519 -r $USER@$NODE_ADDRESS:$LOG_BASE_PATH/* $LOG_BASE_PATH/ 2>/dev/null
        if [ $? -ne 0 ]; then
            echo "Warning: No logs found or failed to download logs from $USER@$NODE_ADDRESS. Skipping this node."
            continue
        fi
        echo "Logs successfully downloaded from $USER@$NODE_ADDRESS."
    done
    echo "============================ Log download complete ============================"
}

# Execute the log download function
download_logs
