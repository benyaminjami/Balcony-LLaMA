#!/bin/bash

DEST_NODE=$1
CHECKPOINT_BASE_PATH=$2

# Ensure correct zero-padding for destination node numbers less than 10
if [ "$DEST_NODE" -lt 10 ]; then
  DEST_NODE="lux-3-node-0${DEST_NODE}"
else
  DEST_NODE="lux-3-node-${DEST_NODE}"
fi

# Determine the latest checkpoint
echo "Fetching the latest checkpoint from source node..."
LATEST_CHECKPOINT=$(cat ${CHECKPOINT_BASE_PATH}/latest.txt)

if [ -z "$LATEST_CHECKPOINT" ]; then
  echo "Error: Failed to retrieve the latest checkpoint."
  exit 1
fi

echo "Latest checkpoint is: $LATEST_CHECKPOINT"
LATEST_CHECKPOINT_PATH="${CHECKPOINT_BASE_PATH}/${LATEST_CHECKPOINT}"

# Perform the rsync operation and exclude the random folder
rsync -avz --info=name --partial --append-verify \
    -e "ssh -i /work/benyamin/smollm/ssh/id_ed25519" \
  ${LATEST_CHECKPOINT_PATH}/ ${DEST_NODE}:${LATEST_CHECKPOINT_PATH}/

if [ $? -eq 0 ]; then
  echo "Checkpoint transfer completed successfully."
else
  echo "Error: Checkpoint transfer failed."
  exit 1
fi
