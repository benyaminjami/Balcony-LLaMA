#!/bin/bash

SERVER_NUMBER=$1

# Ensure correct zero-padding for node numbers less than 10
if [ "$SERVER_NUMBER" -lt 10 ]; then
  NODE="lux-3-node-0${SERVER_NUMBER}"
else
  NODE="lux-3-node-${SERVER_NUMBER}"
fi

# Perform the rsync operation and exclude the log folder
rsync -avz --info=name --partial --append-verify \
    --exclude="logs/" \
    --exclude="checkpoints/" -e "ssh -i /work/benyamin/smollm/ssh/id_ed25519" \
    /work/benyamin/datasets/ benyamin@$NODE:/work/benyamin/datasets/
# /Smollm_1B/history/" \