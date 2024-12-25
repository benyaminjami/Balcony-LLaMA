#!/bin/bash
#SBATCH --job-name=smollm1-135M
#SBATCH --nodes=4
#SBATCH --gres=gpu:8
#SBATCH --qos=high
#SBATCH --output=./logs/train-%j.out
#SBATCH --error=./logs/train-%j.err

set -e

TRAINER_PYTHON_FILE="run_train.py"
CONFIG_PATH_YAML="smollm1/config_smollm1_135M.yaml"
nvidia-smi

# Show some environment variables
echo python3 version = `python3 --version`
echo "Python path: $(which python3)"
echo "NCCL version: $(python -c "import torch;print(torch.cuda.nccl.version())")"
echo "CUDA version: $(python -c "import torch;print(torch.version.cuda)")"

echo "START TIME: $(date)"
secs_to_human() {
    echo "$(( ${1} / 3600 )):$(( (${1} / 60) % 60 )):$(( ${1} % 60 ))"
}
start=$(date +%s)
echo "$(date -d @${start} "+%Y-%m-%d %H:%M:%S"): ${SLURM_JOB_NAME} start id=${SLURM_JOB_ID}\n"

# SLURM stuff
export HOSTNAMES=`scontrol show hostnames "$SLURM_JOB_NODELIST"`
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=6000
export COUNT_NODE=`scontrol show hostnames "$SLURM_JOB_NODELIST" | wc -l`

export CUDA_DEVICE_MAX_CONNECTIONS="1"

echo "Number of nodes: $COUNT_NODE"
echo "Hostnames: $HOSTNAMES"

CMD=" $TRAINER_PYTHON_FILE \
    --config-file $CONFIG_PATH_YAML \
    "
export LAUNCHER="torchrun \
    --nproc_per_node 8 \
    --nnodes $COUNT_NODE \
    --node_rank $SLURM_PROCID \
    --role $SLURMD_NODENAME: \
    --max_restarts 0 \
    --tee 3 \
    "

# Wait a random number between 0 and 1000 (milliseconds) to avoid too many concurrent requests to the hub
random_milliseconds=$(( RANDOM % 1001 ))
sleep_time=$(bc <<< "scale=3; $random_milliseconds / 1000")
echo "Sleeping for $sleep_time seconds..."
sleep $sleep_time

srun $SRUN_ARGS -u bash -c "$LAUNCHER --node_rank $SLURM_PROCID --role $SLURMD_NODENAME: $CMD"

echo "END TIME: $(date)"