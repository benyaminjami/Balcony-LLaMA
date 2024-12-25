# Pre-training
We use [nanotron](https://github.com/huggingface/nanotron/) library for training SmolLM and SmolLM2 base models.

The scripts for training SmolLM v1 can be found in the `smollm1` folder, and those for training SmolLM2 can be found in the `smollm2` folder, we will add the details for the data mixture soon. SmolLM2 uses similar architecture as SmolLM but uses an improved data mixture and significantly longer training periods (11 trillion tokens for the 1.7B, 4 trillion for the 360M and 2 trillion for the 135M).

## Setup

Please refer to [nanotron](https://github.com/huggingface/nanotron/) for detailed instructions on setting up your training environment and launching jobs.

After setting up the environment and tokenizing the training datasets with [datatrove](https://github.com/huggingface/datatrove) (instructions available [here](https://github.com/huggingface/nanotron/blob/main/docs/nanoset.md#nanosets)), you can modify the configurations to match your number of nodes and local paths.

Below is an example of launching SmolLM1 135M training on 1 node (you can change the DP value to 8 in the config and adjust the batch size) and run:

```bash
git clone https://github.com/huggingface/nanotron
cd nanotron
# follow installation
CUDA_DEVICE_MAX_CONNECTIONS=1 torchrun --nproc_per_node=8 run_train.py --config-file smollm1/config_smollm1_135M.yaml
```

If you are working on a slurm cluster, you can modify the `launch.slurm` and launch the training with:

```bash
sbatch launch.slurm
```
> [!NOTE]
> Don't forget to create the logs directory before launching the job:

## Continual pre-training

The nanotron checkpoints for SmolLM2 models are available at: https://huggingface.co/HuggingFaceTB/SmolLM2-nanotron-ckpt 

To resume the training without the optimizer states or lr schedule (recommended for data mixture changes), use this branch: https://github.com/huggingface/nanotron/pull/253