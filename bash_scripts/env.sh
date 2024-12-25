source ~/miniconda3/etc/profile.d/conda.sh
pip install conda-pack
mkdir ~/miniconda3/envs/smol
tar -xzf /work/benyamin/smollm/conda_env.tar.gz -C ~/miniconda3/envs/smol/
source ~/miniconda3/envs/smol/bin/activate
conda-unpack
pip install /work/benyamin/smollm/nanotron/
