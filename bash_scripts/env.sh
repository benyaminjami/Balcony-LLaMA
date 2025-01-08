set -e
cd /work/benyamin/smollm/nanotron
pip install --upgrade pip
pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu121
pip install -e .

# Install dependencies if you want to use the example scripts
pip install datasets transformers
pip install triton "flash-attn>=2.5.0" --no-build-isolation
pip install "datatrove[io,processing]@git+https://github.com/huggingface/datatrove"
pip install numba
pip install "numpy<=2"