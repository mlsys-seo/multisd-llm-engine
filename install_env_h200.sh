export TORCH_CUDA_ARCH_LIST="9.0" # "8.0" for A30

# TORCH_CUDA_ARCH_LIST="8.0" for A30, A100
# TORCH_CUDA_ARCH_LIST="8.6" for A6000
# TORCH_CUDA_ARCH_LIST="9.0" for H100

# Delete unused files
pip install flashinfer-python ray

pip install -e .

pip install --upgrade filelock msgpack datasets ray soxr transformers==4.53.3 

mkdir -p /workspace/cache
ln -s ~/.models/cheezestick/ /workspace/cache/

sudo apt-get update
sudo apt-get install -y nvtop tmux
