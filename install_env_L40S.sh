export TORCH_CUDA_ARCH_LIST="8.9"

# TORCH_CUDA_ARCH_LIST="8.0" for A30, A100
# TORCH_CUDA_ARCH_LIST="8.6" for A6000, A40
# TORCH_CUDA_ARCH_LIST="8.9" for L40S
# TORCH_CUDA_ARCH_LIST="9.0" for H100


python -m pip install --upgrade pip

# Delete PyTorch and other packages for PyTorch
pip uninstall -y torch torchdata torchtext torchvision

pip install torch==2.6.0

pip install -e .

wget https://github.com/flashinfer-ai/flashinfer/releases/download/v0.2.4/flashinfer_python-0.2.4+cu124torch2.6-cp38-abi3-linux_x86_64.whl
TORCH_CUDA_ARCH_LIST="8.9" pip install flashinfer_python-0.2.4+cu124torch2.6-cp38-abi3-linux_x86_64.whl

apt update && apt install nvtop
apt install tmux

pip install --upgrade datasets ray soxr transformers==4.53.3 
#pip install flash_attn==2.6.3