export TORCH_CUDA_ARCH_LIST="9.0" # "8.0" for A30

# TORCH_CUDA_ARCH_LIST="8.0" for A30, A100
# TORCH_CUDA_ARCH_LIST="8.6" for A6000
# TORCH_CUDA_ARCH_LIST="9.0" for H100

# Delete unused files
mkdir -p /workspace
cd /workspace

python -m pip install --upgrade pip

# Delete PyTorch and other packages for PyTorch
#pip uninstall -y torch torchdata torchtext torchvision

# Install the flashinfer package
git clone --branch v0.2.4 https://github.com/flashinfer-ai/flashinfer.git --recursive
cd flashinfer

pip install --no-build-isolation --verbose --editable .

# Install the vLLM package
# git clone https://github.com/vllm-project/vllm.git
# cd /workspace/vllm
# git checkout tags/v0.6.0
# pip install -r requirements-build.txt && pip install -e . --no-build-isolation

# Set the default directory to /workspace
cd /workspace


pip install --upgrade filelock msgpack datasets ray soxr transformers==4.53.3 
