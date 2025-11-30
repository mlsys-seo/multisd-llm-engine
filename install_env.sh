export TORCH_CUDA_ARCH_LIST="8.6"

# TORCH_CUDA_ARCH_LIST="8.0" for A30, A100
# TORCH_CUDA_ARCH_LIST="8.6" for A6000, A40
# TORCH_CUDA_ARCH_LIST="9.0" for H100


python -m pip install --upgrade pip

# Delete PyTorch and other packages for PyTorch
pip uninstall -y torch torchdata torchtext torchvision

wget https://github.com/flashinfer-ai/flashinfer/releases/download/v0.2.4/flashinfer_python-0.2.4+cu124torch2.6-cp38-abi3-linux_x86_64.whl
pip install flashinfer_python-0.2.4+cu124torch2.6-cp38-abi3-linux_x86_64.whl


# # A6000
# wget https://github.com/flashinfer-ai/flashinfer/releases/download/v0.2.4/flashinfer_python-0.2.4+cu121torch2.5-cp38-abi3-linux_x86_64.whl
# TORCH_CUDA_ARCH_LIST="8.6" pip install flashinfer_python-0.2.4+cu121torch2.5-cp38-abi3-linux_x86_64.whl


wget https://github.com/flashinfer-ai/flashinfer/releases/download/v0.2.4/flashinfer_python-0.2.4+cu124torch2.6-cp38-abi3-linux_x86_64.whl
TORCH_CUDA_ARCH_LIST="8.6" pip install flashinfer_python-0.2.4+cu124torch2.6-cp38-abi3-linux_x86_64.whl


wget https://github.com/flashinfer-ai/flashinfer/releases/download/v0.2.4/flashinfer_python-0.2.4+cu124torch2.6-cp38-abi3-linux_x86_64.whl
TORCH_CUDA_ARCH_LIST="8.6" pip install flashinfer_python-0.2.4+cu124torch2.6-cp38-abi3-linux_x86_64.whl



# Set the default directory to /workspace
cd /workspace


apt install apt-transport-https curl gnupg -y
curl -fsSL https://bazel.build/bazel-release.pub.gpg | gpg --dearmor >bazel-archive-keyring.gpg
mv bazel-archive-keyring.gpg /usr/share/keyrings
echo "deb [arch=amd64 signed-by=/usr/share/keyrings/bazel-archive-keyring.gpg] https://storage.googleapis.com/bazel-apt stable jdk1.8" | tee /etc/apt/sources.list.d/bazel.list


apt update && apt install bazel

pip install --upgrade datasets ray soxr transformers==4.53.3 