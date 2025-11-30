FROM nvcr.io/nvidia/pytorch:25.01-py3
# FROM nvcr.io/nvidia/pytorch:23.10-py3

### Environment Variables
# This is for flashinfer
ENV TORCH_CUDA_ARCH_LIST="9.0"
# TORCH_CUDA_ARCH_LIST="8.0" for A30, A100
# TORCH_CUDA_ARCH_LIST="8.6" for A6000
# TORCH_CUDA_ARCH_LIST="9.0" for H100


# Delete unused files
WORKDIR /workspace
RUN rm -rf *

RUN apt-get update
RUN apt-get install -y tmux
RUN python -m pip install --upgrade pip

# Delete PyTorch and other packages for PyTorch
# RUN pip uninstall -y torch torchdata torchtext torchvision

# Install the flashinfer package
# RUN wget https://github.com/flashinfer-ai/flashinfer/releases/download/v0.2.0/flashinfer-0.2.0+cu121torch2.4-cp310-cp310-linux_x86_64.whl
# RUN pip install flashinfer-0.2.0+cu121torch2.4-cp310-cp310-linux_x86_64.whl

RUN wget https://github.com/flashinfer-ai/flashinfer/releases/download/v0.2.4/flashinfer_python-0.2.4+cu124torch2.6-cp38-abi3-linux_x86_64.whl
RUN pip install flashinfer_python-0.2.4+cu124torch2.6-cp38-abi3-linux_x86_64.whl

# Set the default directory to /workspace
WORKDIR /workspace
