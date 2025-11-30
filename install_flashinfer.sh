export TORCH_CUDA_ARCH_LIST="8.6"

TORCH_VERSION=$(python -c "import torch; print(torch.__version__)")
CUDA_VERSION=$(python -c "import torch; print(torch.version.cuda)")
GPU_ARCHES=$(python -c "import torch; arches = [arch for arch in torch.cuda.get_arch_list() if float(arch.replace('sm_','')) >= 86]; print(' '.join(arches))")

echo "Torch Version: $TORCH_VERSION"
echo "CUDA Version: $CUDA_VERSION"
echo "GPU Architectures: $GPU_ARCHES"

cd ..
git clone https://github.com/flashinfer-ai/flashinfer.git --recursive

cd flashinfer
TORCH_CUDA_ARCH_LIST="$GPU_ARCHES" FLASHINFER_ENABLE_AOT=1 pip install --no-build-isolation --verbose --editable .

TORCH_CUDA_ARCH_LIST="8.0" FLASHINFER_ENABLE_AOT=1 pip install --no-build-isolation --verbose --editable .

TORCH_CUDA_ARCH_LIST="8.6" FLASHINFER_ENABLE_AOT=1 pip install --no-build-isolation --verbose --editable .