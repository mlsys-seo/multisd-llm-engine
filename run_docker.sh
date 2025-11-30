sudo docker build -t sdprofiler:0.1 .


sudo docker run -d -it \
    -v ${PWD}:/workspace/sdprofiler \
    -v ${HOME}/cache:/workspace/cache \
    --name sk-sdprofiler \
    --ipc=host \
    --gpus all \
    --cap-add=SYS_ADMIN \
    --ulimit memlock=-1 \
    --restart=unless-stopped \
    --privileged \
    sdprofiler:0.2 bash

sudo docker run -d -it \
    -v ${PWD}:/workspace/sdprofiler \
    -v ${HOME}/cache:/workspace/cache \
    --name sk-sdprofiler \
    --ipc=host \
    --gpus all \
    --cap-add=SYS_ADMIN \
    --ulimit memlock=-1 \
    --restart=unless-stopped \
    --privileged \
    nvcr.io/nvidia/pytorch:23.10-py3 bash
    

''' ETRI (applying cache dir)
sudo docker run -d -it \
    -v ${PWD}:/workspace/sdprofiler \
    -v ${HOME}/cache:/workspace/cache \
    --name sk-flashinfer-fix \
    --ipc=host \
    --gpus all \
    --cap-add=SYS_ADMIN \
    --ulimit memlock=-1 \
    --restart=unless-stopped \
    --privileged \
    sdprofiler:0.1 bash
'''

'''
sudo docker run -d -it \
    -v ${PWD}:/workspace/mlsys \
    -v ${HOME}/cache:/workspace/cache \
    --name sk-latency \
    --ipc=host \
    --gpus all \
    --cap-add=SYS_ADMIN \
    --ulimit memlock=-1 \
    --restart=unless-stopped \
    --privileged \
    nvcr.io/nvidia/pytorch:23.10-py3 bash
'''

# export TORCH_CUDA_ARCH_LIST="8.6"
# pip install flashinfer -i https://flashinfer.ai/whl/cu121/torch2.4/