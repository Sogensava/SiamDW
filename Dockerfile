FROM mcr.microsoft.com/azureml/o16n-base/python-assets@sha256:20a8b655a3e5b9b0db8ddf70d03d048a7cf49e569c4f0382198b1ee77631a6ad AS inferencing-assets

FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04
USER root:root

ENV com.nvidia.cuda.version $CUDA_VERSION
ENV com.nvidia.volumes.needed nvidia_driver
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV DEBIAN_FRONTEND noninteractive
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64
ENV NCCL_DEBUG=INFO
ENV HOROVOD_GPU_ALLREDUCE=NCCL

# Inference
COPY --from=inferencing-assets /artifacts /var/

# Install Common Dependencies
RUN apt-get update && \
    apt-get install -y apt-transport-https && \
    apt-get install -y --no-install-recommends \
    # SSH and RDMA
    libmlx4-1 \
    libmlx5-1 \
    librdmacm1 \
    libibverbs1 \
    libmthca1 \
    libdapl2 \
    dapl2-utils \
    openssh-client \
    openssh-server \
    sudo \
    iproute2 && \
    # Others
    apt-get install -y --no-install-recommends \
    --allow-change-held-packages \
    build-essential \
    bzip2 \
    libbz2-1.0\
    systemd \
    git \
    wget \
    vim \
    tmux \
    unzip \
    ca-certificates \
    libjpeg-dev \
    cpio \
    jq \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1-mesa-glx \
    libturbojpeg \
    fuse && \
    apt-get clean -y && \
    rm -rf /var/lib/apt/lists/*


# Set timezone
RUN ln -sf /usr/share/zoneinfo/Asia/Shanghai /etc/localtime

# create dir for sshd
RUN mkdir -p /var/run/sshd /root/.ssh

# Inference
# Copy logging utilities, nginx and rsyslog configuration files, IOT server binary, etc.
# COPY --from=inferencing-assets /artifacts /var/
# RUN /var/requirements/install_system_requirements.sh && \
#     cp /var/configuration/rsyslog.conf /etc/rsyslog.conf && \
#     cp /var/configuration/nginx.conf /etc/nginx/sites-available/app && \
#     ln -s /etc/nginx/sites-available/app /etc/nginx/sites-enabled/app && \
#     rm -f /etc/nginx/sites-enabled/default
# ENV SVDIR=/var/runit
# ENV WORKER_TIMEOUT=300
# EXPOSE 5001 8883 8888

# Conda Environment
ENV MINICONDA_VERSION latest
ENV PATH /opt/miniconda/bin:$PATH
RUN wget -qO /tmp/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    bash /tmp/miniconda.sh -bf -p /opt/miniconda && \
    conda clean -ay && \
    rm -rf /opt/miniconda/pkgs && \
    rm /tmp/miniconda.sh && \
    find / -type d -name __pycache__ | xargs rm -rf
# Open-MPI installation
ENV OPENMPI_VERSION 3.1.2
RUN mkdir /tmp/openmpi && \
    cd /tmp/openmpi && \
    wget https://download.open-mpi.org/release/open-mpi/v3.1/openmpi-${OPENMPI_VERSION}.tar.gz && \
    tar zxf openmpi-${OPENMPI_VERSION}.tar.gz && \
    cd openmpi-${OPENMPI_VERSION} && \
    ./configure --enable-orterun-prefix-by-default && \
    make -j $(nproc) all && \
    make install && \
    ldconfig && \
    rm -rf /tmp/openmpi

RUN conda install -c r -y conda python=3.9 pip
RUN conda install -y numpy pyyaml ipython mkl scikit-learn matplotlib pandas setuptools Cython h5py graphviz libgcc mkl-include cmake cffi typing cython && \
    conda install -y -c mingfeima mkldnn && \
    conda install -c anaconda gxx_linux-64

RUN conda install -c anaconda gxx_linux-64
RUN conda clean -ya
#RUN pip install boto3 addict tqdm regex pyyaml opencv-python opencv-contrib-python azureml-defaults nltk spacy future tensorboard wandb filelock tokenizers sentencepiece
RUN pip install regex pyyaml opencv-contrib-python future

# Set CUDA_ROOT
RUN export CUDA_HOME="/usr/local/cuda"

# Install littleboy related packages

# Install pytorch
#RUN conda install pytorch torchvision torchaudio cudatoolkit=10.0 -c pytorch 
#RUN conda install pytorch==1.10.1 torchvision==0.11.2 cudatoolkit=10.0 torchaudio==0.10.1 -c pytorch
#RUN conda install torchvision==0.19.1

# -c conda-forge

# Install other packages
RUN pip install PyYAML && \
    pip install easydict && \
    pip install cython && \
    pip install opencv-python && \
    pip install pandas && \
    pip install tqdm && \
    pip install pycocotools && \
    # apt-get install libturbojpeg && \
    pip install jpeg4py && \
    pip install tb-nightly && \
    pip install tikzplotlib && \
    pip install thop && \
    pip install lmdb && \
    pip install scipy && \
    pip install visdom && \
    pip install wandb && \
    pip install tensorboardX && \
    pip install yacs && \
    pip install timm && \
    pip install class_registry && \
    pip install shapely && \
    pip install gdown
# conda env installation
# RUN conda install --file ./docker_requirements.txt

# create workspace 
RUN mkdir /home/vot_ws

RUN git clone https://github.com/votchallenge/toolkit /home/toolkit
WORKDIR /home/toolkit
RUN cd /home/toolkit && python /home/toolkit/setup.py install
WORKDIR /home/vot_ws
RUN vot initialize vot2022/shorttermbox --workspace .
RUN git clone https://github.com/Sogensava/SiamDW.git
RUN cd ./SiamDW
RUN mkdir ./snapshot
RUN cd ./snapshot 
WORKDIR /home/vot_ws/SiamDW/snapshot
RUN gdown https://drive.google.com/file/d/1tBllNtv-90Ih2EP_lnRCBzxnZeFPRnPx
WORKDIR /home/vot_ws/SiamDW
RUN cd ..

COPY ./trackers.ini /home/vot_ws/trackers.ini