# Stage 1: Base
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04 as base

SHELL ["/bin/bash", "-o", "pipefail", "-c"]
ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Europe/London \
    PYTHONUNBUFFERED=1 \
    SHELL=/bin/bash

# Install Ubuntu packages
RUN apt update && \
    apt -y upgrade && \
    apt install -y --no-install-recommends \
        build-essential \
        software-properties-common \
        python3.10-venv \
        python3-pip \
        python3-tk \
        python3-dev \
        nodejs \
        npm \
        bash \
        dos2unix \
        git \
        git-lfs \
        ncdu \
        nginx \
        net-tools \
        inetutils-ping \
        openssh-server \
        libglib2.0-0 \
        libsm6 \
        libgl1 \
        libxrender1 \
        libxext6 \
        ffmpeg \
        wget \
        curl \
        psmisc \
        rsync \
        vim \
        zip \
        unzip \
        p7zip-full \
        htop \
        screen \
        tmux \
        bc \
        pkg-config \
        plocate \
        libcairo2-dev \
        libgoogle-perftools4 \
        libtcmalloc-minimal4 \
        apt-transport-https \
        ca-certificates && \
    update-ca-certificates && \
    apt clean && \
    rm -rf /var/lib/apt/lists/* && \
    echo "en_US.UTF-8 UTF-8" > /etc/locale.gen

# Set Python
RUN ln -s /usr/bin/python3.10 /usr/bin/python

# Install Torch, xformers and tensorrt
ARG INDEX_URL
ARG TORCH_VERSION
ARG XFORMERS_VERSION
RUN pip3 install --no-cache-dir torch==${TORCH_VERSION} torchvision torchaudio --index-url ${INDEX_URL} && \
    pip3 install --no-cache-dir xformers==0.0.22  
    #pip3 install --no-cache-dir tensorrt

# Stage 2: Install applications
FROM base as setup

RUN mkdir -p /sd-models

# Add SDXL models and VAE
# These need to already have been downloaded:
#   wget https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0.safetensors
#   wget https://huggingface.co/stabilityai/stable-diffusion-xl-refiner-1.0/resolve/main/sd_xl_refiner_1.0.safetensors
#   wget https://huggingface.co/madebyollin/sdxl-vae-fp16-fix/resolve/main/sdxl_vae.safetensors
#COPY sd_xl_base_1.0.safetensors /sd-models/sd_xl_base_1.0.safetensors
#COPY sd_xl_refiner_1.0.safetensors /sd-models/sd_xl_refiner_1.0.safetensors
#COPY sdxl_vae.safetensors /sd-models/sdxl_vae.safetensors

# Clone the git repo of the Stable Diffusion Web UI by Automatic1111
# and set version
ARG WEBUI_VERSION
WORKDIR /
RUN git clone https://github.com/AUTOMATIC1111/stable-diffusion-webui.git && \
    cd /stable-diffusion-webui && \
    git checkout tags/${WEBUI_VERSION}

WORKDIR /stable-diffusion-webui
RUN python3 -m venv --system-site-packages /venv && \
    source /venv/bin/activate && \
    pip3 install --no-cache-dir torch==${TORCH_VERSION} torchvision torchaudio --index-url ${INDEX_URL} && \
    pip3 install --no-cache-dir xformers && \
    pip3 install tensorflow[and-cuda] && \
    deactivate

# Install the dependencies for the Automatic1111 Stable Diffusion Web UI
COPY a1111/cache-sd-model.py a1111/install-automatic.py ./
RUN source /venv/bin/activate && \
    pip3 install -r requirements_versions.txt && \
    python3 -m install-automatic --skip-torch-cuda-test && \
    deactivate

# Cache the Stable Diffusion Models
# SDXL models result in OOM kills with 8GB system memory, need 30GB+ to cache these
#RUN source /venv/bin/activate && \
    #python3 cache-sd-model.py --no-half-vae --no-half --xformers --use-cpu=all --ckpt /sd-models/sd_xl_base_1.0.safetensors && \
    #python3 cache-sd-model.py --no-half-vae --no-half --xformers --use-cpu=all --ckpt /sd-models/sd_xl_refiner_1.0.safetensors && \
    #deactivate
    
Run git clone https://huggingface.co/embed/negative embeddings/negative && \
    git clone https://huggingface.co/embed/lora models/Lora/positive
    
# Clone the Automatic1111 Extensions
RUN git clone https://github.com/Mikubill/sd-webui-controlnet.git extensions/sd-webui-controlnet && \
    git clone --depth=1 https://github.com/deforum-art/sd-webui-deforum.git extensions/deforum && \
    git clone --depth=1 https://github.com/ashleykleynhans/a1111-sd-webui-locon.git extensions/a1111-sd-webui-locon && \
    git clone --depth=1 https://github.com/Gourieff/sd-webui-reactor.git extensions/sd-webui-reactor && \
    git clone --depth=1 https://github.com/zanllp/sd-webui-infinite-image-browsing.git extensions/infinite-image-browsing && \
    git clone --depth=1 https://github.com/Uminosachi/sd-webui-inpaint-anything.git extensions/inpaint-anything && \
    git clone --depth=1 https://github.com/Bing-su/adetailer.git extensions/adetailer && \
    git clone --depth=1 https://github.com/civitai/sd_civitai_extension.git extensions/sd_civitai_extension && \
    git clone https://github.com/BlafKing/sd-civitai-browser-plus.git extensions/sd-civitai-browser-plus && \
    git clone --depth=1 https://github.com/Coyote-A/ultimate-upscale-for-automatic1111 extensions/ultimate-upscale-for-automatic1111 && \
    git clone --depth=1 https://github.com/etherealxx/batchlinks-webui extensions/batchlinks-webui && \
    git clone --depth=1 https://github.com/continue-revolution/sd-webui-animatediff extensions/sd-webui-animatediff

RUN cd /stable-diffusion-webui/extensions/sd-webui-animatediff/model && \
    wget https://huggingface.co/conrevo/AnimateDiff-A1111/resolve/main/motion_module/mm_sd15_v3.safetensors && \
    cd /stable-diffusion-webui/models/Stable-diffusion && \
    wget https://civitai.com/api/download/models/148087 --content-disposition && \
    wget https://civitai.com/api/download/models/179525 --content-disposition && \
    cd /stable-diffusion-webui/models/Lora && \
    wget https://civitai.com/api/download/models/132876 --content-disposition && \
    mkdir -p /stable-diffusion-webui/models/ESRGAN && \
    cd /stable-diffusion-webui/models/ESRGAN && \
    wget https://huggingface.co/embed/upscale/resolve/main/4x-UltraSharp.pth && \
    wget https://huggingface.co/uwg/upscaler/resolve/main/ESRGAN/8x_NMKD-Superscale_150000_G.pth
    
# Install dependencies for Deforum, ControlNet, ReActor, Infinite Image Browsing,
# After Detailer, and CivitAI Browser+ extensions
ARG CONTROLNET_COMMIT
ARG CIVITAI_BROWSER_PLUS_VERSION
RUN source /venv/bin/activate && \
    pip3 install basicsr && \
    cd /stable-diffusion-webui/extensions/sd-webui-controlnet && \
    pip3 install -r requirements.txt && \
    cd /stable-diffusion-webui/extensions/deforum && \
    pip3 install -r requirements.txt && \
    cd /stable-diffusion-webui/extensions/sd-webui-reactor && \
    pip3 install -r requirements.txt && \
    #pip3 install onnxruntime-gpu=1.17.1 && \
    cd /stable-diffusion-webui/extensions/infinite-image-browsing && \
    pip3 install -r requirements.txt && \
    cd /stable-diffusion-webui/extensions/adetailer && \
    python3 -m install && \
    cd /stable-diffusion-webui/extensions/sd_civitai_extension && \
    pip3 install -r requirements.txt && \
    deactivate

# Install dependencies for inpaint anything extension
RUN source /venv/bin/activate && \
    pip3 install segment_anything lama_cleaner && \
    deactivate

# Install dependencies for Civitai Browser+ extension
RUN source /venv/bin/activate && \
    cd /stable-diffusion-webui/extensions/sd-civitai-browser-plus && \
    pip3 install send2trash beautifulsoup4 ZipUnicode fake-useragent packaging pysocks && \
    deactivate

# Set Dreambooth extension version
#ARG DREAMBOOTH_COMMIT
#WORKDIR /stable-diffusion-webui/extensions/sd_dreambooth_extension
#RUN git checkout main && \
    #git reset ${DREAMBOOTH_COMMIT} --hard

# Install the dependencies for the Dreambooth extension
#WORKDIR /stable-diffusion-webui
#RUN source /venv/bin/activate && \
    #cd /stable-diffusion-webui/extensions/sd_dreambooth_extension && \
    #pip3 install -r requirements.txt && \
    #pip3 cache purge && \
    #deactivate

# Add inswapper model for the ReActor extension
RUN mkdir -p /stable-diffusion-webui/models/insightface && \
    cd /stable-diffusion-webui/models/insightface && \
    wget https://github.com/facefusion/facefusion-assets/releases/download/models/inswapper_128.onnx

#Controlnet models
WORKDIR /stable-diffusion-webui/extensions/sd-webui-controlnet/models
RUN wget https://huggingface.co/ckpt/ControlNet-v1-1/resolve/main/control_v11p_sd15_canny_fp16.safetensors && \
    wget https://huggingface.co/ckpt/ControlNet-v1-1/raw/main/control_v11p_sd15_canny_fp16.yaml && \
    wget https://huggingface.co/ckpt/ControlNet-v1-1/resolve/main/control_v11f1p_sd15_depth_fp16.safetensors && \
    wget https://huggingface.co/ckpt/ControlNet-v1-1/raw/main/control_v11f1p_sd15_depth_fp16.yaml && \
    wget https://huggingface.co/ckpt/ControlNet-v1-1/resolve/main/control_v11p_sd15_normalbae_fp16.safetensors && \
    wget https://huggingface.co/ckpt/ControlNet-v1-1/raw/main/control_v11p_sd15_normalbae_fp16.yaml && \
    wget https://huggingface.co/ckpt/ControlNet-v1-1/resolve/main/control_v11p_sd15_mlsd_fp16.safetensors && \
    wget https://huggingface.co/ckpt/ControlNet-v1-1/raw/main/control_v11p_sd15_mlsd_fp16.yaml && \
    wget https://huggingface.co/ckpt/ControlNet-v1-1/resolve/main/control_v11p_sd15_openpose_fp16.safetensors && \
    wget https://huggingface.co/ckpt/ControlNet-v1-1/raw/main/control_v11p_sd15_openpose_fp16.yaml && \
    wget https://huggingface.co/ckpt/ControlNet-v1-1/resolve/main/control_v11p_sd15_lineart_fp16.safetensors && \
    wget https://huggingface.co/ckpt/ControlNet-v1-1/resolve/main/control_v11p_sd15s2_lineart_anime_fp16.safetensors && \
    wget https://huggingface.co/ckpt/ControlNet-v1-1/raw/main/control_v11p_sd15_lineart_fp16.yaml && \
    wget https://huggingface.co/ckpt/ControlNet-v1-1/raw/main/control_v11p_sd15s2_lineart_anime_fp16.yaml && \
    wget https://huggingface.co/ckpt/ControlNet-v1-1/resolve/main/control_v11p_sd15_inpaint_fp16.safetensors && \
    wget https://huggingface.co/ckpt/ControlNet-v1-1/raw/main/control_v11p_sd15_inpaint_fp16.yaml && \
    wget https://huggingface.co/ckpt/ControlNet-v1-1/resolve/main/control_v11p_sd15_scribble_fp16.safetensors && \
    wget https://huggingface.co/ckpt/ControlNet-v1-1/raw/main/control_v11p_sd15_scribble_fp16.yaml && \
    wget https://huggingface.co/ckpt/ControlNet-v1-1/resolve/main/control_v11p_sd15_softedge_fp16.safetensors && \
    wget https://huggingface.co/ckpt/ControlNet-v1-1/raw/main/control_v11p_sd15_softedge_fp16.yaml && \
    wget https://huggingface.co/ckpt/ControlNet-v1-1/resolve/main/control_v11f1e_sd15_tile_fp16.safetensors && \
    wget https://huggingface.co/ckpt/ControlNet-v1-1/raw/main/control_v11f1e_sd15_tile_fp16.yaml && \
    wget https://huggingface.co/ckpt/ControlNet-v1-1/resolve/main/control_v11e_sd15_shuffle_fp16.safetensors && \
    wget https://huggingface.co/ckpt/ControlNet-v1-1/raw/main/control_v11e_sd15_shuffle_fp16.yaml && \
    wget https://huggingface.co/ckpt/ControlNet-v1-1/resolve/main/control_v11e_sd15_ip2p_fp16.safetensors && \
    wget https://huggingface.co/ckpt/ControlNet-v1-1/raw/main/control_v11e_sd15_ip2p_fp16.yaml && \
    wget https://huggingface.co/monster-labs/control_v1p_sd15_qrcode_monster/resolve/main/control_v1p_sd15_qrcode_monster.safetensors && \
    wget https://huggingface.co/monster-labs/control_v1p_sd15_qrcode_monster/resolve/main/control_v1p_sd15_qrcode_monster.yaml && \
    wget https://huggingface.co/lllyasviel/sd_control_collection/resolve/main/ip-adapter_sd15.pth && \
    wget https://huggingface.co/h94/IP-Adapter/resolve/main/models/ip-adapter-plus-face_sd15.bin && \
    wget https://huggingface.co/h94/IP-Adapter/resolve/main/models/ip-adapter-plus_sd15.bin 
    
# Configure ReActor to use the GPU instead of the CPU
RUN echo "CUDA" > /stable-diffusion-webui/extensions/sd-webui-reactor/last_device.txt

# Install Kohya_ss
ARG KOHYA_VERSION
RUN git clone https://github.com/bmaltais/kohya_ss.git /kohya_ss && \
    cd /kohya_ss && \
    git checkout ${KOHYA_VERSION} && \
    git submodule update --init --recursive

WORKDIR /kohya_ss
COPY kohya_ss/requirements* ./
RUN python3 -m venv --system-site-packages venv && \
    source venv/bin/activate && \
    pip3 install --no-cache-dir torch==2.0.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 && \
    pip3 install --no-cache-dir xformers==0.0.22 \
        bitsandbytes==0.41.1 \
        tensorboard==2.14.1 \
        tensorflow==2.14.0 \
        wheel \
        scipy \
    pip3 install -r requirements.txt && \
    pip3 install . && \
    pip3 cache purge && \
    deactivate


# Install ComfyUI
RUN git clone https://github.com/comfyanonymous/ComfyUI.git /ComfyUI
WORKDIR /ComfyUI
RUN python3 -m venv --system-site-packages venv && \
    source venv/bin/activate && \
    pip3 install --no-cache-dir torch==${TORCH_VERSION} torchvision torchaudio --index-url ${INDEX_URL} && \
    pip3 install --no-cache-dir xformers==0.0.22 && \
    pip3 install -r requirements.txt && \
    deactivate

# Install ComfyUI Custom Nodes
RUN git clone https://github.com/ltdrdata/ComfyUI-Manager.git custom_nodes/ComfyUI-Manager && \
    cd custom_nodes/ComfyUI-Manager && \
    source /ComfyUI/venv/bin/activate && \
    pip3 install -r requirements.txt && \
    pip3 cache purge && \
    deactivate

# Install Application Manager
WORKDIR /
RUN git clone https://github.com/ashleykleynhans/app-manager.git /app-manager && \
    cd /app-manager && \
    npm install

# Install Jupyter, Tensorboad, gdown and OhMyRunPod
RUN pip3 uninstall -y tensorboard tb-nightly && \
    pip3 install -U --no-cache-dir jupyterlab \
        jupyterlab_widgets \
        ipykernel \
        ipywidgets \
        tensorboard==2.14.1 tensorflow==2.14.0 \
        gdown \
        OhMyRunPod && \
    pip3 cache purge


# Install RunPod File Uploader
RUN curl -sSL https://github.com/kodxana/RunPod-FilleUploader/raw/main/scripts/installer.sh -o installer.sh && \
    chmod +x installer.sh && \
    ./installer.sh

# Install rclone
RUN curl https://rclone.org/install.sh | bash

# Install runpodctl
ARG RUNPODCTL_VERSION
RUN wget "https://github.com/runpod/runpodctl/releases/download/${RUNPODCTL_VERSION}/runpodctl-linux-amd64" -O runpodctl && \
    chmod a+x runpodctl && \
    mv runpodctl /usr/local/bin

# Install croc
RUN curl https://getcroc.schollz.com | bash

# Install speedtest CLI
RUN curl -s https://packagecloud.io/install/repositories/ookla/speedtest-cli/script.deb.sh | bash && \
    apt install speedtest

# Install CivitAI Model Downloader
ARG CIVITAI_DOWNLOADER_VERSION
RUN git clone https://github.com/ashleykleynhans/civitai-downloader.git && \
    cd civitai-downloader && \
    git checkout tags/${CIVITAI_DOWNLOADER_VERSION} && \
    cp download.py /usr/local/bin/download-model && \
    chmod +x /usr/local/bin/download-model && \
    cd .. && \
    rm -rf civitai-downloader

# Copy Stable Diffusion Web UI config files
COPY a1111/relauncher.py a1111/webui-user.sh a1111/config.json a1111/ui-config.json /stable-diffusion-webui/

# ADD SDXL styles.csv
ADD https://raw.githubusercontent.com/Douleb/SDXL-750-Styles-GPT4-/main/styles.csv /stable-diffusion-webui/styles.csv

# Copy ComfyUI Extra Model Paths (to share models with A1111)
COPY comfyui/extra_model_paths.yaml /ComfyUI/

# Remove existing SSH host keys
RUN rm -f /etc/ssh/ssh_host_*

# NGINX Proxy
COPY nginx/nginx.conf /etc/nginx/nginx.conf
COPY nginx/502.html /usr/share/nginx/html/502.html

# Set template version
ARG RELEASE
ENV TEMPLATE_VERSION=${RELEASE}

# Set the main venv path
ARG VENV_PATH
ENV VENV_PATH=${VENV_PATH}

# Copy the scripts
WORKDIR /
COPY --chmod=755 scripts/* ./

# Copy the accelerate configuration
COPY kohya_ss/accelerate.yaml ./

# Start the container
SHELL ["/bin/bash", "--login", "-c"]
CMD [ "/start.sh" ]
