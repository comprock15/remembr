FROM nvcr.io/nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

SHELL [ "/bin/bash", "-c" ]
ENV DEBIAN_FRONTEND noninteractive
ENV USER=docker_user

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        sudo git wget curl nano unzip ffmpeg libsm6 libxext6 && \
    rm -rf /var/lib/apt/lists/*

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh
ENV PATH="/opt/conda/bin:$PATH"

# RUN /opt/conda/bin/conda update -n base -y conda
# RUN /opt/conda/bin/conda config --set always_yes yes && \
#     echo "channels:" > ~/.condarc && \
#     echo "  - defaults" >> ~/.condarc

ARG UID=1000
ARG GID=1000
RUN useradd -m ${USER} --uid=${UID} \
    && usermod -s /bin/bash ${USER} \
    && usermod -a -G sudo ${USER}

WORKDIR /home/${USER}/remembr
COPY vila_setup.sh requirements.txt ../deps/VILA/ /home/${USER}/remembr/

RUN curl -fsSL https://ollama.com/install.sh | sh

RUN chmod +x vila_setup.sh

RUN /bin/bash -c "\
    . /opt/conda/etc/profile.d/conda.sh && \
    conda init bash && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r && \
    conda create -y --name remembr python=3.10 && \
    conda run -n remembr ./vila_setup.sh remembr && \
    conda run -n remembr python -m pip install -r requirements.txt" 
    
RUN chown -R ${USER}:${USER} /opt/conda && \
    chown -R ${USER}:${USER} /home/${USER}

USER ${USER}