#!/bin/bash
CODE=${2:-`pwd`}

docker run -itd --rm \
           --ipc host \
           --network host \
           --gpus all \
           --env "NVIDIA_DRIVER_CAPABILITIES=all" \
           -v $CODE:/home/docker_user/remembr:rw \
           --name remembr_container_vila remembr_image:vila