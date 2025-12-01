#!/bin/bash
CODE=${2:-`pwd`}
PATH_TO_CODA=""

docker run -itd --rm \
           --ipc host \
           --network host \
           --gpus all \
           --env "NVIDIA_DRIVER_CAPABILITIES=all" \
           -v $CODE:/home/docker_user/remembr:rw \
           -v $PATH_TO_CODA:/home/docker_user/CODA_data:rw \
           --name remembr_container_vila remembr_image:vila