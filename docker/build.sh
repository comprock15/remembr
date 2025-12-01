#!/bin/bash

docker build -f ./docker/Dockerfile \
             -t remembr_image:vila \
             --build-arg UID=${UID} \
             --build-arg GID=${UID} .