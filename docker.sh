#!/bin/bash

# Define variables
IMAGE_NAME="challenge_image"
CONTAINER_NAME="challenge_container"

# Build the Docker image
docker build -t $IMAGE_NAME .


# Run the Docker container
docker run --rm --name $CONTAINER_NAME -it --ipc=host -v $(pwd):/app/ $IMAGE_NAME # Add --gpus all if you have a gpu, i personally have MPS which is not supported
