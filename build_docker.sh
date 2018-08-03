#!/bin/bash
#Delete all containers
echo 'Removing any previous containers'
sudo docker rm $(docker ps -a -q)

# Delete all images
# echo 'Delete all previous images'
# docker rmi $(docker images -q)

#Delete the previous lr_model_img
echo 'Removing framework_img image'
sudo docker rmi -f mrbrains18

#Build the lr_model_img from the dockerfile
echo 'Building framework_img image'
sudo nvidia-docker build -t mrbrains18 .
