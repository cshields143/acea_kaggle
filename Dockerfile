FROM tensorflow/tensorflow:latest-jupyter
LABEL maintainer="christopher.shields143@gmail.com"
SHELL ["/bin/bash", "-c"]
RUN mkdir /tf/app
WORKDIR /tf/app
