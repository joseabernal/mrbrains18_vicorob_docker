FROM nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04

RUN apt-get update && apt-get install -y \
    aufs-tools \
    automake \
    apt-utils \
    build-essential \
    libsqlite3-dev \
    wget \
    openssl \
    unzip \
 && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y \
    python \
    python-setuptools \
    python-pip
RUN pip install --upgrade pip

RUN wget -O- http://neuro.debian.net/lists/xenial.de-md.full | tee /etc/apt/sources.list.d/neurodebian.sources.list
RUN apt-key adv --recv-keys --keyserver hkp://pool.sks-keyservers.net:80 0xA5D32F012649A5A9
RUN apt-get update && apt-get install fsl -y

RUN mkdir -p ./QCFiles
COPY ./QCFiles/ ./QCFiles

RUN apt-get update -y \
 && mkdir /mcr-install \
 && cd /mcr-install \
 && cp /QCFiles/MCR_R2014b_glnxa64_installer.zip /mcr-install \
# && wget -nv http://ssd.mathworks.com/supportfiles/downloads/R2014b/deployment_files/R2014b/installers/glnxa64/MCR_R2014b_glnxa64_installer.zip \
 && unzip MCR_R2014b_glnxa64_installer.zip \
 && ./install -mode silent -agreeToLicense yes \
 && rm -Rf /mcr-install

ENV MCRROOT=/usr/local/MATLAB/MATLAB_Compiler_Runtime/v84
ENV LD_LIBRARY_PATH $LD_LIBRARY_PATH:/usr/lib/fsl/5.0
ENV MCR_LD_LIBRARY_PATH $MCR_LD_LIBRARY_PATH:$MCRROOT/runtime/glnxa64
ENV MCR_LD_LIBRARY_PATH $MCR_LD_LIBRARY_PATH:$MCRROOT/bin/glnxa64
ENV MCR_LD_LIBRARY_PATH $MCR_LD_LIBRARY_PATH:$MCRROOT/sys/os/glnxa64
ENV MCR_LD_LIBRARY_PATH $MCR_LD_LIBRARY_PATH:$MCRROOT/sys/java/jre/glnxa64/jre/lib/amd64
ENV MCR_CACHE_VERBOSE=true
ENV MCR_CACHE_ROOT=/tmp

RUN pip install --upgrade pip
RUN pip install  -r ./QCFiles/requirements.txt

RUN ["chmod", "+x", "./QCFiles/tools/run_spm12.sh"]

ENV HOME /QCFiles

WORKDIR /QCFiles/

ENTRYPOINT python -u main.py
