FROM ubuntu:xenial-20180228@sha256:e348fbbea0e0a0e73ab0370de151e7800684445c509d46195aef73e090a49bd6

LABEL maintainer="Olivier Nguyen <nguyenolive@gmail.com>"

ARG KERAS_VERSION=2.1.5
ARG TENSORFLOW_VERSION=1.7.0
ARG TENSORFLOW_ARCH=cpu

ENV PASSWORD ''

USER root

# Install all OS dependencies for fully functional notebook server
RUN apt-get update && apt-get install -yq --no-install-recommends \
		libav-tools \
    build-essential \
    git \
    netcat \
    pandoc \
    unzip \
    nano \
    curl \
    libfreetype6-dev \
    libhdf5-serial-dev \
    libpng12-dev \
    libzmq3-dev \
    pkg-config \
    python-dev \
    python-setuptools \
    rsync \
    software-properties-common \
    && apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN curl -O https://bootstrap.pypa.io/get-pip.py && \
    python get-pip.py && \
    rm get-pip.py

# Install TensorFlow
RUN pip --no-cache-dir install \
	https://storage.googleapis.com/tensorflow/linux/${TENSORFLOW_ARCH}/tensorflow-${TENSORFLOW_VERSION}-cp27-none-linux_x86_64.whl

#RUN ln -s -f /usr/bin/python3 /usr/bin/python

ADD requirements.txt /
RUN pip --no-cache-dir install -r requirements.txt && \
		python -m ipykernel.kernelspec

# Install Twitter tokenizer
RUN	pip install git+https://github.com/erikavaris/tokenizer.git

# Install Keras
RUN pip --no-cache-dir install git+git://github.com/fchollet/keras.git@${KERAS_VERSION}

#USER $NB_UID

COPY jupyter_notebook_config.py /root/.jupyter/
COPY run_jupyter.sh /

RUN ["mkdir", "pass"]
COPY . /pass
 
###
### start up the server
###
EXPOSE 8888 6006

WORKDIR /pass/notebooks
#CMD ["/run_jupyter.sh", "--ip=0.0.0.0", "--allow-root"]
CMD ["/run_jupyter.sh", "--allow-root"]

### Usage
#docker build -t pass .
#docker run -p 127.0.0.1:8888:8888 --name pass -t pass
#docker stop pass
#docker rm pass
#docker stop pass; docker rm pass; docker build -t pass .; docker run -p 127.0.0.1:8888:8888 --name pass -t pass
