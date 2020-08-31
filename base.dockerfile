FROM nvidia/cuda

WORKDIR /usr/src
COPY requirements.txt .

RUN ls -l
RUN apt update && apt install python3.7 python3-pip -y
RUN pip3 install -r requirements.txt

ENV LANG C.UTF-8
ENV LANGUAGE C:en
ENV LC_ALL C.UTF-8