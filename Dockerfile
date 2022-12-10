FROM python:3.8

WORKDIR /usr/src/app

RUN mkdir onnx2torch

COPY . . && cd onnx2torch

RUN pip install pipenv -i https://pypi.tuna.tsinghua.edu.cn/simple
