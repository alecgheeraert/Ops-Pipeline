FROM python:3.10-slim

RUN apt-get update && apt-get -y install qemu-user-static python3-opencv

WORKDIR /fastapi
COPY ./requirements.txt /fastapi
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r /fastapi/requirements.txt

RUN apt -y remove qemu-user-static

COPY outputs /fastapi
COPY main.py /fastapi

CMD uvicorn main:app --host=0.0.0.0 --port=80
