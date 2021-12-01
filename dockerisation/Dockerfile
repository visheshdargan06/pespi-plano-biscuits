FROM nvidia/cuda:11.0-base
CMD nvidia-smi
FROM python:3.8
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
COPY install_cuda.sh install_cuda.sh
COPY main.py main.py
RUN python main.py


#Successfully built a0d141ae1bd5
# Successfully tagged cuda-gpu:latest
# further requires manual cudnn installation - refer to install_cuda.sh
# installed cuda in 29e43ef7860a #FINAL