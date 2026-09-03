FROM python:3.11.13-slim
RUN pip install --no-cache-dir numpy==2.2.6 sympy==1.14.0
WORKDIR /work
ENV OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
