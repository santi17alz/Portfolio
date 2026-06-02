#FROM arm64v8/python:3.11-slim
FROM python:3.11-slim

WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
    iproute2 \
    procps \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir numpy psutil docker flask matplotlib

RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

COPY training/ ./training/
COPY telemetry/ ./telemetry/
COPY dashboard/ ./dashboard/
COPY scripts/ ./scripts/

CMD ["python3"]
