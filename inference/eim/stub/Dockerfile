# A stand-in for an EIM serving container, for exercising the container lifecycle without vLLM.
#
# Mirrors the parts of the real image ICN depends on: it listens on 8000, carries the same
# HEALTHCHECK shape, and runs with no arguments the way EIM's entrypoint defaults to `serve`.
FROM python:3.12-slim

WORKDIR /workspace
COPY server.py /workspace/server.py

ENV INFERENCE_MODEL_ID=""
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status==200 else 1)" || exit 1

ENTRYPOINT ["python3", "-u", "/workspace/server.py"]
