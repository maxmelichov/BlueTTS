# Before building:
#
#   1. Download model weights (from repo root):
#      uv run hf download notmax123/blue-onnx --repo-type model --local-dir ./onnx_models \
#          --exclude "voices/all_voices/**"
#      optional 2000+ voices: uv run hf download notmax123/blue-onnx voices/all_voices/ \
#          --repo-type model --local-dir ./onnx_models
#      wget -O model.onnx https://huggingface.co/thewh1teagle/renikud/resolve/main/model.onnx
#      (voices/ and config/tts.json ship with this repo)
#
#   2. Build and run:
#      docker build -t blue-tts .
#      docker run --rm blue-tts
#      docker run --rm blue-tts --lang en --text "Hello"

FROM python:3.12-slim

WORKDIR /app

# System deps: soundfile (libsndfile), phonemizer / espeak-ng-loader (espeak-ng)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 curl espeak-ng \
    && rm -rf /var/lib/apt/lists/*

# Pin uv for reproducible image builds (bump when upgrading uv in CI/local)
COPY --from=ghcr.io/astral-sh/uv:0.11.6 /uv /usr/local/bin/uv

# Install python deps (lockfile matches requires-python >=3.12)
COPY pyproject.toml uv.lock /app/
RUN uv sync --frozen --no-dev --no-install-project

# Copy package + app + models + assets
COPY src/ /app/src/
COPY examples/ /app/examples/
COPY onnx_models /app/onnx_models
COPY model.onnx /app/model.onnx
COPY voices /app/voices
COPY config/tts.json /app/tts.json

# Runtime environment
ENV PYTHONUNBUFFERED=1 \
    ORT_DISABLE_CPU_AFFINITY=1 \
    ORT_INTRA=4 \
    ORT_INTER=1

ENTRYPOINT ["uv", "run", "python", "-u", "examples/app.py"]
CMD ["--text", "שלום עולם"]
