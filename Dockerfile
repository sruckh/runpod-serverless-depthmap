# All Python packages, upstream source and validation live in Docker stages.
# Nothing is installed on the source checkout host by these definitions.
FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04 AS base
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONSAFEPATH=1 \
    PATH=/opt/venv/bin:$PATH \
    PYTHONPATH=/app:/opt/lotus
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.12 python3.12-venv ca-certificates \
    && rm -rf /var/lib/apt/lists/*

FROM base AS dependencies
COPY requirements.txt /build/requirements.txt
RUN python3.12 -m venv /opt/venv \
    && python -m pip install --no-cache-dir pip==25.0.1 \
    && python -m pip install --no-cache-dir torch==2.8.0 torchvision==0.23.0 \
       --index-url https://download.pytorch.org/whl/cu128 \
    && python -m pip install --no-cache-dir -r /build/requirements.txt \
    && python -m pip check

FROM base AS lotus-source
# A reproducible revision, not whatever main happens to contain on a cold start.
ARG LOTUS_REVISION=b737e12238d2a41fc582ee8c6c6b1104d318d799
RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/* \
    && git init /opt/lotus \
    && git -C /opt/lotus remote add origin https://github.com/EnVision-Research/Lotus.git \
    && git -C /opt/lotus fetch --depth 1 origin "$LOTUS_REVISION" \
    && git -C /opt/lotus checkout --detach FETCH_HEAD \
    && test "$(git -C /opt/lotus rev-parse HEAD)" = "$LOTUS_REVISION" \
    && rm -rf /opt/lotus/.git

FROM base AS runtime
COPY --from=dependencies /opt/venv /opt/venv
COPY --from=lotus-source /opt/lotus /opt/lotus
WORKDIR /app
COPY src/ /app/
# No GPU, model weights, API keys, or network are needed for this import check.
RUN python -c "import handler; from pipeline import LotusDPipeline; from PIL import features; assert all(features.check(c) for c in ('jpg', 'zlib', 'webp'))"
CMD ["bash", "/app/bootstrap.sh"]

# CI builds this target before publishing. Dev tools do not ship in final.
FROM runtime AS quality
RUN apt-get update && apt-get install -y --no-install-recommends shellcheck \
    && rm -rf /var/lib/apt/lists/*
COPY requirements-dev.txt /build/requirements-dev.txt
RUN python -m pip install --no-cache-dir -r /build/requirements-dev.txt
COPY pyproject.toml /app/pyproject.toml
COPY tests/ /app/tests/
RUN ruff check /app \
    && mypy --config-file /app/pyproject.toml /app/handler.py /app/image_input.py /app/model_config.py \
    && shellcheck /app/bootstrap.sh /app/prefetch.sh \
    && bash -n /app/bootstrap.sh /app/prefetch.sh \
    && pytest -q -p no:cacheprovider /app/tests

FROM runtime AS final
