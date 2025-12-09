#!/usr/bin/env bash
set -euo pipefail

# Build image
docker build -t irdf-infer:latest .

# Create models dir if not already
mkdir -p models

# Choose which set to serve by default: policy | baseline
IRDF_SET="${1:-policy}"

docker run --rm \
  --name irdf \
  --cpus=0.25 \
  --memory=256m \
  -p 8000:8000 \
  -e IRDF_MODEL_SET="${IRDF_SET}" \
  -v "$(pwd)/models":/models:ro \
  irdf-infer:latest
