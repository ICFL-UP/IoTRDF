FROM python:3.10-slim

# --- Enable inspection tools for namespace demo (ps, ip commands) ---
RUN apt-get update && \
    apt-get install -y procps iproute2 && \
    rm -rf /var/lib/apt/lists/*

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Copy app code and models into the image (models can be overridden via -v)
COPY app /app
COPY models /models

# Install dependencies
RUN pip install --no-cache-dir -r /app/requirements.txt

EXPOSE 8000

# Start API server
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]

