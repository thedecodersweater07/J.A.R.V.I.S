FROM python:3.11.7-slim-bookworm AS builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y \
    --no-install-recommends \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy and install requirements
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Final stage
FROM python:3.11.7-slim-bookworm

WORKDIR /app

# Create non-root user
RUN groupadd -r jarvis && useradd -r -g jarvis jarvis \
    && chown -R jarvis:jarvis /app

# Copy only runtime dependencies
COPY --from=builder /root/.local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
RUN apt-get update && apt-get install -y \
    --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy application code
COPY --chown=jarvis:jarvis . .

# Set proper permissions
RUN chmod -R 755 /app

# Switch to non-root user
USER jarvis

# Add healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')" || exit 1

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/app:${PATH}"

CMD ["python", "main.py"]
