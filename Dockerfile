FROM python:3.12-slim

WORKDIR /app

# Install system dependencies for PyMuPDF
RUN apt-get update && apt-get install -y --no-install-recommends \
    libmupdf-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy source and install as package (eliminates sys.path hacks)
COPY pyproject.toml .
COPY requirements.txt .
COPY kg/ kg/
COPY web/ web/
COPY bin/ bin/

RUN pip install --no-cache-dir .

# Cloud Run sets PORT env var
ENV PORT=8080

CMD ["sh", "-c", "uvicorn web.app:app --host 0.0.0.0 --port $PORT"]
