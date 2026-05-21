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
COPY configs/ configs/

RUN pip install --no-cache-dir .

# Research data should be mounted at runtime, e.g.:
#   docker run -v /path/to/data:/app/data -e INSTINCT_DATA_DIR=/app/data
ENV INSTINCT_DATA_DIR=/app/data
ENV PORT=8080

CMD ["sh", "-c", "uvicorn web.app:app --host 0.0.0.0 --port $PORT"]
