# Dockerfile for Sign0 FastAPI Backend
FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend files and ONNX models
COPY backend/ ./backend/

EXPOSE 8000

ENV PORT=8000
ENV KMP_DUPLICATE_LIB_OK=TRUE

CMD ["python", "-m", "uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "8000"]
