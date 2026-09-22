FROM python:3.11-slim

# Install system dependencies: Tesseract OCR, ffmpeg, and OpenCV runtime libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend application code and traineddata
COPY . .

# Ensure image and template directories exist
RUN mkdir -p images test-assets templates

# Render automatically binds to $PORT (typically 10000)
ENV PORT=10000
EXPOSE 10000

CMD ["sh", "-c", "python -m uvicorn endpoints:app --host 0.0.0.0 --port ${PORT:-10000}"]
