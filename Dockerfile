# Multi-Agent RFP Intelligence Platform Dockerfile
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies for PyMuPDF and pdfplumber
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency requirements
COPY requirements.txt .

# Install Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Expose Streamlit (8501) and FastAPI (8000) ports
EXPOSE 8501 8000

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0

# Default command: launch platform via run.py
CMD ["python", "run.py"]
