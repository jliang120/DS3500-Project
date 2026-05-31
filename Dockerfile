FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project
COPY . .

# Create necessary directories
RUN mkdir -p data_processing/data_storage/processed \
             data_processing/data_storage/cleaned \
             data_processing/data_storage/raw \
             data_acquisition/data_storage

# Expose Panel dashboard port
EXPOSE 5006

# Run pipeline first, then serve the dashboard
CMD ["sh", "-c", "cd data_processing && python pipeline.py && panel serve dashboard.py --address 0.0.0.0 --port 5006 --allow-websocket-origin=*"]