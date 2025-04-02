FROM python:3.11-slim

# Set the working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    g++ \
    build-essential \
    && rm -rf /var/lib/apt/lists/*  # Clean up to reduce image size

# Copy the application files
COPY . .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Define the command to run the application
CMD ["python", "src/main.py"]