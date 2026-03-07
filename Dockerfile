# Use the official Python dense image for minimal footprint
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose port (standard uvicorn port)
EXPOSE 8000

# Start Uvicorn cleanly
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
