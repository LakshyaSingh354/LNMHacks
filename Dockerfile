# Use Python base image
FROM python:3.9-slim

# Set working directory
WORKDIR /app


# Copy requirements first (to leverage Docker cache)
COPY requirements.txt .
COPY .env .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .


# Expose the port the app runs on
EXPOSE 80

# Command to run the application
CMD ["uvicorn", "agent:app", "--host", "0.0.0.0", "--port", "80"]