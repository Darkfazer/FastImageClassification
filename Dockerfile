# Use an official Python runtime as a parent image
FROM python:3.8-slim

# Install system dependencies needed by OpenCV and other packages
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libopencv-dev \
    git

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install a compatible OpenCV version (before FastImageClassification)
RUN pip install opencv-python==4.5.5.62

# Install the FastImageClassification from GitHub
RUN pip install --no-deps git+https://github.com/CVxTz/FastImageClassification

# Expose the port the app runs on
EXPOSE 8080

# Command to run the app using Uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]