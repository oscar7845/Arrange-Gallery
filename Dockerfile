# Use Python 3.8 slim-buster as base image
FROM python:3.8-slim-buster

# Set environment variables
ENV DEBIAN_FRONTEND noninteractive

# Install the necessary dependencies
RUN apt-get update && apt-get install -y \
    tesseract-ocr tesseract-ocr-eng \
    build-essential \
    cmake \
    git \
    libatlas-base-dev \
    libgtk-3-dev \
    libboost-all-dev \
    python3-dev \
    python3-setuptools \
    python3-wheel \
    libtesseract-dev \
    libleptonica-dev \
    pkg-config \
    libmagickwand-dev \
    && rm -rf /var/lib/apt/lists/* \
    && pip3 install --upgrade pip

# Set the working directory
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install the Python packages
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy the application code into the container
COPY . .
