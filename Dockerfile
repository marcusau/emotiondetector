FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /app

# These are crucial for dlib to compile correctly
RUN apt-get update -y && \
    apt-get install -y \
    build-essential \
    gcc \ 
    g++ \ 
    make \
    cmake \
    pkg-config \
    libx11-dev \
    libopenblas-dev  \
    liblapack-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \ 
    libhdf5-dev \
    libboost-python-dev \
    libboost-thread-dev \
    libboost-system-dev \
    libboost-date-time-dev \
    libboost-serialization-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file (if any) and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY models.py .
COPY predictor.py .
COPY config.py .
COPY app.py .
COPY util.py .
COPY setting.yaml .
COPY svm_model.pkl .

# Expose the port the app will run on
EXPOSE 8000

# Start the application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000","--workers", "1", "--log-level", "info","--loop","uvloop","--http","httptools"]