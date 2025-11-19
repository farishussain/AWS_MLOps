# Use the official TensorFlow runtime as base image
FROM tensorflow/tensorflow:2.15.0-py3

# Set working directory
WORKDIR /app

# Install additional requirements
RUN pip install --no-cache-dir \
    google-cloud-storage==2.10.0 \
    google-cloud-aiplatform==1.37.0 \
    scikit-learn==1.3.0 \
    pandas==2.0.3 \
    numpy==1.24.3

# Copy training script
COPY train.py /app/train.py

# Make script executable
RUN chmod +x /app/train.py

# Set the entrypoint
ENTRYPOINT ["python", "/app/train.py"]