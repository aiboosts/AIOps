# Use a plain Python 3.9 image
FROM python:3.13-slim

# Set the working directory
WORKDIR /app

# Install curl and dependencies
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# Install the necessary dependencies
RUN pip install --no-cache-dir \
    tensorflow==2.19.0 \
    requests \
    numpy \
    scikit-learn

# Copy the Python script and model file into the container
COPY cpu_usage_prediction.py /app/cpu_usage_prediction.py
COPY cpu_usage_model.h5 /app/cpu_usage_model.h5

# Command to run the script
CMD ["python", "cpu_usage_prediction.py"]