FROM python:3.9-slim

# Set the working directory in the container to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY serve/predict.py /app/

# Install Flask and google-cloud-storage
RUN pip install Flask google-cloud-storage joblib scikit-learn grpcio gcsfs python-dotenv

ENV AIP_STORAGE_URI=gs://mlops_pipeline/model
ENV AIP_HEALTH_ROUTE=/ping
ENV AIP_PREDICT_ROUTE=/predict
ENV AIP_HTTP_PORT=8080
ENV BUCKET_NAME=mlops_pipeline
ENV PROJECT_ID=solar-maker-407904


# Run serve.py when the container launches
ENTRYPOINT ["python", "predict.py"]