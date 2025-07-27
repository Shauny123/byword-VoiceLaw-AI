#!/bin/bash

# Cloud Run deployment script for byword-voicelaw-ai

PROJECT_ID="durable-trainer-466014-h8"
SERVICE_NAME="byword-voicelaw-ai"
REGION="us-central1"
IMAGE_NAME="gcr.io/$PROJECT_ID/$SERVICE_NAME"

echo "Building and deploying $SERVICE_NAME to Cloud Run..."

# Build the Docker image
echo "Building Docker image..."
docker build -t $IMAGE_NAME .

# Push to Google Container Registry
echo "Pushing to GCR..."
docker push $IMAGE_NAME

# Deploy to Cloud Run with explicit configuration
echo "Deploying to Cloud Run..."
gcloud run deploy $SERVICE_NAME \
  --image $IMAGE_NAME \
  --region $REGION \
  --platform managed \
  --allow-unauthenticated \
  --set-env-vars PORT=8080 \
  --port 8080 \
  --memory 512Mi \
  --cpu 1 \
  --timeout 300s \
  --concurrency 80 \
  --min-instances 0 \
  --max-instances 20

echo "Deployment complete!"
echo "Service URL: https://$SERVICE_NAME-1050086748568.$REGION.run.app"
