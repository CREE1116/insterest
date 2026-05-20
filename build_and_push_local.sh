#!/bin/bash

SERVICES=("auth_service" "upload_service" "generation_service" "interaction_service" "comment_service" "user_service" "recommendation_service" "frontend" "console_frontend")
IMAGE_NAMES=("auth-service" "upload-service" "generation-service" "interaction-service" "comment-service" "user-service" "recommendation-service" "frontend" "console-frontend")
DOCKERHUB_USER="cree1116"

echo "🚀 Starting local build and push for all services..."

for i in "${!SERVICES[@]}"; do
  service_path="${SERVICES[$i]}"
  image_name="${IMAGE_NAMES[$i]}"
  
  echo "------------------------------------------------"
  echo "📦 Building $image_name from $service_path..."
  echo "------------------------------------------------"
  
  docker build -t $DOCKERHUB_USER/$image_name:latest ./$service_path
  
  if [ $? -eq 0 ]; then
    echo "✅ Build successful. Pushing to Docker Hub..."
    docker push $DOCKERHUB_USER/$image_name:latest
  else
    echo "❌ Build failed for $image_name. Skipping push."
  fi
done

echo "------------------------------------------------"
echo "✨ All done! Checking pod status in 10 seconds..."
sleep 10
kubectl get pods
