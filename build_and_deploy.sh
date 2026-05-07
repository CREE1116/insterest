#!/bin/bash
TAG=${1:-"v32"}
CLUSTER_NAME="desktop"
echo "🚀 Starting Robust Interest MSA Deployment (Tag: $TAG)..."

# Enable Docker BuildKit for better caching
export DOCKER_BUILDKIT=1

# 1. Image and Directory Mapping
services=("auth-service" "upload-service" "generation-service" "interaction-service" "comment-service" "user-service" "recommendation-service" "frontend")
dirs=("auth_service" "upload_service" "generation_service" "interaction_service" "comment_service" "user_service" "recommendation_service" "frontend")
# Postgres version updated to 16-alpine to match latest postgres.yaml
base_images=("postgres:16-alpine" "redis:7-alpine" "apache/kafka:3.7.0")

# 2. Build Docker Images with Caching
echo "📦 Building Application Docker images (using cache)..."
for i in "${!services[@]}"; do
    service=${services[$i]}
    dir=${dirs[$i]}
    echo "  🔨 Building $service:$TAG..."
    docker build -t "$service:$TAG" "$dir" --build-arg BUILDKIT_INLINE_CACHE=1
done

# 3. Load images into kind cluster
if docker ps --format '{{.Names}}' | grep -q "$CLUSTER_NAME-control-plane"; then
    echo "📥 Checking and Loading images into kind cluster ($CLUSTER_NAME)..."
    
    # containerd image check helper
    check_image() {
      docker exec "$CLUSTER_NAME-control-plane" ctr -n k8s.io images ls | grep -q "$1"
    }

    echo "  📦 Loading Application Images..."
    for service in "${services[@]}"; do
        if check_image "$service:$TAG"; then
            echo "    ✅ $service:$TAG already exists in cluster, skipping."
            continue
        fi

        echo "    🔄 Loading $service:$TAG..."
        if command -v kind &> /dev/null; then
            kind load docker-image "$service:$TAG" --name "$CLUSTER_NAME"
        else
            docker save "$service:$TAG" | docker exec -i "$CLUSTER_NAME-control-plane" ctr -n k8s.io images import -
        fi
    done
    
    echo "  📦 Loading Base Infrastructure Images..."
    for img in "${base_images[@]}"; do
        # containerd often prefixes with docker.io/library/
        if check_image "$img"; then
            echo "    ✅ $img already exists in cluster, skipping."
            continue
        fi

        echo "    󰚰 Pulling $img since it's not in local docker daemon or cluster..."
        docker pull "$img"
        
        echo "    📥 Loading $img..."
        if command -v kind &> /dev/null; then
            kind load docker-image "$img" --name "$CLUSTER_NAME"
        else
            docker save "$img" | docker exec -i "$CLUSTER_NAME-control-plane" ctr -n k8s.io images import -
        fi
    done
fi

# 4. Apply Base Infrastructure
echo "☸️ Applying Base Infrastructure..."

if ! kubectl get namespace ingress-nginx &> /dev/null; then
    echo "🌐 Installing Ingress Controller (Nginx)..."
    kubectl apply -f https://raw.githubusercontent.com/kubernetes/ingress-nginx/main/deploy/static/provider/cloud/deploy.yaml
    echo "⏳ Waiting for Ingress Controller to be ready..."
    kubectl wait --namespace ingress-nginx \
      --for=condition=ready pod \
      --selector=app.kubernetes.io/component=controller \
      --timeout=180s
fi

kubectl apply -f infra/k8s/db/postgres.yaml
kubectl apply -f infra/k8s/db/kafka.yaml
kubectl apply -f infra/k8s/db/redis.yaml

echo "⏳ Applying Ingress configuration with retry..."
for i in {1..10}; do
    if kubectl apply -f infra/k8s/ingress.yaml; then
        echo "  ✅ Ingress applied successfully."
        break
    else
        echo "  ⏳ Ingress webhook not ready yet, retrying in 5s... $i/10"
        sleep 5
    fi
done

echo "⏳ Waiting for Base Infrastructure (DB, Redis, Kafka) to be available..."
kubectl wait --for=condition=available --timeout=300s deployment/postgres
kubectl wait --for=condition=available --timeout=300s deployment/redis
kubectl wait --for=condition=available --timeout=300s deployment/kafka

# 5. Apply Service Manifests
echo "☸️ Applying Application Manifests..."
kubectl apply -f infra/k8s/auth/deployment.yaml
kubectl apply -f infra/k8s/upload/deployment.yaml
kubectl apply -f infra/k8s/generation/deployment.yaml
kubectl apply -f infra/k8s/interaction/deployment.yaml
kubectl apply -f infra/k8s/comment/deployment.yaml
kubectl apply -f infra/k8s/user/deployment.yaml
kubectl apply -f infra/k8s/recommendation/deployment.yaml
kubectl apply -f infra/k8s/recommendation/cronjob.yaml
kubectl apply -f infra/k8s/frontend/deployment.yaml

# 6. Patch and Restart
echo "🛠 Patching and Refreshing deployments to $TAG..."
k_patch() {
  kubectl patch deployment $1 --patch "{\"spec\": {\"template\": {\"spec\": {\"containers\": [{\"name\": \"$2\", \"image\": \"$3:$TAG\"}]}}}}"
}

k_patch auth-app auth-app auth-service
k_patch upload-service upload-service upload-service
k_patch generation-service generation-service generation-service
k_patch interaction-service interaction-service interaction-service
k_patch comment-service comment-service comment-service
k_patch user-app user-app user-service
k_patch recommendation-service recommendation-service recommendation-service
k_patch frontend frontend frontend

kubectl rollout restart deployment auth-app upload-service generation-service interaction-service comment-service user-app recommendation-service frontend

echo "⏳ Waiting for recommendation-service to be ready for sync..."
kubectl wait --for=condition=available --timeout=300s deployment/recommendation-service

# 7. Port Forwarding
echo "🔌 Setting up Port Forwarding..."
pkill -f "port-forward" || true
kubectl port-forward -n ingress-nginx service/ingress-nginx-controller 80:80 > /dev/null 2>&1 &

for i in {1..10}; do
    if nc -z localhost 80; then
        echo "  ✅ Port 80 is ready."
        break
    fi
    echo "  ⏳ Waiting for port 80..."
    sleep 2
done

# 8. Trigger Backfill and Training with Updated API paths
echo "🔄 Triggering one-time 128-dim embedding backfill..."
for i in {1..5}; do
    echo "  🚀 Backfill attempt $i..."
    # Updated path from /intel/backfill to /discovery/sync
    if curl -s -f -X POST http://localhost/api/v1/discovery/sync; then
        echo "  ✅ Backfill triggered successfully!"
        break
    else
        echo "  ⚠️  Failed to connect (attempt $i), retrying in 5s..."
        sleep 5
    fi
done

echo "⏳ Waiting 15s for service to stabilize before training..."
sleep 15

echo "🏋️ Triggering one-time model training..."
for i in {1..5}; do
    echo "  🚀 Training attempt $i..."
    # Updated path from /intel/train to /discovery/train
    if curl -s -f -X POST http://localhost/api/v1/discovery/train; then
        echo "  ✅ Training started successfully (Background)!"
        break
    else
        echo "  ⚠️  Failed to connect (attempt $i), retrying in 10s..."
        sleep 10
    fi
done

echo "✅ All services updated to $TAG!"
echo "🔥 Access platform at http://localhost"
