#!/bin/bash

# Check if Gemini API Key is provided
if [ -z "$1" ]; then
    echo "Usage: ./inject_secrets.sh <YOUR_GEMINI_API_KEY>"
    exit 1
fi

GEMINI_API_KEY=$1

echo "🚀 Injecting Gemini API Key into Kubernetes..."

# 1. Create or Update the manual secret
kubectl create secret generic generation-secrets-manual \
  --from-literal=GEMINI_API_KEY="$GEMINI_API_KEY" \
  --dry-run=client -o yaml | \
  kubectl annotate --local -f - argocd.argoproj.io/compare-options=IgnoreExtraneous -o yaml | \
  kubectl apply -f -

# 2. Restart the generation service to pick up the new secret
echo "♻️ Restarting generation-service..."
kubectl rollout restart deployment generation-service

echo "✅ Secret injection complete! Check logs with: kubectl logs -l app=generation-service -f"
