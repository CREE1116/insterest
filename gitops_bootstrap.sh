#!/bin/bash
set -e

echo "🚀 Starting GitOps Bootstrap with ArgoCD..."

# 1. 아르고CD 설치
echo "📦 Installing ArgoCD..."
kubectl create namespace argocd || true
kubectl apply -n argocd -f https://raw.githubusercontent.com/argoproj/argo-cd/stable/manifests/install.yaml

# 2. Ingress Controller 설치 (로컬 접속용)
echo "🌐 Installing Ingress Controller (Nginx)..."
kubectl apply -f https://raw.githubusercontent.com/kubernetes/ingress-nginx/main/deploy/static/provider/cloud/deploy.yaml

# 2. 아르고CD 서버가 뜰 때까지 대기
echo "⏳ Waiting for ArgoCD server to be ready..."
kubectl wait --for=condition=available --timeout=300s deployment/argocd-server -n argocd

echo "⏳ Waiting for Ingress Controller to be ready..."
kubectl wait --namespace ingress-nginx \
  --for=condition=ready pod \
  --selector=app.kubernetes.io/component=controller \
  --timeout=300s

# 3. Insterest 인프라 레포지토리를 바라보는 ArgoCD App 생성
echo "🔗 Linking GitHub Infra Repository to ArgoCD..."
cat << 'EOF' | kubectl apply -f -
apiVersion: argoproj.io/v1alpha1
kind: Application
metadata:
  name: insterest-app
  namespace: argocd
spec:
  project: default
  source:
    repoURL: 'https://github.com/CREE1116/insterest-infra.git'
    targetRevision: master
    path: k8s
    directory:
      recurse: true
  destination:
    server: 'https://kubernetes.default.svc'
    namespace: default
  syncPolicy:
    automated:
      prune: true
      selfHeal: true
    syncOptions:
      - CreateNamespace=true
EOF

# 4. 초기 비밀번호 가져오기 (시크릿이 생성될 시간을 약간 부여)
echo "🔑 Fetching ArgoCD Admin Password..."
sleep 5
PASSWORD=$(kubectl -n argocd get secret argocd-initial-admin-secret -o jsonpath="{.data.password}" | base64 -d)

# 5. 백그라운드 자동 포트포워딩
echo "🔌 Starting automatic port-forwarding in the background..."
pkill -f "kubectl port-forward" || true
kubectl port-forward svc/argocd-server -n argocd 8080:443 > /dev/null 2>&1 &
kubectl port-forward -n ingress-nginx service/ingress-nginx-controller 80:80 > /dev/null 2>&1 &

echo "✅ GitOps Bootstrap Complete!"
echo "-------------------------------------------------"
echo "🐙 아르고CD가 성공적으로 설치되었고, 포트포워딩이 켜졌습니다."
echo "🌍 타겟 레포지토리: https://github.com/CREE1116/insterest-infra"
echo ""
echo "📊 ArgoCD 대시보드 접속 방법:"
echo "1. 브라우저 접속: https://localhost:8080 (주의: 반드시 https로 접속)"
echo "2. 아이디: admin"
echo "3. 비밀번호: $PASSWORD"
echo "-------------------------------------------------"
