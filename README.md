# SLLMs on Kubernetes

Deploy Small Large Language Models (SLLMs) on Kubernetes using FastAPI. This repository provides two deployment options: a standard PyTorch-based API and an ONNX-optimized version for faster inference.

## Features

- FastAPI-based REST API for text generation
- Two deployment variants:
  - **PyTorch**: Standard Hugging Face Transformers implementation
  - **ONNX**: Optimized runtime for faster inference
- Health check endpoints (`/health`, `/ready`) for Kubernetes probes
- Configurable via environment variables
- Kubernetes manifests with liveness/readiness probes
- Uses [EleutherAI/gpt-neo-125M](https://huggingface.co/EleutherAI/gpt-neo-125M) model (configurable)

## Project Structure

```
llm-kubernetes/
├── fast-api/                 # PyTorch-based API
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── run.py
│   └── k8s/
│       ├── namespace.yaml
│       ├── deployment.yaml
│       └── service.yaml
└── onnx-fast-api/            # ONNX-optimized API
    ├── Dockerfile
    ├── requirements.txt
    ├── run.py
    ├── conversion.py         # Script to convert model to ONNX
    └── k8s/
        ├── namespace.yaml
        ├── deployment.yaml
        └── service.yaml
```

## Requirements

- Docker
- Minikube (or any Kubernetes cluster v1.21+)
- kubectl
- Python 3.9+ (for local development and ONNX conversion)

## Environment Setup

### 1. Start Minikube

```bash
# Start Minikube with recommended resources for LLM workloads
minikube start --cpus=4 --memory=8192 --driver=docker

# Verify Minikube is running
minikube status
```

### 2. Configure Docker Environment

To build Docker images directly in Minikube's Docker daemon (avoiding the need to push to a registry):

```bash
# Configure shell to use Minikube's Docker daemon
eval $(minikube docker-env)

# Verify you're using Minikube's Docker
docker info | grep -i name
```

> **Note:** Run `eval $(minikube docker-env)` in each new terminal session, or add it to your shell profile.

### 3. Verify Setup

```bash
# Check kubectl can connect to the cluster
kubectl cluster-info

# Check nodes are ready
kubectl get nodes

# Check Docker is working
docker ps
```

### Useful Minikube Commands

```bash
# Stop Minikube
minikube stop

# Delete Minikube cluster
minikube delete

# Open Kubernetes dashboard
minikube dashboard

# Get Minikube IP (for accessing services)
minikube ip

# Access a LoadBalancer service
minikube service <service-name> -n <namespace>
```

## Getting Started

### Clone the Repository

```bash
git clone https://github.com/yourusername/llm-kubernetes.git
cd llm-kubernetes
```

### Option 1: PyTorch Deployment

Build and deploy the standard PyTorch-based API:

```bash
# Make sure you're using Minikube's Docker daemon
eval $(minikube docker-env)

# Build the Docker image
docker build -t llm-api ./fast-api

# Deploy to Kubernetes
kubectl apply -f fast-api/k8s/namespace.yaml
kubectl apply -f fast-api/k8s/deployment.yaml
kubectl apply -f fast-api/k8s/service.yaml

# Wait for the pod to be ready
kubectl wait --for=condition=ready pod -l app=llm -n llm --timeout=300s

# Access the service
# Option A: Use minikube service (opens browser with auto-assigned port)
minikube service llm -n llm

# Option B: Use port-forward (access at http://localhost:8000)
kubectl port-forward -n llm svc/llm 8000:80
```

### Option 2: ONNX Deployment

For optimized inference, use the ONNX version:

```bash
# Make sure you're using Minikube's Docker daemon
eval $(minikube docker-env)

# First, convert the model to ONNX format
cd onnx-fast-api
pip install optimum[exporters]
python conversion.py

# Build the Docker image
docker build -t llm-api-onnx .

# Deploy to Kubernetes
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml

# Wait for the pod to be ready
kubectl wait --for=condition=ready pod -l app=llm-onnx -n llm-onnx --timeout=300s

# Access the service
# Option A: Use minikube service (opens browser with auto-assigned port)
minikube service llm-onnx -n llm-onnx

# Option B: Use port-forward (access at http://localhost:8000)
kubectl port-forward -n llm-onnx svc/llm-onnx 8000:80
```

### Verify Deployment

```bash
# Check pod status
kubectl get pods -n llm          # For PyTorch
kubectl get pods -n llm-onnx     # For ONNX

# Check logs
kubectl logs -f deployment/llm -n llm          # For PyTorch
kubectl logs -f deployment/llm-onnx -n llm-onnx  # For ONNX

# Get service URL
minikube service llm -n llm --url          # For PyTorch
minikube service llm-onnx -n llm-onnx --url  # For ONNX
```

## API Usage

Both deployments expose endpoints for text generation and health checks.

### Generate Text

```
POST /generate
```

#### Request Body (PyTorch)

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `prompt` | string | required | The input text to continue |
| `max_tokens` | int | 50 | Maximum tokens to generate (1-512) |
| `temperature` | float | 1.0 | Sampling temperature (0.1-2.0) |
| `top_k` | int | 50 | Top-k sampling parameter (1-100) |
| `top_p` | float | 0.95 | Top-p nucleus sampling (0.1-1.0) |

#### Request Body (ONNX)

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `prompt` | string | required | The input text to continue |
| `max_tokens` | int | 30 | Maximum tokens to generate (1-512) |
| `top_k` | int | 10 | Top-k sampling parameter (1-100) |

#### Example Request

```bash
# Using port-forward (kubectl port-forward -n llm svc/llm 8000:80)
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "The future of AI is", "max_tokens": 50, "temperature": 0.8}'
```

#### Example Response

```json
{
  "generated_text": "The future of AI is bright and full of possibilities..."
}
```

### Health Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Liveness probe - checks if service is running |
| `/ready` | GET | Readiness probe - checks if model is loaded |

```bash
# Check health (using port-forward on port 8000)
curl http://localhost:8000/health

# Check readiness
curl http://localhost:8000/ready
```

### API Documentation

FastAPI provides interactive documentation at (using port-forward):
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Configuration

### Environment Variables

#### PyTorch Service

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_ID` | `EleutherAI/gpt-neo-125M` | Hugging Face model ID |
| `DEFAULT_MAX_TOKENS` | `50` | Default max tokens for generation |
| `DEFAULT_TEMPERATURE` | `1.0` | Default sampling temperature |
| `DEFAULT_TOP_K` | `50` | Default top-k value |
| `DEFAULT_TOP_P` | `0.95` | Default top-p value |
| `DEFAULT_REPETITION_PENALTY` | `1.2` | Repetition penalty |

#### ONNX Service

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_ID` | `EleutherAI/gpt-neo-125M` | Hugging Face model ID (for tokenizer) |
| `MODEL_PATH` | `model/model.onnx` | Path to ONNX model file |
| `DEFAULT_MAX_TOKENS` | `30` | Default max tokens for generation |
| `DEFAULT_TOP_K` | `10` | Default top-k value |

### Kubernetes Configuration

Environment variables can be configured in the deployment manifests under `spec.template.spec.containers[].env`.

## Resource Configuration

Default resource limits per pod:

| Resource | Request | Limit |
|----------|---------|-------|
| CPU | 500m | 1 |
| Memory | 1Gi | 2Gi |

Adjust these values in the deployment manifests based on your needs.

### Kubernetes Probes

Both deployments include liveness and readiness probes:

- **Liveness Probe**: Checks `/health` endpoint to restart unhealthy pods
- **Readiness Probe**: Checks `/ready` endpoint to route traffic only when model is loaded

## Troubleshooting

### Pod stuck in ImagePullBackOff

This usually means the Docker image wasn't built in Minikube's Docker daemon:

```bash
# Configure Docker to use Minikube
eval $(minikube docker-env)

# Rebuild the image
docker build -t llm-api ./fast-api
```

### Pod stuck in Pending

Check if there are enough resources:

```bash
kubectl describe pod -n llm
```

If resources are insufficient, increase Minikube resources:

```bash
minikube stop
minikube delete
minikube start --cpus=4 --memory=8192
```

### Service not accessible

For LoadBalancer services in Minikube, use the tunnel or service command:

```bash
# Option 1: Use minikube service
minikube service llm -n llm

# Option 2: Use minikube tunnel (run in separate terminal)
minikube tunnel
```

### Model loading takes too long

The first deployment may take time as the model downloads. Check logs:

```bash
kubectl logs -f deployment/llm -n llm
```

### Reset everything

```bash
# Delete deployments
kubectl delete -f fast-api/k8s/
kubectl delete -f onnx-fast-api/k8s/

# Or reset Minikube entirely
minikube delete
minikube start --cpus=4 --memory=8192
```

## Contributing

Contributions are welcome! Please open issues or pull requests for improvements.

## License

This project is licensed under the MIT License.
