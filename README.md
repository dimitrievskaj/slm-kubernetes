# llm-kubernetes

Deploy Large Language Models (LLMs) on Kubernetes using FastAPI. This repository provides two deployment options: a standard PyTorch-based API and an ONNX-optimized version for faster inference.

## Features

- FastAPI-based REST API for text generation
- Two deployment variants:
  - **PyTorch**: Standard Hugging Face Transformers implementation
  - **ONNX**: Optimized runtime for faster inference
- Kubernetes manifests for easy deployment
- Uses [EleutherAI/gpt-neo-125M](https://huggingface.co/EleutherAI/gpt-neo-125M) model

## Project Structure

```
llm-kubernetes/
├── fast-api/                 # PyTorch-based API
│   ├── Dockerfile
│   ├── run.py
│   └── k8s/
│       ├── namespace.yaml
│       ├── deployment.yaml
│       └── service.yaml
└── onnx-fast-api/            # ONNX-optimized API
    ├── Dockerfile
    ├── run.py
    ├── conversion.py         # Script to convert model to ONNX
    └── k8s/
        ├── namespace.yaml
        ├── deployment.yaml
        └── service.yaml
```

## Requirements

- Docker
- Kubernetes cluster (v1.21+)
- kubectl
- Python 3.9+ (for local development)

## Getting Started

### Clone the Repository

```bash
git clone https://github.com/yourusername/llm-kubernetes.git
cd llm-kubernetes
```

### Option 1: PyTorch Deployment

Build and deploy the standard PyTorch-based API:

```bash
# Build the Docker image
docker build -t llm-api ./fast-api

# Deploy to Kubernetes
kubectl apply -f fast-api/k8s/namespace.yaml
kubectl apply -f fast-api/k8s/deployment.yaml
kubectl apply -f fast-api/k8s/service.yaml
```

### Option 2: ONNX Deployment

For optimized inference, use the ONNX version:

```bash
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
```

## API Usage

Both deployments expose a `/generate` endpoint for text generation.

### Endpoint

```
POST /generate
```

### Request Body

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `prompt` | string | required | The input text to continue |
| `max_tokens` | int | 50 (PyTorch) / 30 (ONNX) | Maximum tokens to generate |

### Example Request

```bash
curl -X POST http://localhost:80/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "The future of AI is", "max_tokens": 50}'
```

### Example Response

```json
{
  "generated_text": "The future of AI is bright and full of possibilities..."
}
```

## Resource Configuration

Default resource limits per pod:

| Resource | Request | Limit |
|----------|---------|-------|
| CPU | 500m | 1 |
| Memory | 1Gi | 2Gi |

Adjust these values in the deployment manifests based on your needs.

## Contributing

Contributions are welcome! Please open issues or pull requests for improvements.

## License

This project is licensed under the MIT License.
