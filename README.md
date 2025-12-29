# HuggingFace Model Cache Loader

A Kubernetes init container that downloads HuggingFace models and caches them in Civo Object Store (or any S3-compatible storage).

The problem this solves: you don't want every pod downloading a 700GB model from HuggingFace, 4 times for multiple replicas. Download once, cache in object storage, share a single RWX S3 backed version across all your replicas.

## What it does

1. Checks if the model already exists in your Civo bucket
2. If not, downloads it from HuggingFace (using `hf_transfer` for speed)
3. Uploads it to Civo Object Store for future use
4. If it's already there, exits immediately

The main container starts and the model is ready to go.

## Features

**Fast transfers**: Uses `hf_transfer` for downloads (up to 500MB/s on good connections) and concurrent uploads with 10 workers for 5-10x faster object storage uploads.

**Gated model support**: Works with private/gated models via HuggingFace tokens.

**Security hardened**: Input validation to prevent injection attacks, credential masking in logs, runs as non-root user.

**Built for Kubernetes**: Fail-fast behavior, proper exit codes, and designed to work as an init container.

**Tested**: 70%+ test coverage with pytest, uses moto for mocked S3 testing.

## Environment Variables

Required:
- `MODEL_NAME`: HuggingFace repo ID (e.g., `meta-llama/Llama-2-7b-hf`)
- `S3_BUCKET`: Your Civo bucket name
- `S3_ENDPOINT_URL`: Civo endpoint (e.g., `https://objectstore.lon1.civo.com`)
- `AWS_ACCESS_KEY_ID`: Civo access key
- `AWS_SECRET_ACCESS_KEY`: Civo secret key

Optional:
- `S3_PREFIX`: Organize models with a prefix (default: `models/`)
- `HF_TOKEN`: For gated models like Llama
- `DOWNLOAD_DIR`: Where to download temporarily (default: `/tmp`)
- `DISABLE_PROGRESS`: Set to `true` in CI to hide progress bars

All inputs are validated - no path traversal or injection attempts will work.

## Container Registry

Pre-built container images are available at GitHub Container Registry:

```bash
# Pull latest stable version
docker pull ghcr.io/jmesout/hf-model-downloader:latest

# Pull specific version (recommended for production)
docker pull ghcr.io/jmesout/hf-model-downloader:v1.0.0

# Pull by commit SHA (most secure - immutable)
docker pull ghcr.io/jmesout/hf-model-downloader@sha256:abc123...
```

**Available tags:**
- `latest` - Latest stable build from main branch
- `v1.0.0`, `v1.0`, `v1` - Semantic version tags
- `main`, `develop` - Branch-based tags
- `main-abc123` - SHA-tagged builds for traceability

**Multi-architecture support:** Images are available for both `linux/amd64` and `linux/arm64`.

## Image Verification

All container images are signed with Cosign for supply chain security:

```bash
# Install cosign
brew install cosign  # macOS
# or see https://docs.sigstore.dev/cosign/installation for other platforms

# Verify image signature
cosign verify \
  --certificate-identity=https://github.com/jmesout/hf-model-downloader/.github/workflows/ci.yml@refs/heads/main \
  --certificate-oidc-issuer=https://token.actions.githubusercontent.com \
  ghcr.io/jmesout/hf-model-downloader:v1.0.0
```

## Security Scanning

All images are automatically scanned with Trivy for vulnerabilities. View scan results in the [Security tab](https://github.com/jmesout/hf-model-downloader/security) of this repository.

Images with HIGH or CRITICAL vulnerabilities are not published until they're resolved.

## Quick Start

### 1. Create a Civo bucket

```bash
civo objectstore create my-model-cache --region LON1
```

Or use the [Civo Dashboard](https://dashboard.civo.com/object-stores). Grab your credentials from Object Stores > Credentials.

### 2. Create Kubernetes secrets

```bash
kubectl create secret generic civo-credentials \
  --from-literal=access-key-id=your-access-key \
  --from-literal=secret-access-key=your-secret-key

# Optional: for gated models like Llama
kubectl create secret generic hf-token \
  --from-literal=token=your-hf-token
```

Or with YAML:

```yaml
apiVersion: v1
kind: Secret
metadata:
  name: civo-credentials
type: Opaque
stringData:
  access-key-id: "your-access-key"
  secret-access-key: "your-secret-key"
---
apiVersion: v1
kind: Secret
metadata:
  name: hf-token
type: Opaque
stringData:
  token: "hf_xxxxxxxxxxxxx"
```

### 3. Add it to your pod

Here's a basic example:

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: model-inference
spec:
  initContainers:
  - name: cache-model
    image: ghcr.io/jmesout/hf-model-downloader:latest
    env:
    - name: MODEL_NAME
      value: "gpt2"
    - name: S3_BUCKET
      value: "my-model-cache"
    - name: S3_ENDPOINT_URL
      value: "https://objectstore.lon1.civo.com"
    - name: S3_PREFIX
      value: "models/"
    - name: AWS_ACCESS_KEY_ID
      valueFrom:
        secretKeyRef:
          name: civo-credentials
          key: access-key-id
    - name: AWS_SECRET_ACCESS_KEY
      valueFrom:
        secretKeyRef:
          name: civo-credentials
          key: secret-access-key
    - name: HF_TOKEN
      valueFrom:
        secretKeyRef:
          name: hf-token
          key: token
          optional: true
  containers:
  - name: inference
    image: your-inference-image:latest
    # Your main container that uses the model
```

## More Examples

For a gated model like Llama 2, add the HF_TOKEN:

```yaml
env:
  - name: MODEL_NAME
    value: "meta-llama/Llama-2-7b-hf"
  - name: HF_TOKEN
    valueFrom:
      secretKeyRef:
        name: hf-token
        key: token
  # ... rest of the config
```

To organize models by environment:

```yaml
env:
  - name: S3_PREFIX
    value: "production/models/"
  # ... rest of the config
```

## Civo Setup Notes

Endpoint URLs are region-specific:
- London: `https://objectstore.lon1.civo.com`
- Frankfurt: `https://objectstore.fra1.civo.com`
- New York: `https://objectstore.nyc1.civo.com`

Your credentials need ListBucket and PutObject permissions. Civo Object Store uses S3-style IAM policies - full bucket access is easiest.

## Building from Source

If you prefer to build your own images:

```bash
# Build with Docker BuildKit (recommended)
DOCKER_BUILDKIT=1 docker build -t hf-model-cache:custom \
  --build-arg BUILD_DATE=$(date -u +'%Y-%m-%dT%H:%M:%SZ') \
  --build-arg VERSION=custom \
  --build-arg VCS_REF=$(git rev-parse HEAD) \
  .

# Build for multiple platforms
docker buildx build --platform linux/amd64,linux/arm64 \
  -t hf-model-cache:custom .

# Run security scan before using
trivy image --severity HIGH,CRITICAL hf-model-cache:custom
```

## Local Testing

Build and run locally:

```bash
docker build -t hf-model-cache:test .

docker run --rm \
  -e MODEL_NAME=gpt2 \
  -e S3_BUCKET=my-model-cache \
  -e S3_ENDPOINT_URL=https://objectstore.lon1.civo.com \
  -e AWS_ACCESS_KEY_ID=your-access-key \
  -e AWS_SECRET_ACCESS_KEY=your-secret-key \
  hf-model-cache:test
```

Or run the Python script directly:

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements-dev.txt

export MODEL_NAME=gpt2
export S3_BUCKET=my-model-cache
export S3_ENDPOINT_URL=https://objectstore.lon1.civo.com
export AWS_ACCESS_KEY_ID=your-access-key
export AWS_SECRET_ACCESS_KEY=your-secret-key

python cache_model.py
```

## How It Works

The script validates environment variables, then checks if the model exists in Civo (using an efficient `list_objects_v2` call). If it's already there, exits immediately. Otherwise, downloads the model from HuggingFace to `/tmp`, uploads it to Civo with 10 concurrent workers, cleans up, and exits.

Exit code 0 means success. Exit code 1 means error (Kubernetes will retry based on your restart policy).

## Troubleshooting

**"Access Denied" errors**: Check your Civo credentials are correct, the bucket exists (`civo objectstore ls`), and your secret is properly created. Your credentials need ListBucket and PutObject permissions.

**"401 Unauthorized" from HuggingFace**: For gated models like Llama, you need to accept the license on HuggingFace first, then create a token at https://huggingface.co/settings/tokens and add it as the `HF_TOKEN` secret.

**"Disk full" errors**: Large models need more space. Increase ephemeral storage:
```yaml
resources:
  limits:
    ephemeral-storage: "50Gi"
```

**Slow downloads**: The Dockerfile already enables `hf_transfer` for speed. If it's still slow, check your network bandwidth to HuggingFace.

**View logs**:
```bash
kubectl logs <pod-name> -c cache-model
kubectl logs -f <pod-name> -c cache-model  # follow live
```

## Performance

Rough timings (assuming ~100MB/s bandwidth, faster with `hf_transfer`):
- Small models (500MB like GPT-2): 2-4 minutes
- Medium models (5GB like 7B params): 10-20 minutes
- Large models (20GB like 13B params): 40-80 minutes

The concurrent upload feature speeds things up significantly for models with many files - typically 5-10x faster than sequential uploads.

To speed things up: pre-populate your bucket with frequently-used models, use Civo regions close to your cluster, or use nodes with more bandwidth.

## Architecture

```
Kubernetes Pod
├── Init Container (this script)
│   ├── 1. Check Civo Object Store
│   ├── 2. If missing, download from HuggingFace
│   └── 3. Upload to Civo
└── Main Container (your inference app)
```

Your models end up in Civo at `s3://bucket/prefix/model-name/`.

## Testing

Run tests with pytest:

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
pytest tests/ --cov=cache_model --cov-report=html  # with coverage
```

Tests use `moto` to mock S3, so no real credentials needed. Tests run automatically on GitHub Actions for every push. All container images require 70%+ test coverage before being published.

## Security

**Container Security:**
- Multi-stage build with minimal attack surface (no build tools in production image)
- Runs as non-root user (UID 1000)
- All inputs validated to prevent path traversal and injection attacks
- Credentials masked in logs (showing only first/last 4 characters)
- Secure temp directories (0700 permissions) with automatic cleanup
- Regular vulnerability scanning with Trivy
- Image signing with Cosign for supply chain security

**Best Practices:**
- Always use Kubernetes secrets for credentials - never hardcode them
- Rotate your Civo credentials regularly
- Limit bucket permissions to only what's needed (ListBucket, PutObject)
- Use specific image versions or SHA digests in production (not `latest`)

## Contributing

PRs welcome! Just test with both public and gated models before submitting.

## License

MIT - see LICENSE file.

## Links

- [Civo Object Store docs](https://www.civo.com/docs/object-stores)
- [HuggingFace docs](https://huggingface.co/docs)
- [hf-transfer](https://github.com/huggingface/hf_transfer) (the fast download backend)
