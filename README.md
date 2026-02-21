# Nail Segmentation API

Nail image segmentation API using YOLOv8 instance segmentation (ONNX runtime). Detects individual nails in photos and returns polygon coordinates and/or extracted nail images with transparency.

## Architecture

- **Runtime**: Google Cloud Run (serverless, pay-per-request)
- **CI/CD**: GitHub Actions (push to `main` auto-deploys)
- **Inference**: ONNX Runtime (CPU-optimized)
- **Image**: ~400-500MB (stripped torch/ultralytics)

```
Client -> Cloud Run -> app_fastapi.py -> segmentation_engine.py (OnnxSegmenter) -> ONNX model
```

## API Endpoints

| Endpoint | Method | Auth | Description |
|---|---|---|---|
| `/health` | GET | No | Health check / warm-up ping |
| `/segment` | POST | `x-api-key` | Returns bounding boxes + polygon coordinates |
| `/extract_nails_sync` | POST | `x-api-key` | Returns cropped nail PNGs (base64) with polygons |
| `/extract_nails` | POST | `x-api-key` | Async extraction with callback URL |

## Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run locally
uvicorn app_fastapi:app --host 0.0.0.0 --port 8000 --workers 4

# Run with Docker
docker build -t nail-seg . && docker run -p 8080:8080 -e API_KEY=your-key nail-seg

# Test health
curl http://localhost:8080/health

# Test segmentation
curl -X POST -H "x-api-key: your-key" -F "file=@test_img_3.jpeg" http://localhost:8080/segment
```

## Deployment

### Prerequisites

- GCP project with billing enabled
- GitHub repository: `iditsasson/nail-negmentation-api`
- `gcloud` CLI installed

### One-Time GCP Setup

1. **Enable APIs**:
   ```bash
   gcloud services enable run.googleapis.com artifactregistry.googleapis.com iamcredentials.googleapis.com
   ```

2. **Create Artifact Registry repository**:
   ```bash
   gcloud artifacts repositories create nail-seg \
     --repository-format=docker \
     --location=us-central1
   ```

3. **Create service account for GitHub Actions**:
   ```bash
   gcloud iam service-accounts create github-actions-deployer \
     --display-name="GitHub Actions Deployer"

   # Grant roles
   PROJECT_ID=$(gcloud config get-value project)
   SA_EMAIL=github-actions-deployer@${PROJECT_ID}.iam.gserviceaccount.com

   gcloud projects add-iam-policy-binding $PROJECT_ID \
     --member="serviceAccount:${SA_EMAIL}" \
     --role="roles/artifactregistry.writer"

   gcloud projects add-iam-policy-binding $PROJECT_ID \
     --member="serviceAccount:${SA_EMAIL}" \
     --role="roles/run.admin"

   gcloud projects add-iam-policy-binding $PROJECT_ID \
     --member="serviceAccount:${SA_EMAIL}" \
     --role="roles/iam.serviceAccountUser"
   ```

4. **Set up Workload Identity Federation**:
   ```bash
   # Create pool
   gcloud iam workload-identity-pools create github-pool \
     --location="global" \
     --display-name="GitHub Pool"

   # Create OIDC provider
   gcloud iam workload-identity-pools providers create-oidc github-provider \
     --location="global" \
     --workload-identity-pool="github-pool" \
     --display-name="GitHub Provider" \
     --attribute-mapping="google.subject=assertion.sub,attribute.repository=assertion.repository" \
     --issuer-uri="https://token.actions.githubusercontent.com"

   # Allow GitHub repo to impersonate SA
   POOL_ID=$(gcloud iam workload-identity-pools describe github-pool --location=global --format="value(name)")

   gcloud iam service-accounts add-iam-policy-binding $SA_EMAIL \
     --role="roles/iam.workloadIdentityUser" \
     --member="principalSet://iam.googleapis.com/${POOL_ID}/attribute.repository/iditsasson/nail-negmentation-api"
   ```

5. **Add GitHub Secrets** (Settings > Secrets and variables > Actions):
   - `GCP_PROJECT_ID` — your GCP project ID
   - `WIF_PROVIDER` — `projects/<PROJECT_NUMBER>/locations/global/workloadIdentityPools/github-pool/providers/github-provider`
   - `WIF_SERVICE_ACCOUNT` — `github-actions-deployer@<PROJECT_ID>.iam.gserviceaccount.com`
   - `API_KEY` — the API key for the service

### CI/CD Flow

Push to `main` triggers the GitHub Actions workflow which:
1. Builds the Docker image
2. Pushes to Artifact Registry (tagged with commit SHA + `latest`)
3. Deploys to Cloud Run with configured resources

### Manual Deploy

For emergencies, deploy directly:
```bash
PROJECT_ID=$(gcloud config get-value project)
IMAGE=us-central1-docker.pkg.dev/${PROJECT_ID}/nail-seg/nail-segmentation-api

docker build -t ${IMAGE}:latest .
docker push ${IMAGE}:latest

gcloud run deploy nail-segmentation-api \
  --image ${IMAGE}:latest \
  --region us-central1 \
  --memory 1Gi --cpu 1 \
  --min-instances 0 --max-instances 5 \
  --concurrency 4 --timeout 120s \
  --cpu-boost --no-cpu-throttling \
  --execution-environment gen2 \
  --allow-unauthenticated \
  --set-env-vars="API_KEY=your-key"
```

## Cloud Run Settings

| Setting | Value | Rationale |
|---|---|---|
| Memory | 1Gi | 46MB model + inference buffers need headroom |
| CPU | 1 | Sufficient for single-image ONNX inference |
| Min instances | 0 | Budget constraint (pay-per-request) |
| Max instances | 5 | Cost cap |
| Concurrency | 4 | CPU-bound inference, avoid overload |
| Timeout | 120s | Generous for async extract_nails |
| CPU boost | Enabled | Faster cold starts (free) |
| No CPU throttling | Enabled | Background tasks need CPU after response |
| Execution env | gen2 | Better cold start behavior |

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `API_KEY` | No | API key for `x-api-key` auth (has hardcoded default) |
| `PORT` | No | Server port (default: 8080, set by Cloud Run) |

## Mobile App Integration

The service uses `min-instances=0` to save costs, meaning cold starts (~5-8s) will occur. The mobile app should pre-warm the instance:

### 1. Update API Base URL

```
https://nail-segmentation-api-<HASH>-uc.a.run.app
```

### 2. Warm-up on App Foreground

```typescript
import { AppState } from 'react-native';

useEffect(() => {
  const subscription = AppState.addEventListener('change', (state) => {
    if (state === 'active') {
      fetch(`${API_BASE_URL}/health`).catch(() => {});  // fire-and-forget
    }
  });
  return () => subscription.remove();
}, []);
```

### 3. Warm-up on Pre-Segmentation Screen

```typescript
useFocusEffect(
  useCallback(() => {
    fetch(`${API_BASE_URL}/health`).catch(() => {});
  }, [])
);
```

### 4. No Changes Needed

The `/segment` and `/extract_nails` endpoints have the same API contract and auth header.

### 5. Optional Loading State

If `/health` returns `{"model_loaded": false}` or fails, show a "Preparing..." indicator before proceeding to segmentation.

### Why This Works

The user goes through several screens before reaching `/segment`. The foreground ping starts the cold boot (~5-8s). By the time they navigate to the camera, take a photo, and tap "Analyze", the instance is warm.
