#!/usr/bin/env bash
#
# One-time GCP setup for GitHub Actions Workload Identity Federation.
# Run locally with: bash scripts/setup-gcp-wif.sh
# Requires: gcloud CLI authenticated as project owner.
#
# After running, add the two printed values as GitHub repository secrets:
#   WORKLOAD_IDENTITY_PROVIDER
#   GCP_SERVICE_ACCOUNT
#
set -euo pipefail

PROJECT_ID="knowledge-graph-app-kg"
PROJECT_NUMBER="314185672280"
REGION="australia-southeast1"
REPO_NAME="kg-web"
POOL_NAME="github-actions-pool"
PROVIDER_NAME="github-oidc"
SA_NAME="github-actions-deploy"
GITHUB_REPO="RaggedR/math-research-tools"

echo "==> Setting project to ${PROJECT_ID}"
gcloud config set project "${PROJECT_ID}"

echo "==> Enabling required APIs"
gcloud services enable \
  iam.googleapis.com \
  iamcredentials.googleapis.com \
  run.googleapis.com \
  artifactregistry.googleapis.com

echo "==> Creating Artifact Registry repository"
gcloud artifacts repositories create "${REPO_NAME}" \
  --repository-format=docker \
  --location="${REGION}" \
  --description="Knowledge Graph web app Docker images" \
  2>/dev/null || echo "    (repository already exists)"

echo "==> Creating Workload Identity Pool"
gcloud iam workload-identity-pools create "${POOL_NAME}" \
  --location="global" \
  --display-name="GitHub Actions Pool" \
  2>/dev/null || echo "    (pool already exists)"

echo "==> Creating OIDC Provider"
gcloud iam workload-identity-pools providers create-oidc "${PROVIDER_NAME}" \
  --location="global" \
  --workload-identity-pool="${POOL_NAME}" \
  --display-name="GitHub OIDC" \
  --attribute-mapping="google.subject=assertion.sub,attribute.repository=assertion.repository" \
  --attribute-condition="assertion.repository == '${GITHUB_REPO}'" \
  --issuer-uri="https://token.actions.githubusercontent.com" \
  2>/dev/null || echo "    (provider already exists)"

echo "==> Creating Service Account"
gcloud iam service-accounts create "${SA_NAME}" \
  --display-name="GitHub Actions Deploy" \
  2>/dev/null || echo "    (service account already exists)"

SA_EMAIL="${SA_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"

echo "==> Granting roles to service account"
for ROLE in roles/run.developer roles/artifactregistry.writer roles/iam.serviceAccountUser; do
  gcloud projects add-iam-policy-binding "${PROJECT_ID}" \
    --member="serviceAccount:${SA_EMAIL}" \
    --role="${ROLE}" \
    --condition=None \
    --quiet
done

echo "==> Binding Workload Identity Pool to Service Account"
gcloud iam service-accounts add-iam-policy-binding "${SA_EMAIL}" \
  --role="roles/iam.workloadIdentityUser" \
  --member="principalSet://iam.googleapis.com/projects/${PROJECT_NUMBER}/locations/global/workloadIdentityPools/${POOL_NAME}/attribute.repository/${GITHUB_REPO}"

echo ""
echo "============================================"
echo "  Add these as GitHub repository secrets:"
echo "============================================"
echo ""
echo "WORKLOAD_IDENTITY_PROVIDER:"
echo "  projects/${PROJECT_NUMBER}/locations/global/workloadIdentityPools/${POOL_NAME}/providers/${PROVIDER_NAME}"
echo ""
echo "GCP_SERVICE_ACCOUNT:"
echo "  ${SA_EMAIL}"
echo ""
echo "Done!"
