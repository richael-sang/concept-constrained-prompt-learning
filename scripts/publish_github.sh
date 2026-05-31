#!/bin/bash
# Create GitHub repo and push release_ccpl contents.
#
# Usage (run from repository root after git init + commit):
#   export GITHUB_USERNAME="your-github-username"
#   export GITHUB_TOKEN="ghp_xxxxxxxx"   # classic token with `repo` scope
#   bash scripts/publish_github.sh
#
# Optional:
#   export GITHUB_REPO="concept-constrained-prompt-learning"  # default
#   export GITHUB_VISIBILITY="public"                         # default

set -euo pipefail

REPO_NAME="${GITHUB_REPO:-concept-constrained-prompt-learning}"
VISIBILITY="${GITHUB_VISIBILITY:-public}"
USERNAME="${GITHUB_USERNAME:?Set GITHUB_USERNAME}"
TOKEN="${GITHUB_TOKEN:?Set GITHUB_TOKEN}"

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [ ! -d .git ]; then
  echo "Run from a git-initialized directory first."
  exit 1
fi

API="https://api.github.com/user/repos"
PAYLOAD=$(printf '{"name":"%s","description":"Concept-Constrained Prompt Learning for Few-Shot CLIP Adaptation","private":false,"auto_init":false}' "$REPO_NAME")

echo "Creating GitHub repository: ${USERNAME}/${REPO_NAME} (${VISIBILITY})"
HTTP_CODE=$(curl -s -o /tmp/gh_create_repo.json -w "%{http_code}" \
  -X POST "$API" \
  -H "Authorization: token ${TOKEN}" \
  -H "Accept: application/vnd.github+json" \
  -d "$PAYLOAD")

if [ "$HTTP_CODE" = "201" ]; then
  echo "Repository created."
elif [ "$HTTP_CODE" = "422" ]; then
  echo "Repository may already exist (HTTP 422). Continuing to push..."
else
  echo "Failed to create repository (HTTP ${HTTP_CODE}):"
  cat /tmp/gh_create_repo.json
  exit 1
fi

REMOTE_URL="https://${TOKEN}@github.com/${USERNAME}/${REPO_NAME}.git"

if git remote | grep -q '^origin$'; then
  git remote set-url origin "https://github.com/${USERNAME}/${REPO_NAME}.git"
else
  git remote add origin "https://github.com/${USERNAME}/${REPO_NAME}.git"
fi

git branch -M main
git push -u "$REMOTE_URL" main

echo ""
echo "Done: https://github.com/${USERNAME}/${REPO_NAME}"
