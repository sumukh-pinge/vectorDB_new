#!/usr/bin/env bash
set -euo pipefail

GIT_REPO="${GIT_REPO:-}"
GIT_REF="${GIT_REF:-main}"
SRC_DIR="${SRC_DIR:-/workspace/src/nrp}"

if [[ -z "${GIT_REPO}" ]]; then
  echo "ERROR: GIT_REPO not set (e.g., https://github.com/sumukh-pinge/nrp_bierNQ_mini.git)"
  exit 1
fi

mkdir -p "$(dirname "${SRC_DIR}")"
if [[ -d "${SRC_DIR}/.git" ]]; then
  echo "🔄 Pulling latest ${GIT_REF} in ${SRC_DIR}..."
  git -C "${SRC_DIR}" fetch --all --prune
  git -C "${SRC_DIR}" checkout "${GIT_REF}"
  git -C "${SRC_DIR}" pull --ff-only
else
  echo "📥 Cloning ${GIT_REPO} -> ${SRC_DIR}"
  git clone --branch "${GIT_REF}" --depth 1 "${GIT_REPO}" "${SRC_DIR}"
fi

cd "${SRC_DIR}/app"
echo "✅ Code ready at $(pwd)"
echo "🚀 Executing: $*"
exec "$@"
