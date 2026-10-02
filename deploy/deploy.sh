#!/usr/bin/env bash
# Runs ON the VPS. .github/workflows/ci.yml pipes it over SSH after copying
# deploy/docker-compose.yml into place. Pulls one exact image from GHCR,
# switches the app to it, and puts the previous image back if the new one
# does not come up healthy.
#
# Environment:
#   VPS_PATH    directory holding docker-compose.yml and .env
#   IMAGE       image repository, e.g. ghcr.io/owner/name
#   IMAGE_TAG   tag to deploy (the commit SHA)
#   GHCR_USER, GHCR_TOKEN   optional registry login, needed while the package is private
set -euo pipefail

: "${VPS_PATH:?}" "${IMAGE:?}" "${IMAGE_TAG:?}"
export IMAGE IMAGE_TAG

cd "$VPS_PATH"
if [ ! -f .env ]; then
  echo "::error::$VPS_PATH/.env does not exist on the VPS. Create it from .env.example (it holds the Groq API keys) and re-run."
  exit 1
fi

# What is running now, so it can be restored (and kept when cleaning up).
previous=""
previous_id=""
running=$(docker compose ps -q app 2>/dev/null || true)
if [ -n "$running" ]; then
  previous=$(docker inspect --format '{{.Config.Image}}' "$running")
  previous_id=$(docker inspect --format '{{.Image}}' "$running")
fi

if ! docker image inspect "$IMAGE:$IMAGE_TAG" >/dev/null 2>&1; then
  # Log in through a throwaway config directory, deleted when this script
  # ends: the token CI passes must not be left on this machine, and must not
  # replace a GHCR login that other projects here rely on.
  registry_config=$(mktemp -d)
  trap 'rm -rf "$registry_config"' EXIT
  if [ -n "${GHCR_TOKEN:-}" ]; then
    if ! printf '%s' "$GHCR_TOKEN" | docker --config "$registry_config" login ghcr.io -u "${GHCR_USER:?}" --password-stdin >/dev/null 2>&1; then
      echo "::error::Could not log in to ghcr.io from the VPS."
      exit 1
    fi
  fi
  if ! docker --config "$registry_config" pull --quiet "$IMAGE:$IMAGE_TAG"; then
    echo "::error::Could not pull $IMAGE:$IMAGE_TAG. The running release was left as it is."
    exit 1
  fi
fi

wait_until_healthy() {
  local cid status state
  cid=$(docker compose ps -q app)
  for _ in $(seq 1 60); do
    status=$(docker inspect --format '{{.State.Health.Status}}' "$cid" 2>/dev/null || echo gone)
    state=$(docker inspect --format '{{.State.Status}}' "$cid" 2>/dev/null || echo gone)
    [ "$status" = healthy ] && return 0
    # Crashed on start (and being restarted), or failed its health check.
    if [ "$status" = unhealthy ] || [ "$state" != running ]; then
      return 1
    fi
    sleep 2
  done
  return 1
}

docker compose up -d --remove-orphans

if wait_until_healthy; then
  echo "Deployed $IMAGE:$IMAGE_TAG and healthy."
  docker compose ps app
  # Point the local `prod` tag at this release, so a plain `docker compose
  # up -d` by hand (which asks for :prod) starts it without needing to reach
  # the registry. Done only now, so until a release is healthy :prod still
  # means the one before it.
  docker tag "$IMAGE:$IMAGE_TAG" "$IMAGE:prod"
  # Keep this release and the one just replaced (for a quick rollback),
  # under whatever tags they have; drop older ones.
  current_id=$(docker image inspect --format '{{.Id}}' "$IMAGE:$IMAGE_TAG")
  docker image ls "$IMAGE" --no-trunc --format '{{.Repository}}:{{.Tag}} {{.ID}}' | while read -r ref id; do
    if [ "$id" != "$current_id" ] && [ "$id" != "$previous_id" ]; then
      docker image rm "$ref" >/dev/null 2>&1 || true
    fi
  done
  exit 0
fi

echo "::error::$IMAGE:$IMAGE_TAG did not become healthy."
docker compose logs --tail 60 app || true
if [ -n "$previous" ] && [ "$previous" != "$IMAGE:$IMAGE_TAG" ]; then
  echo "Rolling back to $previous"
  IMAGE="${previous%:*}" IMAGE_TAG="${previous##*:}" docker compose up -d --remove-orphans
  if wait_until_healthy; then
    echo "Rolled back: $previous is serving again."
  else
    echo "::error::The rollback to $previous is not healthy either. The site is down."
  fi
fi
exit 1
