#!/usr/bin/env bash
# Download an additional Ollama model using temporary outbound Docker access.
# Usage: bash scripts/pull-ollama-model.sh qwen3:8b
set -Eeuo pipefail

model="${1:-qwen3:8b}"
if [[ -z "$model" ]]; then
  printf 'Model name must not be empty.\n' >&2
  exit 2
fi

# Resolve the network belonging to THIS Compose project, not another stack.
project="$(docker compose config --format json | python3 -c 'import json,sys; print(json.load(sys.stdin)["name"])')"
network="${project}_ollama-egress"
container="$(docker compose ps -q ollama)"
if [[ -z "$container" ]]; then
  printf 'Ollama is not running. Start the stack first.\n' >&2
  exit 1
fi

# Refuse network changes if the expected network does not exist.
docker network inspect "$network" >/dev/null
was_connected="$(docker inspect --format '{{range $name, $network := .NetworkSettings.Networks}}{{println $name}}{{end}}' "$container" | grep -Fx "$network" || true)"
connected_here=0
cleanup() {
  if [[ "$connected_here" == 1 ]]; then
    if ! docker network disconnect "$network" "$container"; then
      printf 'WARNING: Disconnect %s from %s manually.\n' "$container" "$network" >&2
    fi
  fi
}
trap cleanup EXIT

# The init image can change shared volume ownership; repair it explicitly
# without deleting the existing model volume.
docker compose run --rm --no-deps --entrypoint /bin/sh ollama-init \
  -c 'chown -R 10001:10001 /opt/ollama-models'

if [[ -z "$was_connected" ]]; then
  docker network connect "$network" "$container"
  connected_here=1
fi

docker compose exec -T ollama ollama pull "$model"
printf 'Model %s downloaded successfully.\n' "$model"
