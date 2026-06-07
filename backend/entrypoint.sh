#!/bin/sh
set -eu

if [ "$(id -u)" = "0" ]; then
  mkdir -p /app/audio /app/output /home/app/.cache/huggingface
  chown -R app:app /app/audio /app/output /home/app/.cache/huggingface
  exec su app -s /bin/sh -c 'exec "$@"' -- sh "$@"
fi

exec "$@"
