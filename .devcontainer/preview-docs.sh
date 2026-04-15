#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

./docs/build/pandocs.sh
exec python3 -m http.server 8000 --directory docs/site
