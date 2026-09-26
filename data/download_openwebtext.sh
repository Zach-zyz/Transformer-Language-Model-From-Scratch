#!/usr/bin/env bash
set -euo pipefail
python "$(dirname "$0")/download_data.py" \
  --manifest "$(dirname "$0")/openwebtext_source_manifest.json" \
  --output-root "$(dirname "$0")/openwebtext" \
  "$@"
