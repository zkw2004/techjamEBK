#!/usr/bin/env bash
# Fetch and verify KuaiRand-Pure. Idempotent — safe to re-run.
#
# The dataset is git-ignored on purpose: 194MB extracted, largest file 83MB,
# and a blob that size is permanent in git history. It is one public file from
# a stable Zenodo DOI, so fetching beats committing.
set -euo pipefail

# Public checksum of the organiser archive, not a credential — the high
# entropy trips detect-secrets, so it is allowlisted inline.
# The checksum is a public integrity value, not a credential; detect-secrets
# flags any high-entropy hex string.
ARCHIVE_SHA256="c814bf6f3624c0cfae83c57de3df26b2ed206e5c57bab4c4dcbfabbabe20cbf0"  # pragma: allowlist secret  # pragma: allowlist secret
URL="https://zenodo.org/records/10439422/files/KuaiRand-Pure.tar.gz"

cd "$(dirname "$0")/.."
mkdir -p data

if [ -d data/KuaiRand-Pure/data ]; then
  echo "✓ data/KuaiRand-Pure/data already present — nothing to do"
  exit 0
fi

if [ ! -f data/KuaiRand-Pure.tar.gz ]; then
  echo "→ downloading KuaiRand-Pure (45MB)"
  curl -L --fail --progress-bar -o data/KuaiRand-Pure.tar.gz "$URL"
fi

# The archive hash is recorded in DAY0.md and goes into the run manifest
# (Section 8.3). A silent dataset change must fail loudly, not drift.
echo "→ verifying checksum"
actual=$(shasum -a 256 data/KuaiRand-Pure.tar.gz | awk '{print $1}')
if [ "$actual" != "$ARCHIVE_SHA256" ]; then
  echo "✗ checksum mismatch"
  echo "  expected $ARCHIVE_SHA256"
  echo "  got      $actual"
  echo "  Delete data/KuaiRand-Pure.tar.gz and re-run. If it persists, stop"
  echo "  and tell the team — every recorded baseline assumes this archive."
  exit 1
fi

echo "→ extracting"
tar xzf data/KuaiRand-Pure.tar.gz -C data

echo "✓ done: data/KuaiRand-Pure/data"
echo
echo "Verify the baselines reproduce (expect valid primary 0.6015, ~25s):"
echo "  cd kuairand-starter-kit"
echo "  python3 baseline.py --model fm --data_dir ../data/KuaiRand-Pure/data"
