#!/bin/sh
# Build paper/main.pdf with Tectonic (a self-contained LaTeX engine; fetches packages on first use).
# The binary and its package cache live on the share, not in the 4 GB home directory.
#   sh paper/build.sh
# Overleaf works too: upload the paper/ folder (ACL template files acl.sty and acl_natbib.bst are included).
set -e
cd "$(dirname "$0")"
TECTONIC_CACHE_DIR=/home/morg/NLP_2526b/oripetel/.cache/Tectonic \
  /home/morg/NLP_2526b/oripetel/tools/tectonic -X compile --keep-logs main.tex
grep -E "Overfull|undefined" main.log || true
echo "-> $(pwd)/main.pdf"
