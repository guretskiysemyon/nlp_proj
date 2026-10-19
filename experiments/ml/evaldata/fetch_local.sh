# Sourced by the evaldata sbatch scripts: put the models into a node-local HF cache
# (NFS model loads are slow and big NFS downloads failed with EIO), then go offline.
export HF_HOME=/tmp/oripetel_ml_eval_hf
mkdir -p $HF_HOME
HF_HUB_OFFLINE=0 python -c "
import sys
from huggingface_hub import snapshot_download as s
pats = {'facebook/nllb-200-3.3B': ['*.json', '*.bin', '*.model'],
        'google/madlad400-3b-mt': ['*.json', 'model.safetensors', '*.model']}
for r in sys.argv[1:]:
    print(s(r, allow_patterns=pats.get(r)), flush=True)
" "$@"
export HF_HUB_OFFLINE=1
