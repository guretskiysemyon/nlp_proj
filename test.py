from huggingface_hub import hf_hub_download
import os
print('token set:', bool(os.environ.get('HF_TOKEN')))
p = hf_hub_download('google/gemma-2-2b-it', 'model.safetensors.index.json', token=os.environ.get('HF_TOKEN'))
print('OK:', p)
