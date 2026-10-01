"""Write one local safetensors checkpoint for the causal evaluation wrapper."""
import torch
from safetensors.torch import save_file as save_safetensors

def _save_single_variant(ckpt_path, base_config_dict, out_dir):
    state = torch.load(ckpt_path, map_location='cpu')
    if 'state_dict' in state:
        state = state['state_dict']
    config = dict(base_config_dict)
    # Clone tied tensors because safetensors cannot serialize shared storage.
    weights = {}
    for name, tensor in state.items():
        name = name.removeprefix('module.')
        if not name.startswith('model.'):
            name = 'model.' + name
        weights[name] = tensor.detach().clone().contiguous()
    config['vocab_size'] = weights['model.embedding.word_embedding.weight'].shape[0]
    save_safetensors(weights, str(out_dir / 'model.safetensors'))
    return config
