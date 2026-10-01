"""Prepare the pinned upstream GPT-BERT code with the paper's local changes."""
import argparse
import ast
import shutil
import subprocess
from pathlib import Path

REVISION = 'e6bb176f4ae5eac3d4b660a480a94182146eac6d'

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--upstream', type=Path, required=True,
                    help='Local checkout of babylm-org/multilingual-training')
args = parser.parse_args()
revision = subprocess.check_output(
    ['git', '-C', str(args.upstream), 'rev-parse', 'HEAD'], text=True).strip()
if revision != REVISION:
    raise SystemExit(f'Expected upstream revision {REVISION}, got {revision}')
if subprocess.check_output(['git', '-C', str(args.upstream), 'diff', 'HEAD'], text=True):
    raise SystemExit('Use an unmodified upstream checkout.')

scripts = Path(__file__).resolve().parent
vendor = scripts.parent / 'vendor/gpt_bert'
if (vendor / 'train.py').exists():
    raise SystemExit(f'Already prepared: {vendor}')
source = args.upstream.resolve() / 'babybabellm-gptbert'
for name in ['dataset.py', 'lamb.py', 'model_extra.py', 'model_logging.py', 'utils.py']:
    shutil.copy2(source / 'pretraining' / name, vendor / name)
shutil.copy2(source / 'pretraining/train_single_gpu.py', vendor / 'train.py')
subprocess.run(['patch', '--batch', '-p1', '-i', str(scripts / 'gptbert-training.patch')],
               cwd=vendor, check=True)

# Use only local conversion functions; omit the upstream upload commands.
converter = (source / 'scripts/convert_and_push_mono_1_2.py').read_text()
names = {'write_remote_code_files', '_write_primary_config'}
functions = [ast.get_source_segment(converter, node) for node in ast.parse(converter).body
             if isinstance(node, ast.FunctionDef) and node.name in names]
assert len(functions) == 2
helper = ('from __future__ import annotations\nimport json\nimport textwrap\n'
          'from pathlib import Path\nfrom typing import Any, Dict\nimport torch\n'
          'from gptbert_save import _save_single_variant\n\n'
          + '\n\n'.join(functions) + '\n')
helper = helper.replace('PROJECT_ROOT / "pretraining" / "model_extra.py"',
                        'Path(__file__).with_name("model_extra.py")')
# Transformers 5 requires the tied-weight mapping before loading weights.
helper = helper.replace("base_model_prefix = 'gpt_bert'",
                        "base_model_prefix = 'gpt_bert'\n            all_tied_weights_keys = {}")
shutil.copy2(scripts / 'gptbert_save.py', vendor / 'gptbert_save.py')
(vendor / 'convert_helpers.py').write_text(helper)
print(f'Prepared {vendor} from {REVISION}')
