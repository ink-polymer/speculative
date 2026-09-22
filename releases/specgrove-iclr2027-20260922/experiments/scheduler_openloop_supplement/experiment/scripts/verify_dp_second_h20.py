"""Verify second-host frozen source, model bytes and runtime; contains no credentials."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import socket
import subprocess
import torch

ROOT=Path(__file__).resolve().parents[1]


def digest(path):
    value=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):value.update(block)
    return value.hexdigest()


def fingerprint():
    sources=sorted((ROOT/'src').rglob('*.py'))+sorted((ROOT/'scripts').glob('*.py'))
    cfg=json.loads((ROOT/'configs/adaptive_block_qwen3_8b.json').read_text())['model']
    files={str(p.relative_to(ROOT)):digest(p) for p in sources}
    model_files={}
    for name,revision in [(cfg['target'],cfg['target_revision']),(cfg['draft'],cfg['draft_revision'])]:
        base=Path('/root/autodl-tmp/hf-cache/hub')/('models--'+name.replace('/','--'))/'snapshots'/revision
        if not base.is_dir():raise RuntimeError('Missing pinned model snapshot: '+str(base))
        for path in sorted(base.rglob('*')):
            if path.is_file():model_files[name+'/'+str(path.relative_to(base))]=digest(path)
    packages={}
    for distribution in importlib.metadata.distributions():
        name=distribution.metadata.get('Name')
        if name:packages[name.lower().replace('_','-')]=distribution.version
    gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,driver_version,memory.total','--format=csv,noheader'],text=True).strip()
    return {'sources':files,'model_files':model_files,'model_config':cfg,'packages':packages,
            'python':platform.python_version(),'torch':torch.__version__,'torch_cuda':torch.version.cuda,'gpu':gpu}


def main():
    parser=argparse.ArgumentParser();group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--write-reference',type=Path);group.add_argument('--reference',type=Path)
    parser.add_argument('--report',type=Path);args=parser.parse_args();actual=fingerprint()
    if args.write_reference:
        args.write_reference.parent.mkdir(parents=True,exist_ok=True)
        args.write_reference.write_text(json.dumps(actual,indent=2)+'\n');status='reference_written'
    else:
        expected=json.loads(args.reference.read_text())
        differences=[key for key in actual if actual[key]!=expected.get(key)]
        if differences:raise RuntimeError('Second-host contract mismatch: '+', '.join(differences))
        status='verified'
    report={'status':status,'hostname':socket.gethostname(),'fingerprint':actual,
            'scope':'full source and cached pinned-model SHA256; installed package versions; Python/Torch/CUDA/GPU identity. Not an AR-law certificate.'}
    if args.report:
        args.report.parent.mkdir(parents=True,exist_ok=True)
        args.report.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'status':status,'hostname':report['hostname'],'model_files_hashed':len(actual['model_files']),
                      'source_files_hashed':len(actual['sources']),'gpu':actual['gpu']}),flush=True)


if __name__=='__main__':main()
