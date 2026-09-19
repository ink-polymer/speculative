"""Validated, exact-PID pilot shutdown; preserves every committed result group."""
import argparse
import json
import os
from pathlib import Path
import signal
import time


def main():
    ap = argparse.ArgumentParser(__doc__)
    ap.add_argument('--model', required=True, choices=('qwen3_4b','qwen3_8b'))
    ap.add_argument('--parent', required=True, type=int)
    ap.add_argument('--worker', required=True, type=int)
    args = ap.parse_args()
    parent = Path(f'/proc/{args.parent}/cmdline').read_bytes().replace(b'\0', b' ').decode()
    worker = Path(f'/proc/{args.worker}/cmdline').read_bytes().replace(b'\0', b' ').decode()
    if f'run_main_natural.sh {args.model}' not in parent or f'--model {args.model}' not in worker or 'scripts/benchmark_main_natural.py' not in worker or '--requests 32' not in worker or '/root/dp-main-natural-cache-20260917/' not in worker:
        raise RuntimeError('Exact pilot target validation failed')
    status = Path(f'/proc/{args.worker}/status').read_text()
    if f'PPid:\t{args.parent}\n' not in status: raise RuntimeError('Worker parent changed')
    # Parent has an old EXIT callback that would restart a superseded GPU phase.
    # Prevent that callback before stopping ONLY the validated 32-task worker.
    os.kill(args.parent, signal.SIGKILL)
    try: os.kill(args.worker, signal.SIGTERM)
    except ProcessLookupError: pass
    for _ in range(100):
        if not Path(f'/proc/{args.worker}').exists(): break
        time.sleep(.1)
    else: raise RuntimeError('Pilot did not stop; refuse to start another GPU job')
    output = Path('/root/dp-main-natural-cache-20260917')/args.model/'superseded_by_natural128.json'
    output.write_text(json.dumps({'reason':'User requested128 natural tasks for formal paper suite', 'preserved_committed_groups':True,
        'interrupted_uncommitted_group_not_reused':True, 'old_quality_callback_suppressed':True, 'parent_pid':args.parent,'worker_pid':args.worker}, indent=2)+'\n')
    print(json.dumps({'pilot_stopped':args.model,'committed_results_preserved':True}))


if __name__ == '__main__': main()
