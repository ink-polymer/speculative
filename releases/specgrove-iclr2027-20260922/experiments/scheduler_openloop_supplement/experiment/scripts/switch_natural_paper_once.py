"""Stop only the verified old three-repeat queue, preserving committed groups."""
import argparse
import json
import os
from pathlib import Path
import signal
import time


def main():
    ap = argparse.ArgumentParser(__doc__)
    ap.add_argument('--model',required=True,choices=('qwen3_4b','qwen3_8b'))
    ap.add_argument('--parent',required=True,type=int)
    ap.add_argument('--worker',required=True,type=int)
    args=ap.parse_args()
    parent=Path(f'/proc/{args.parent}/cmdline').read_bytes().replace(b'\0',b' ').decode()
    worker=Path(f'/proc/{args.worker}/cmdline').read_bytes().replace(b'\0',b' ').decode()
    expected_parent=f'bash /root/dp-paper-natural-suite-20260917/scripts/run_natural_paper_suite.sh {args.model}'
    expected_output=f'--output /root/dp-paper-natural-results-20260917/{args.model}/main128'
    if parent.strip()!=expected_parent or 'scripts/run_natural_paper_phase.py' not in worker or f'--model {args.model}' not in worker or '--phase main128' not in worker or expected_output not in worker:
        raise RuntimeError('Exact queue/worker validation failed; nothing stopped')
    if f'PPid:\t{args.parent}\n' not in Path(f'/proc/{args.worker}/status').read_text():
        raise RuntimeError('Worker parent changed')
    # Stop parent first so it cannot advance to another GPU stage after this
    # worker exits. It has no checkpoint state; all group commits live in worker.
    os.kill(args.parent,signal.SIGKILL)
    try: os.kill(args.worker,signal.SIGTERM)
    except ProcessLookupError: pass
    for _ in range(100):
        if not Path(f'/proc/{args.worker}').exists(): break
        time.sleep(.1)
    else: raise RuntimeError('GPU worker still present; refuse concurrent launch')
    output=Path('/root/dp-paper-natural-results-20260917')/args.model/'main128'
    progress=json.loads((output/'progress.json').read_text())
    marker={'reason':'User approved one timing repeat per seed/method/group',
            'old_results_preserved':True,'old_progress':progress,
            'new_results_root':'/root/dp-paper-natural-once-results-20260917',
            'original_parent_pid':args.parent,'original_worker_pid':args.worker}
    (output/'switched_to_single_repeat.json').write_text(json.dumps(marker,indent=2)+'\n')
    print(json.dumps({'stopped_three_repeat_queue':args.model,'preserved_groups':progress['completed_groups']}))


if __name__=='__main__':main()
