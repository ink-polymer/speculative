#!/usr/bin/env bash
# Run on the old host after authenticating the new host's temporary SSH master.
# No passwords or private keys are recorded. Copy and launch survive local disconnects.
set -euo pipefail
study=/root/dp-paper-study-20260916
runtime=/root/autodl-tmp/envs/speculative/bin/python
copy_socket=/tmp/dp-new-h20-copy-20260916.sock
reference=/root/dp-paper-results-20260916/SECOND_H20_REFERENCE.json
export PYTHONPATH="$study/src:$study/scripts"
"$runtime" "$study/scripts/verify_dp_second_h20.py" --write-reference "$reference"
rsync -aR --info=progress2 -e "ssh -S $copy_socket -p 54917 -o BatchMode=yes" \
  /root/autodl-tmp/./envs/speculative \
  /root/autodl-tmp/./hf-cache/hub/models--Qwen--Qwen3-8B \
  /root/autodl-tmp/./hf-cache/hub/models--z-lab--Qwen3-8B-DFlash-b16 \
  /root/autodl-tmp/./data/sampling-t1-formal-16c0e91 \
  /root/autodl-tmp/./outputs/audited-global-tree-matrix-20260916/qwen3_8b/calibration.json \
  root@region-42.seetacloud.com:/root/autodl-tmp/
rsync -a -e "ssh -S $copy_socket -p 54917 -o BatchMode=yes" "$reference" \
  root@region-42.seetacloud.com:/root/dp-paper-results-20260916/
ssh -S "$copy_socket" -p 54917 -o BatchMode=yes root@region-42.seetacloud.com \
  'export PYTHONPATH=/root/dp-paper-study-20260916/src:/root/dp-paper-study-20260916/scripts
   /root/autodl-tmp/envs/speculative/bin/python /root/dp-paper-study-20260916/scripts/verify_dp_second_h20.py --reference /root/dp-paper-results-20260916/SECOND_H20_REFERENCE.json --report /root/dp-paper-results-20260916/HOST_PROVENANCE.json &&
   screen -dmS dp-paper-8b-20260916 bash -c "bash /root/dp-paper-study-20260916/scripts/run_dp_paper_shard.sh qwen3_8b > /root/dp-paper-results-20260916/queue-8b.log 2>&1" &&
   screen -ls'
printf 'Second-host deployment verified and persistent 8B shard launched.\n'
