#!/bin/bash
# Launch one WP7 GPU worker on the first local GPU that nvidia-smi shows idle (no compute process, <5% util)
# in two checks 30 s apart. CUDA index in PCI order: 0 -> 01:00.0, 1 -> 03:00.0, 2 -> 04:00.0.
cd /home/morg/NLP_2526b/oripetel/nlp_proj/nlp_proj
LOG=results/paper_analysis/circularity/logs
idle() {  # $1 = bus id
  local apps util
  apps=$(nvidia-smi --query-compute-apps=gpu_bus_id --format=csv,noheader 2>/dev/null | grep -c "$1")
  util=$(nvidia-smi --query-gpu=pci.bus_id,utilization.gpu --format=csv,noheader,nounits 2>/dev/null | grep "$1" | awk -F', ' '{print $2}')
  [ "$apps" = "0" ] && [ -n "$util" ] && [ "$util" -lt 5 ]
}
BUS=(00000000:01:00.0 00000000:03:00.0 00000000:04:00.0)
while true; do
  for i in 0 1 2; do
    if idle ${BUS[$i]}; then
      sleep 30
      if idle ${BUS[$i]}; then
        echo "$(date) GPU bus ${BUS[$i]} idle -> CUDA $i" >> $LOG/gpu_watch.log
        export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$i
        PY=".venv/bin/python -m experiments.paper_analysis.circularity.fit --threads 3 --device cuda"
        $PY --plan main --method fixed >> $LOG/gpu_main_fixed.log 2>&1
        $PY --plan etagamma --method fixed >> $LOG/gpu_etagamma.log 2>&1
        $PY --plan main --method rn >> $LOG/gpu_main_rn.log 2>&1
        echo "$(date) GPU worker done" >> $LOG/gpu_watch.log
        exit 0
      fi
    fi
  done
  sleep 60
done
