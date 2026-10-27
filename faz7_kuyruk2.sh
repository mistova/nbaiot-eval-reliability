#!/bin/bash
# Phase 7 second queue: when IoT-23 (N) finishes, the memory-intensive R measurements (MedBIoT, Engelen CICIDS2017, Kitsune) run in sequence.
cd "$(dirname "$0")"
L=faz7_ciktilar/logs
while pgrep -f faz7_N_iot23.py > /dev/null; do sleep 30; done
OMP_NUM_THREADS=4 python3 faz7_R_bilesenler.py cicids2017_engelen medbiot kitsune >> $L/R_b.log 2>&1
echo "KUYRUK2 TAMAM $(date)" >> $L/kuyruk.log
