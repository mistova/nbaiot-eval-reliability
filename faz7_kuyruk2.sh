#!/bin/bash
# Faz 7 ikinci kuyruk: IoT-23 (N) bitince bellek yoğun R ölçümleri (MedBIoT, Engelen CICIDS2017, Kitsune) sırayla.
cd "$(dirname "$0")"
L=faz7_ciktilar/logs
while pgrep -f faz7_N_iot23.py > /dev/null; do sleep 30; done
OMP_NUM_THREADS=4 python3 faz7_R_bilesenler.py cicids2017_engelen medbiot kitsune >> $L/R_b.log 2>&1
echo "KUYRUK2 TAMAM $(date)" >> $L/kuyruk.log
