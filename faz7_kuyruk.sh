#!/bin/bash
# Phase 7 job queue: starts the remaining seeds as the first wave (manually started processes) finishes.
# Every job is checkpointed; at the end all jobs are called once more (a job stopped by the watchdog resumes where it left off).
cd "$(dirname "$0")"
L=faz7_ciktilar/logs
bekle() { while kill -0 "$1" 2>/dev/null; do sleep 30; done; }
CPU="FAZ7_NJOBS=3 OMP_NUM_THREADS=3"
GPU="OMP_NUM_THREADS=2"

( bekle 2182742; env $CPU python3 faz7_Q_protokoller.py 2024 >> $L/Q_2024.log 2>&1 ) &
( bekle 2182743; env $CPU python3 faz7_Q_protokoller.py 13   >> $L/Q_13.log 2>&1 ) &
( bekle 2182744; env $CPU python3 faz7_H_arumugam.py 4 5 6 >> $L/H_b.log 2>&1 ) &
( bekle 2182745; env $CPU python3 faz7_H_arumugam.py 7 8 9 >> $L/H_c.log 2>&1 ) &
( bekle 2184858; env $GPU python3 faz7_J_gpu_protokoller.py 123 2024 >> $L/J_b.log 2>&1 ) &
( bekle 2187269; env $GPU python3 faz7_J_gpu_protokoller.py 13 >> $L/J_c.log 2>&1 ) &
wait

# second pass: finish everything left incomplete
for s in 42 7 123 2024 13; do env $CPU python3 faz7_Q_protokoller.py $s >> $L/Q_tur2.log 2>&1; done &
env $CPU python3 faz7_H_arumugam.py 42 0 1 2 3 4 5 6 7 8 9 >> $L/H_tur2.log 2>&1 &
( env $GPU python3 faz7_J_gpu_protokoller.py 42 7 123 2024 13 >> $L/J_tur2.log 2>&1
  env $GPU python3 faz7_K_kisayol.py >> $L/K_tur2.log 2>&1
  env $GPU python3 faz7_P_meidan.py >> $L/P_tur2.log 2>&1
  env $GPU python3 faz7_I_ezber.py >> $L/I_tur2.log 2>&1 ) &
wait
echo "KUYRUK TAMAM $(date)" >> $L/kuyruk.log
