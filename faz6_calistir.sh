#!/bin/bash
# Starts all Faz 6 experiments at once. Every part has checkpoints; rerunning resumes where it left off.
#   E  : one-class autoencoder (TensorFlow, GPU)              — in its own process
#   C→G: Faysal replication, then CIC-IoT2023 audit              — 4 cores
#   D  : protocols, one process per seed (42, 7, 123)           — 4 cores per process
# A and B were completed earlier (tables are under faz6_ciktilar/tables/).
cd "$(dirname "$0")"
L=faz6_ciktilar/logs; mkdir -p "$L"
echo "başladı: $(date)"

python3 faz6_E_tek_sinif.py > "$L/E.log" 2>&1 &
( FAZ6_NJOBS=4 python3 faz6_dogrulama.py C > "$L/C.log" 2>&1
  FAZ6_NJOBS=4 python3 faz6_dogrulama.py G > "$L/G.log" 2>&1 ) &
for s in 42 7 123; do
  FAZ6_NJOBS=4 OMP_NUM_THREADS=4 python3 faz6_dogrulama.py D "$s" > "$L/D_$s.log" 2>&1 &
done
wait

python3 faz6_dogrulama.py D ozet > "$L/D_ozet.log" 2>&1
python3 faz6_ozet.py > "$L/ozet.log" 2>&1
echo "TAMAM: $(date)"
