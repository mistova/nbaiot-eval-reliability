#!/bin/bash
# Faz 6 deneylerinin tamamını tek seferde başlatır. Her parça checkpoint'li; yeniden çalıştırılınca kaldığı yerden devam eder.
#   E  : tek sınıflı otokodlayıcı (TensorFlow, GPU)           — kendi sürecinde
#   C→G: Faysal tekrarı, ardından CIC-IoT2023 denetimi           — 4 çekirdek
#   D  : protokoller, tohum başına bir süreç (42, 7, 123)       — süreç başına 4 çekirdek
# A ve B daha önce tamamlandı (tablolar faz6_ciktilar/tables/ altında).
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
