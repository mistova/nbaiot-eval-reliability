#!/bin/bash
# Download of Kitsune (UCI) and MedBIoT (TalTech, fine-grained).
# UCI does not support Range requests, so Kitsune is downloaded from scratch on every attempt and the zip integrity is tested.
# MedBIoT files resume where they left off; completed files are skipped.
cd ~/work/IoT_Botnet/veri
mkdir -p kitsune

for t in 1 2 3 4 5 6; do
  rm -f kitsune/kitsune.zip
  curl -sfL --retry 3 --speed-limit 20000 --speed-time 90 -o kitsune/kitsune.zip \
    "https://archive.ics.uci.edu/static/public/516/kitsune+network+attack+dataset.zip" \
    && unzip -tq kitsune/kitsune.zip > /dev/null && { echo "kitsune tamam $(date +%T)"; break; }
  echo "kitsune attempt $t failed $(date +%T)"; sleep 30
done

B=https://cs.taltech.ee/research/data/medbiot/fine-grained
for d in raw_dataset/malware structured_dataset/malware structured_dataset/normal; do
  mkdir -p medbiot/fine-grained/$d
  for f in $(curl -sL $B/$d/ | grep -oE 'href="[^"]+\.(csv|pcap)"' | sed 's/href="//;s/"//'); do
    hedef=medbiot/fine-grained/$d/$f
    boyut=$(curl -sI "$B/$d/$f" | grep -i content-length | tr -dc 0-9)
    [ -f "$hedef" ] && [ "$(stat -c %s "$hedef")" = "$boyut" ] && continue
    for i in $(seq 1 20); do
      curl -sfL -C - --retry 3 --speed-limit 20000 --speed-time 90 -o "$hedef" "$B/$d/$f" && break
      sleep 15
    done
    echo "ok $d/$f $(stat -c %s "$hedef")/$boyut $(date +%T)"
  done
done
echo BITTI $(date)
