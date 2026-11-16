#!/bin/sh
# Full pipeline after data download. Steps write logs/<step>.log; logs/DONE_<step> marks completion.
cd "$(dirname "$0")/.."
PY="${PY:-python}"
mark() { touch "logs/DONE_$1"; }
$PY scripts/01_build_composites.py cities > logs/cities.log 2>&1 && mark composites_cities &
CITIES=$!
$PY scripts/01_build_composites.py padma > logs/padma.log 2>&1 && mark composites_padma &
PADMA=$!
$PY scripts/05_wealth.py > logs/wealth.log 2>&1 && mark wealth
wait $CITIES
$PY scripts/02_builtup.py cities > logs/train_cities.log 2>&1 && mark train_cities
$PY scripts/02_builtup.py calibrate-gbm > logs/calibrate.log 2>&1 && mark calibrate
$PY scripts/03_urban_expansion.py > logs/urban.log 2>&1 && mark urban
wait $PADMA
$PY scripts/02_builtup.py padma > logs/train_padma.log 2>&1 && mark train_padma
$PY scripts/06_padma_causal.py > logs/causal.log 2>&1 && mark causal
mark all
