#!/bin/sh
# Background composite builds: cities and the Padma region in parallel.
cd "$(dirname "$0")/.."
PY="${PY:-python}"
$PY scripts/01_build_composites.py cities > logs/cities.log 2>&1 &
$PY scripts/01_build_composites.py padma > logs/padma.log 2>&1 &
wait
