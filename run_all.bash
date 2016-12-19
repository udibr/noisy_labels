#!/usr/bin/env bash
python jacob-reed.py --seed 42 "$@" &
python jacob-reed.py --seed 43 "$@" &
python jacob-reed.py --seed 44 "$@" &
python jacob-reed.py --seed 45 "$@" &
python jacob-reed.py --seed 46 "$@"
