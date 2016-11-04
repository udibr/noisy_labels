#!/usr/bin/env bash
python mnist-jacob-reed.py --seed 42 "$@" &
python mnist-jacob-reed.py --seed 43 "$@" &
python mnist-jacob-reed.py --seed 44 "$@" &
python mnist-jacob-reed.py --seed 45 "$@" &
python mnist-jacob-reed.py --seed 46 "$@"
