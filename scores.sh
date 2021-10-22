#!/usr/bin/env sh


for f in ../outputs/baseline_4_8m/checkpoints/*.ckpt; do
    echo $f
    PYTHONUNBUFFERED=1 python3 predict.py --ckpt_path $f --data_directory ../contest/data/datasets/ --predicts_file ../contest/predictions.json --device cpu --num_threads 1
    PYTHONUNBUFFERED=1 python3 ../contest/evaluate_predictions.py --gt_file ../contest/data/gt.json --predicts_file ../contest/predictions.json --average 1 --strict 1
done
