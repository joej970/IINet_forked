#!/usr/bin/env bash
python train.py --config_file configs/models/dot_model.yaml \
            --data_config configs/data/sceneflow.yaml \
            --expname sceneflow \
            --summary_freq 500 \
	    --eval_freq 1 \
	    --save_freq 1 \
            --dataset_path "/d/hpc/home/zr1677/datasets/sceneflow/driving_finalpass"
