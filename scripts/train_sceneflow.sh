#!/usr/bin/env bash
python train.py --config_file configs/models/dot_model.yaml \
            --data_config configs/data/sceneflow.yaml \
            --expname sceneflow \
            --summary_freq 500 \
            --dataset_path "D:/OneDrive/OneDrive - Univerza v Ljubljani/datasets/sceneflow/driving_finalpass"