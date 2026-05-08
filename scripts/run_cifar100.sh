#!/bin/bash
# CIFAR100 — CoLOR + curated baselines (DD, uPU, nnPU, SAR-EM, BODASaito)
# Splits available for seeds: 0, 8, 103, 573, 1057
# Run from the repo root: bash scripts/run_cifar100.sh

set -e
SEEDS=( 8 103 573 1057 )
GPU_IDS=( 0 1 2 3 )
NUM_GPUS=${#GPU_IDS[@]}

# CIFAR100 shift hyperparams used in the paper
DATASET=cifar100
NUM_SOURCE_CLASSES=85
FRAC_OOD=0.35
OOD_CLASS=2
OOD_RATIO=0.5

i=0
for seed in "${SEEDS[@]}"; do
    gpu=${GPU_IDS[$((i % NUM_GPUS))]}

    # CoLOR
    CUDA_VISIBLE_DEVICES=$gpu python run.py -m \
        models=precision_at_recall datamodule=random_split_module \
        dataset=$DATASET seed=$seed num_source_classes=$NUM_SOURCE_CLASSES \
        fraction_ood_class=$FRAC_OOD ood_class=$OOD_CLASS ood_class_ratio=$OOD_RATIO \
        use_labels=True &
    sleep 5

    # DD (sourceDiscriminator)
    CUDA_VISIBLE_DEVICES=$gpu python run.py -m \
        models=sourceDiscriminator datamodule=random_split_module \
        dataset=$DATASET seed=$seed num_source_classes=$NUM_SOURCE_CLASSES \
        fraction_ood_class=$FRAC_OOD ood_class=$OOD_CLASS ood_class_ratio=$OOD_RATIO &
    sleep 5

    # uPU
    CUDA_VISIBLE_DEVICES=$gpu python run.py -m \
        models=nnPU datamodule=random_split_module nnPU=False \
        dataset=$DATASET seed=$seed num_source_classes=$NUM_SOURCE_CLASSES \
        fraction_ood_class=$FRAC_OOD ood_class=$OOD_CLASS ood_class_ratio=$OOD_RATIO &
    sleep 5

    # nnPU
    CUDA_VISIBLE_DEVICES=$gpu python run.py -m \
        models=nnPU datamodule=random_split_module nnPU=True \
        dataset=$DATASET seed=$seed num_source_classes=$NUM_SOURCE_CLASSES \
        fraction_ood_class=$FRAC_OOD ood_class=$OOD_CLASS ood_class_ratio=$OOD_RATIO &
    sleep 5

    # SAR-EM
    CUDA_VISIBLE_DEVICES=$gpu python run.py -m \
        models=sarem datamodule=random_split_module \
        dataset=$DATASET seed=$seed num_source_classes=$NUM_SOURCE_CLASSES \
        fraction_ood_class=$FRAC_OOD ood_class=$OOD_CLASS ood_class_ratio=$OOD_RATIO &
    sleep 5

    # BODASaito
    CUDA_VISIBLE_DEVICES=$gpu python run.py -m \
        models=BODASaito datamodule=random_split_module \
        dataset=$DATASET seed=$seed num_source_classes=$NUM_SOURCE_CLASSES \
        fraction_ood_class=$FRAC_OOD ood_class=$OOD_CLASS ood_class_ratio=$OOD_RATIO &
    sleep 5

    wait
    i=$((i+1))
done
