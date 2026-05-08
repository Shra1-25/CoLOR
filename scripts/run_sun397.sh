#!/bin/bash
# SUN397 — CoLOR + curated baselines (DD, uPU, nnPU, SAR-EM, BODASaito)
# Splits available for seeds: 0, 8, 103, 573, 1057
# Requires precomputed CLIP / ResNet50 features (see README "Datasets" section).
# Run from the repo root: bash scripts/run_sun397.sh

set -e
SEEDS=( 8 103 573 1057 )
GPU_IDS=( 0 1 2 3 )
NUM_GPUS=${#GPU_IDS[@]}

DATASET=sun397
ARCH=CLIP_ViT-L14
NUM_SOURCE_CLASSES=358
FRAC_OOD=0.13
OOD_CLASS=0
OOD_RATIO=0.5

i=0
for seed in "${SEEDS[@]}"; do
    gpu=${GPU_IDS[$((i % NUM_GPUS))]}

    # CoLOR
    CUDA_VISIBLE_DEVICES=$gpu python run.py -m \
        models=precision_at_recall datamodule=sun397_datamodule \
        dataset=$DATASET arch=$ARCH seed=$seed num_source_classes=$NUM_SOURCE_CLASSES \
        fraction_ood_class=$FRAC_OOD ood_class=$OOD_CLASS ood_class_ratio=$OOD_RATIO \
        use_labels=True &
    sleep 5

    # DD
    CUDA_VISIBLE_DEVICES=$gpu python run.py -m \
        models=sourceDiscriminator datamodule=sun397_datamodule \
        dataset=$DATASET arch=$ARCH seed=$seed num_source_classes=$NUM_SOURCE_CLASSES \
        fraction_ood_class=$FRAC_OOD ood_class=$OOD_CLASS ood_class_ratio=$OOD_RATIO &
    sleep 5

    # uPU
    CUDA_VISIBLE_DEVICES=$gpu python run.py -m \
        models=nnPU datamodule=sun397_datamodule nnPU=False \
        dataset=$DATASET arch=$ARCH seed=$seed num_source_classes=$NUM_SOURCE_CLASSES \
        fraction_ood_class=$FRAC_OOD ood_class=$OOD_CLASS ood_class_ratio=$OOD_RATIO &
    sleep 5

    # nnPU
    CUDA_VISIBLE_DEVICES=$gpu python run.py -m \
        models=nnPU datamodule=sun397_datamodule nnPU=True \
        dataset=$DATASET arch=$ARCH seed=$seed num_source_classes=$NUM_SOURCE_CLASSES \
        fraction_ood_class=$FRAC_OOD ood_class=$OOD_CLASS ood_class_ratio=$OOD_RATIO &
    sleep 5

    # SAR-EM
    CUDA_VISIBLE_DEVICES=$gpu python run.py -m \
        models=sarem datamodule=sun397_datamodule \
        dataset=$DATASET arch=$ARCH seed=$seed num_source_classes=$NUM_SOURCE_CLASSES \
        fraction_ood_class=$FRAC_OOD ood_class=$OOD_CLASS ood_class_ratio=$OOD_RATIO &
    sleep 5

    # BODASaito
    CUDA_VISIBLE_DEVICES=$gpu python run.py -m \
        models=BODASaito datamodule=sun397_datamodule \
        dataset=$DATASET arch=$ARCH seed=$seed num_source_classes=$NUM_SOURCE_CLASSES \
        fraction_ood_class=$FRAC_OOD ood_class=$OOD_CLASS ood_class_ratio=$OOD_RATIO &
    sleep 5

    wait
    i=$((i+1))
done
