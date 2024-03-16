#!/bin/bash
source activate
conda activate ffcv-ssl
python /scratch/gpfs/eh0560/repos/FFCV-SSL/examples/train_ssl.py \
  --config-file /scratch/gpfs/eh0560/repos/FFCV-SSL/examples/rn50_configs/rn50_baseline.yaml \
  --logging.folder /scratch/gpfs/eh0560/repos/FFCV-SSL/runs \
  --data.train_dataset=/scratch/gpfs/eh0560/data/imagenet_ffcv/imagenet_train.beton \
  --data.val_dataset=/scratch/gpfs/eh0560/data/imagenet_ffcv/imagenet_val.beton \
  --data.num_workers=16 \
  --data.in_memory=1 \
  --training.batch_size 1024 \
  --training.epochs 1000 \
  --training.use_ssl 1 \
  --model.remove_head 0 \
  --model.mlp 8192-8192-8192 \
  --model.fc 0 \
  --model.proj_relu 0 \
  --training.loss byol \
  --dist.use_submitit 0 \
  --training.optimizer adamw \
  --dist.ngpus 4 \
  --dist.world_size 4 \
  --training.base_lr 0.0005 \
  0