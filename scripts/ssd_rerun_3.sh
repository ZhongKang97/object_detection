#!/bin/bash

start=`date +%s`

NAME=ssd_rerun_3
# train
CUDA_VISIBLE_DEVICES=4,5 python train.py \
--experiment_name=$NAME --debug_mode=False \
--dataset=coco \
--batch_size=4 --lr=0.0001 \
--optim=rmsprop \
--max_epoch=20 --schedule 6 12 16 \
--prior_config=v2_512 --ssd_dim=512 \
--resume=$NAME/train/ssd512_COCO_epoch_10_iter_29570

echo "TEST"
# test
CUDA_VISIBLE_DEVICES=4 python test.py \
--experiment_name=$NAME \
--dataset=coco \
--prior_config=v2_512 --ssd_dim=512 \
--trained_model=ssd512_COCO_epoch_19_iter_29570 \
--visualize_thres=.2

end=`date +%s`
runtime=$((end-start))

echo "Completed in" $runtime "seconds"
