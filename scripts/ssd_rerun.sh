#!/bin/bash

start=`date +%s`

NAME=ssd_rerun
## train
#CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py \
#--experiment_name=$NAME --debug_mode=False \
#--dataset=coco \
#--batch_size=4 --lr=0.0001 \
#--optim=sgd \
#--max_epoch=20 --schedule 6 12 16 \
#--prior_config=v2_512 --ssd_dim=512

# test
CUDA_VISIBLE_DEVICES=0 python test.py \
--experiment_name=$NAME \
--dataset=coco \
--prior_config=v2_512 --ssd_dim=512 \
--trained_model=ssd512_COCO_epoch_20_iter_29570 \
--visualize_thres=.2


end=`date +%s`
runtime=$((end-start))

echo "Completed in" $runtime "seconds"
