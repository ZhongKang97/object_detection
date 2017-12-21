#!/bin/bash

start=`date +%s`

# train
CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py \
--experiment_name=ssd_rerun --debug_mode=False \
--dataset=coco \
--batch_size=4 --lr=0.0001 \
--optim=rmsprop \
--max_epoch=20 --schedule 6 12 16 \
--prior_config=v2_512 --ssd_dim=512

end=`date +%s`
runtime=$((end-start))

echo "Completed in" $runtime "seconds"
