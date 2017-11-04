#!/bin/bash

start=`date +%s`

# train and test
CUDA_VISIBLE_DEVICES=2 python holly_cifar.py \
--experiment_name=cifar_base_101_1_depth_20 \
--dataset=cifar \
--model_cifar=resnet \
--epochs=300 \
--schedule_cifar 150 225 \
--lr=0.1 \
--train_batch=128 \
--test_batch=128 \
--deploy

end=`date +%s`
runtime=$((end-start))

echo "Completed in" $runtime "seconds"
