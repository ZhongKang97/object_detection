#!/bin/bash

start=`date +%s`

# train and test
CUDA_VISIBLE_DEVICES=6 python holly_cifar.py \
--experiment_name=cifar_base_104_no_relu_adam \
--dataset=cifar \
--model_cifar=capsule \
--epochs=300 \
--schedule_cifar 150 225 \
--lr=0.01 \
--optim=adam \
--deploy

end=`date +%s`
runtime=$((end-start))

echo "Completed in" $runtime "seconds"
