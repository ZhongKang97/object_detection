#!/bin/bash

start=`date +%s`

# train and test
CUDA_VISIBLE_DEVICES=3 python holly_cifar.py \
--experiment_name=capsule_202_v2_KL_only_final \
--dataset=cifar10 \
--model_cifar=capsule \
--cap_model=v2 \
--cap_N=3 \
--epochs=300 \
--schedule_cifar 150 225 \
--optim=rmsprop \
--lr=0.0001 \
--route_num=2 \
--use_KL \
--multi_crop_test \
--port=3000

end=`date +%s`
runtime=$((end-start))

echo "Completed in" $runtime "seconds"
