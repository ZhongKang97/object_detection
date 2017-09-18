#!/bin/bash

start=`date +%s`

# train
#CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py \
#--save_folder=renew_512_set4 --batch_size=20 --ssd_dim=512 --max_iter=100000 \
#--prior_config=v2_512_stan_more_ar --lr=1e-3 --schedule=60000,80000,90000 --gamma=0.5 \
#--deploy

# test
#CUDA_VISIBLE_DEVICES=1 python eval.py --experiment_name=renew_512_set4 \
#--trained_model=ssd512_0712_iter_65000.pth \
#--ssd_dim=512 --conf_thresh=0.005 --top_k=300 --nms_thresh=0.45 \
#--prior_config=v2_512_stan_more_ar

#CUDA_VISIBLE_DEVICES=1 python eval.py --experiment_name=renew_512_set4 \
#--trained_model=ssd512_0712_iter_75000.pth \
#--ssd_dim=512 --conf_thresh=0.005 --top_k=300 --nms_thresh=0.45 \
#--prior_config=v2_512_stan_more_ar

CUDA_VISIBLE_DEVICES=1 python eval.py --experiment_name=renew_512_set4 \
--trained_model=ssd512_0712_iter_85000.pth \
--ssd_dim=512 --conf_thresh=0.01 --top_k=500 --nms_thresh=0.5 \
--prior_config=v2_512_stan_more_ar --sub_folder_suffix=set1

CUDA_VISIBLE_DEVICES=1 python eval.py --experiment_name=renew_512_set4 \
--trained_model=final_v2.pth \
--ssd_dim=512 --conf_thresh=0.01 --top_k=500 --nms_thresh=0.5 \
--prior_config=v2_512_stan_more_ar --sub_folder_suffix=set1


end=`date +%s`
runtime=$((end-start))

echo "Completed in" $runtime "seconds"
