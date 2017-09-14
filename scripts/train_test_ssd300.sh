#!/bin/bash

start=`date +%s`

CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --save_folder=renew_300_new_scale --batch_size=32 --ssd_dim=300 --deploy

CUDA_VISIBLE_DEVICES=3 python eval.py --experiment_name=renew_300_new_scale \
--trained_model=final_v2.pth --ssd_dim=300 --conf_thresh=0.005 --top_k=300 --nms_thresh=0.45

end=`date +%s`
runtime=$((end-start))

echo "Completed in" $runtime "seconds"
