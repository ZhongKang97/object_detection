#!/bin/bash

start=`date +%s`

CUDA_VISIBLE_DEVICES=4,5,6,7 python train.py \
--save_folder=renew_512_new_scale/ --debug=False --batch_size=32 --ssd_dim=512

CUDA_VISIBLE_DEVICES=7 python eval.py --experiment_name=renew_512_new_scale/ \
--trained_model=final_v2.pth --ssd_dim=512 --conf_thresh=0.005 --top_k=300 --nms_thresh=0.45

end=`date +%s`
runtime=$((end-start))

echo "Completed in" $runtime "seconds"
