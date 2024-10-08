export PYTHONWARNINGS='ignore:semaphore_tracker:UserWarning'
export CUDA_VISIBLE_DEVICES=0,1,2,3
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS=1

python main.py \
    --use_color --use_normal \
    --detector detector_Vote2Cap_DETR \
    --captioner ll3da \
    --pretrained_weights ./pretrained/vote2cap-detr/scannet_vote2cap_detr_XYZ_COLOR_NORMAL.pth \
    --warm_lr_epochs 1 \
    --dataset unified_3dllm,unified_scanqa,unified_densecap_nr3d,unified_densecap_scanrefer \
    --vocab Qwen/Qwen1.5-7B \
    --qformer_vocab bert-base-embedding \
    --checkpoint_dir ./ckpts/Qwen1.5-7B/ll3da-generalist \
    --max_epoch 32 \
    --dist_url tcp://localhost:12345 \
    --eval_every_iteration 10000 \
    --start_eval_after 19999 \
    --save_every 10000 \
    --criterion 'CiDEr' \
    --freeze_detector --freeze_llm \
    --batchsize_per_gpu 4 --ngpus 4 --base_lr 1e-4 --final_lr 1e-6 \
    --max_des_len 512 \
    --max_prompt 1 --use_beam_search
