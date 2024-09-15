export PYTHONWARNINGS='ignore:semaphore_tracker:UserWarning'
export CUDA_VISIBLE_DEVICES=1,2
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS=1

python main.py \
    --use_color --use_normal \
    --detector detector_Vote2Cap_DETR \
    --captioner ll3da \
    --checkpoint_dir ./ckpts/Qwen1.5-7B/ll3da-scanrefer-tuned-ll3da \
    --test_ckpt ./ckpts/Qwen1.5-7B/ll3da-scanrefer-tuned-ll3da/checkpoint_best.pth \
    --dataset unified_densecap_scanrefer \
    --vocab Qwen/Qwen1.5-7B  \
    --qformer_vocab bert-base-embedding \
    --dist_url tcp://localhost:1222 \
    --criterion 'CiDEr@0.5' \
    --freeze_detector --freeze_llm \
    --batchsize_per_gpu 8 --ngpus 2 \
    --max_des_len 256 \
    --max_prompt 1 \
    --use_beam_search \
    --test_only