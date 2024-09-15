export PYTHONWARNINGS='ignore:semaphore_tracker:UserWarning'
export CUDA_VISIBLE_DEVICES=0,1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS=1

python main.py \
    --use_color --use_normal \
    --detector detector_Vote2Cap_DETR \
    --captioner ll3da \
    --checkpoint_dir ./ckpts/Qwen1.5-7B/ll3da-generalist \
    --test_ckpt ./ckpts/Qwen1.5-7B/ll3da-generalist/checkpoint_best.pth \
    --dataset unified_3dllm,unified_scanqa,unified_densecap_nr3d,unified_densecap_scanrefer \
    --vocab Qwen/Qwen1.5-7B \
    --qformer_vocab bert-base-embedding \
    --dist_url tcp://localhost:12345 \
    --criterion 'CiDEr' \
    --freeze_detector --freeze_llm \
    --batchsize_per_gpu 4 --ngpus 2 \
    --max_des_len 512 \
    --max_prompt 1 \
    --use_beam_search \
    --test_only
