python3 train.py \
    --model_config_file='config/fastbert_cls.json' \
    --save_model_path='saved_model/fastbert_test' \
    --run_mode=eval \
    --eval_data='./sample/ChnSentiCorp/dev.tsv' \
    --batch_size=32 \
    --data_load_num_workers=2 \
    --gpu_ids='0,1' \
    --debug_break=0
