python3 infer.py \
    --model_config_file='config/fastbert_cls_21.json' \
    --save_model_path='saved_model/fastbert_test_distill_21_08' \
    --inference_speed=0.8 \
    --infer_data='./sample/project/dev.tsv' \
    --dump_info_file='infer_info_21_08.txt' \
    --data_load_num_workers=2 \
    --gpu_ids=1 \
    --debug_break=0
