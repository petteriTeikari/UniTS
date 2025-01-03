model_name=UniTS
wandb_mode=disabled
project_name=anomaly_detection
exp_name=finetune_few_shot_anomaly_detection_PLR

# Path to the supervised checkpoint
# get ssl pretrained checkpoint: scripts/pretrain_prompt_learning/UniTS_pretrain_x32.sh
ckpt_path=newcheckpoints/units_x32_pretrain_checkpoint.pth
random_port=$((RANDOM % 9000 + 1000))

torchrun --nnodes 1 --nproc-per-node=1  --master_port $random_port  run.py \
  --fix_seed 2021 \
  --is_training 1 \
  --subsample_pct None \
  --model_id $exp_name \
  --pretrained_weight $ckpt_path \
  --model $model_name \
  --prompt_num 10 \
  --patch_len 16 \
  --stride 16 \
  --e_layers 3 \
  --d_model 32 \
  --des 'Exp' \
  --itr 1 \
  --lradj finetune_anl \
  --learning_rate 5e-4 \
  --weight_decay 1e-3 \
  --train_epochs 10 \
  --batch_size 256 \
  --acc_it 1 \
  --dropout 0 \
  --debug $wandb_mode \
  --project_name $project_name \
  --clip_grad 100 \
  --task_data_config_path data_provider/PLR_outlier.yaml \
  --mlflow-tracking-uri repo_desktop_clone/foundation_PLR/src/mlruns \
  --mlflow-experiment PLR_OutlierDetection \
  --mlflow-run UniTS-Outlier-orig-finetune
