nohup python3 main.py \
  --year 2017 \
  --epochs 40 \
  --batch_size 2 \
  --lr 1e-4 \
  --save_every 5 \
  > outputs/train_run_$(date +"%Y-%m-%d_%H-%M-%S").log 2>&1 &