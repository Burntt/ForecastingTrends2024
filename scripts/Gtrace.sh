
#####  main_informer_mlflow.py  #####
#####  main_informer_mlflow.py  #####
#####  main_informer_mlflow.py  #####

# Test (small)

python3 -u main_informer_mlflow.py --experiment_name dummy_exp_1 --model informer --data Gtrace_5m --root_path ./data/Gtrace2019/ --data_path Gtrace_5m.csv --features S --target avg_cpu_usage --freq 5min --checkpoints ./checkpoints --seq_len 24 --label_len 24 --pred_len 12 --enc_in 10 --dec_in 10 --c_out 10 --factor 10 --d_model 128 --n_heads 8 --e_layers 2 --d_layers 1 --d_ff 512 --dropout 0.05 --attn prob --embed timeF --activation gelu --distil --padding 0 --freq m --batch_size 32 --learning_rate 0.00001 --loss mse --lradj type1 --num_workers 0 --itr 1 --train_epochs 1 --patience 1 --des exp --gpu 0 --devices '0,1,2,3'

###### Lambda Labs (Big) #########

# WORKS: Light
sudo docker run --gpus all burntt/informer-training:v1 --experiment_name informer_exp_1 --model informer --data Gtrace_5m --root_path ./data/Gtrace2019/ --data_path Gtrace_5m.csv --features S --target avg_cpu_usage --freq 5min --checkpoints ./checkpoints --seq_len 24 --label_len 24 --pred_len 12 --enc_in 10 --dec_in 10 --c_out 10 --factor 10 --d_model 128 --n_heads 8 --e_layers 2 --d_layers 1 --d_ff 512 --dropout 0.05 --attn prob --embed timeF --activation gelu --distil --padding 0 --freq m --batch_size 32 --learning_rate 0.00001 --loss mse --lradj type1 --num_workers 0 --itr 1 --train_epochs 1 --patience 1 --des exp --gpu 0 --devices '0,1,2,3'

# WORKS: Heavy A10 + seq length 100
sudo docker run -it --gpus all burntt/informer-training:v1 --experiment_name informer_exp_1 --model informer --data Gtrace_5m --root_path ./data/Gtrace2019/ --data_path Gtrace_5m.csv --features S --target avg_cpu_usage --freq 5min --checkpoints ./checkpoints --seq_len 100 --label_len 50 --pred_len 100 --enc_in 8 --dec_in 8 --c_out 8 --factor 8 --d_model 512 --n_heads 6 --e_layers 6 --d_layers 3 --d_ff 512 --dropout 0.05 --attn prob --embed timeF --activation gelu --distil --padding 0 --freq m --batch_size 32 --learning_rate 0.00001 --loss mse --lradj type1 --num_workers 2 --itr 1 --train_epochs 5 --patience 2 --des exp --gpu 0 --devices '0,1,2,3'

# INFORMER PAPER ALIGNED COMMAND:
sudo docker run -it --gpus all burntt/informer-training:v1 --experiment_name informer_exp_1 --model informer --data Gtrace_5m --root_path ./data/Gtrace2019/ --data_path Gtrace_5m.csv --features S --target avg_cpu_usage --freq 5min --checkpoints ./checkpoints --seq_len 120 --label_len 60 --pred_len 120 --enc_in 8 --dec_in 8 --c_out 8 --factor 8 --d_model 512 --n_heads 16 --e_layers 4 --d_layers 2 --d_ff 2048 --dropout 0.1 --attn prob --embed timeF --activation gelu --distil --padding 0 --freq m --batch_size 32 --learning_rate 0.0001 --loss mse --lradj type1 --num_workers 2 --itr 1 --train_epochs 10 --patience 4 --des exp --gpu 0 --devices '0,1,2,3'

# local test
python3 -u main_informer_mlflow.py --experiment_name informer_exp_1 --model informer --data Gtrace_5m --root_path ./data/Gtrace2019/ --data_path Gtrace_5m.csv --features S --target avg_cpu_usage --freq 5min --checkpoints ./checkpoints --seq_len 24 --label_len 24 --pred_len 24 --enc_in 8 --dec_in 8 --c_out 8 --factor 8 --d_model 512 --n_heads 16 --e_layers 4 --d_layers 2 --d_ff 2048 --dropout 0.1 --attn prob --embed timeF --activation gelu --distil --padding 0 --freq m --batch_size 24 --learning_rate 0.0001 --loss mse --lradj type1 --num_workers 0 --itr 1 --train_epochs 10 --patience 4 --des exp --gpu 0 --devices '0,1,2,3'

--data Gtrace_5m --root_path ./data/Gtrace2019/ --data_path Gtrace_5m.csv


## #

sudo docker run -it --gpus all burntt/informer-training:v1 \
          --experiment_name informers_gtrace_pred_fixed \
          --model informer \
          --data Gtrace_5m \
          --root_path ./data/Gtrace2019/ \
          --data_path Gtrace_5m.csv \
          --features S \
          --target avg_cpu_usage \
          --checkpoints ./checkpoints \
          --seq_len 48 \
          --label_len 24 \
          --pred_len 24 \
          --enc_in 8 \
          --dec_in 8 \
          --c_out 8 \
          --factor 8 \
          --d_model 512 \
          --n_heads 8 \
          --e_layers 4 \
          --d_layers 2 \
          --d_ff 2048 \
          --dropout 0.1 \
          --attn prob \
          --embed timeF \
          --activation gelu \
          --padding 0 \
          --freq m \
          --batch_size 24 \
          --learning_rate 0.0001 \
          --loss mse \
          --lradj type1 \
          --num_workers 2 \
          --itr 1 \
          --train_epochs 10 \
          --patience 4 \
          --des exp \
          --gpu 0 \
          --devices '0,1,2,3'