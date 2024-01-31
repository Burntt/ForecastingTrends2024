#!/bin/bash

# Function to display warnings or confirmations
display_message() {
  if [ "$1" == "warning" ]; then
    echo -e "\e[41m\e[97m  CRITICAL WARNING  \e[0m"
    echo -e "\e[1;31mSelf-destruct script (selfdestruct_instance.py) not found.\e[0m"
    echo -e "\e[1;31mInstances will NOT be terminated automatically. Immediate action required.\e[0m"
  else
    echo -e "\e[42m\e[97m  CONFIRMATION  \e[0m"
    echo -e "\e[1;32mSelf-destruct script (selfdestruct_instance.py) found. Instances will be terminated automatically.\e[0m"
  fi
}

# Stop and remove all existing containers if any exist
echo -e "\nStopping and deleting all existing containers to clear GPU memory"
if [ ! -z "$(sudo docker ps -q)" ]; then
  sudo docker stop $(sudo docker ps -q)
fi

if [ ! -z "$(sudo docker ps -a -q)" ]; then
  sudo docker rm $(sudo docker ps -a -q)
fi

# Sequence lengths, Encoder layers, Number of attention heads
seq_lengths=("24" "48" "168" "360" "720")
e_layers=("6" "4" "2")
n_heads=("8" "16")

# Max number of parallel runs
max_parallel_runs=2

# Initialize job queue and metadata
declare -a job_queue
declare -A job_meta
finished_jobs=0
total_jobs=0

# Function to draw progress bar
draw_progress_bar() {
    progress=$1
    total=$2
    percent=$(( ( $progress * 100 ) / $total ))
    echo -ne "Progress: ["
    for i in {1..100}; do
        if [ $i -le $percent ]; then
            echo -ne "#"
        else
            echo -ne " "
        fi
    done
    echo -ne "] $percent%\r"
}

# Populate job queue
for seq_len in "${seq_lengths[@]}"; do
  for e_layer in "${e_layers[@]}"; do
    for n_head in "${n_heads[@]}"; do
      job="sudo docker run -d --gpus all burntt/informer-training:v1 \
          --experiment_name informers_gtrace_seq_is_pred \
          --model informer \
          --data Gtrace_5m \
          --root_path ./data/Gtrace2019/ \
          --data_path Gtrace_5m.csv \
          --features S \
          --target avg_cpu_usage \
          --checkpoints ./checkpoints \
          --seq_len "$seq_len" \
          --label_len "$seq_len" \
          --pred_len "$seq_len" \
          --enc_in 8 \
          --dec_in 8 \
          --c_out 8 \
          --factor 8 \
          --d_model 512 \
          --n_heads "$n_head" \
          --e_layers "$e_layer" \
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
          --devices '0,1,2,3'"
      job_queue+=("$job")
      total_jobs=$((total_jobs + 1))
      echo "Added job with seq_len=$seq_len, e_layer=$e_layer, n_head=$n_head to the queue."
    done
  done
done


# Check for selfdestruct_instance.py and display appropriate message
if [ ! -f "selfdestruct_instance.py" ]; then
  display_message "warning"
else
  display_message "confirmation"
fi

echo "About to enter main loop."


while [ ${#job_queue[@]} -gt 0 ] || [ ${#job_meta[@]} -gt 0 ]; do

  running_jobs=$(sudo docker ps -q | wc -l)
  echo "Current running jobs count : $running_jobs"
  echo "Current job_queue length   : ${#job_queue[@]}"
  echo "Current job_meta keys      : ${!job_meta[@]}"
  
  if [ $running_jobs -lt $max_parallel_runs ] && [ ${#job_queue[@]} -gt 0 ]; then
    job=${job_queue[0]}
    job_queue=("${job_queue[@]:1}")
    container_id=$(eval "$job")
    if [ $? -ne 0 ]; then
      echo "Failed to start Docker container."
      continue
    fi
    job_meta["$container_id"]="running"
    echo "Started new job with container ID: $container_id"
  fi

  for id in $(sudo docker ps --no-trunc -aq); do  # Use --no-trunc to get full container IDs
    status=$(sudo docker inspect --format '{{.State.Status}}' $id 2>/dev/null)
    if [[ ${job_meta[$id]} && ("$status" == "exited" || "$status" == "error") ]]; then
      sudo docker rm $id
      unset job_meta[$id]
      finished_jobs=$((finished_jobs + 1))
      echo "Removed finished job with container ID: $id"
    fi
  done
  
  # Draw progress bar
  echo -e "\n\n========== Monitoring Stats =========="
  nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv
  sudo docker ps
  echo -e "\nProgress - finished_jobs: $finished_jobs, total_jobs: $total_jobs \n"
  draw_progress_bar $finished_jobs $total_jobs
  sleep 10
done

echo -e "\nAll jobs completed."

# Check if selfdestruct_instance.py exists and run it
if [ -f "selfdestruct_instance.py" ]; then
  echo "Running self-destruct script."
  python3 selfdestruct_instance.py
else
  echo -e "\e[41m\e[97m  WARNING  \e[0m \e[1;31mSelf-destruct script (selfdestruct_instance.py) not found. Instances will not be terminated automatically.\e[0m"
fi