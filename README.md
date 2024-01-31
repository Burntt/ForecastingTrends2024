![image](https://github.com/Burntt/Forecasting_Trends_2024/assets/69801109/42423542-1ea6-4a9e-b08b-f53dcc6a88ee)

# Forecasting Trends in Cloud-Edge Computing with Informer

## Overview
This repository is dedicated to the Informer model, a cutting-edge approach designed to forecast cloud-edge computing trends. By leveraging sophisticated attention mechanisms, Informer provides precise and actionable insights into future computational demands and optimization strategies.

## Features
- **Innovative Forecasting**: Employs the Informer model, integrating attention mechanisms for superior accuracy in trend prediction.
- **Tailored for Cloud-Edge**: Specifically designed for the dynamics of cloud and edge computing environments, ensuring relevance and applicability.
- **Efficient and Scalable**: Built to accommodate large datasets and complex computations with high efficiency.

## License
This project is under the MIT License. See the LICENSE file for more details.

## Citations
```
@article{Gort2024TrendsCloudEdgeComputing,
  title={Forecasting Trends in Cloud-Edge Computing: Unleashing the Power of Attention Mechanisms},
  author={Berend J.D. Gort and Maria A. Serrano and Angelos Antonopoulos},
  journal={ArXiv},
  year={2024},
  volume={abs/2401.00001},
  organization={Nearby Computing},
  address={Barcelona, Spain},
  email={{berend.gort, maria.serrano, angelos.antonopoulos}@nearbycomputing.com},
  abstract={
    "This project introduces an innovative approach to forecast trends in cloud-edge computing using advanced attention mechanisms. Our model, named 'Informer', leverages deep learning to analyze temporal data, providing insights into future computing needs and optimizations."
  },
  keywords={Zero-touch, AI, LSTM, Transformers, Informers, Edge Computing, Time-series Forecasting, Deep Learning}
}
```

## Running Dockerized Training Loop with MLFlow

1. **Prepare Docker Environment**: 
   - Ensure Docker is installed and configured on your system.
   - For GPU support, ensure NVIDIA Docker is set up.

2. **Build or Pull Docker Image**: 
   - If you have a `Dockerfile`, build your image with `docker build -t your-image-name .`.
   - Or, pull an existing image using `docker pull your-image-name`.

3. **Set Up MLFlow**: 
   - Ensure MLFlow is installed and configured, either locally or on a server.
   - Set up tracking server details such as IP, port, username, and password in the script.

4. **Prepare Data and Scripts**: 
   - Make sure your data and scripts are accessible to the Docker container.
   - Use volume mounts (`-v /host/path:/container/path`) to make host directories available inside the container.

5. **Run Docker Container**: 
   - Use `docker run` with appropriate flags and arguments to start your training loop. For GPUs, add `--gpus all`.
   - Include all necessary arguments for your script, as specified in the `argparse` setup.

Example Command for Running the Docker Container:
```shell
sudo docker run -it --gpus all your-image-name \
          --experiment_name your_experiment \
          --model informer \
          --data your_data \
          --root_path ./data/path \
          --data_path data.csv \
          --features M \
          --target your_target \
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
```

## Running the Main Script of Training Loop (without MLFLow + Docker)

1. **Prepare Environment**: 
   - Ensure Python, PyTorch, and other required libraries are installed.
   - If using a GPU, verify CUDA is properly set up.

2. **Set Up Parameters**: 
   - Use the `argparse` module to define and parse command-line arguments.
   - Adjust parameters as needed for your experiment, such as `--model`, `--data`, `--seq_len`, etc.

3. **Initialize and Configure Model**: 
   - Create an instance of the `Exp_Informer` class or your model class with the parsed arguments.
   - Ensure the model and data are compatible with your hardware, using GPUs if available.

4. **Run Training Loop**: 
   - Execute the `train` method of your model instance to start the training process.
   - Monitor the training progress and performance metrics.

5. **Test and Evaluate**: 
   - After training, use the `test` method to evaluate the model on a test dataset.
   - Analyze the results to assess model performance.

6. **Optional Prediction**: 
   - If your script supports prediction on unseen data (`--do_predict`), execute the `predict` method.
   - Review prediction outcomes for further analysis or application.

## Acknowledgments
We extend our gratitude to the authors and contributors of the original Informer GitHub repository for their foundational work, which facilitated the rapid adaptation and implementation of their code into our project. Their innovative approach to long sequence time-series forecasting has been instrumental in guiding our research direction.