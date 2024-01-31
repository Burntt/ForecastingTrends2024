import os
import torch
import numpy as np
import sys
import mlflow
import time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from mlflow_utils.mlflow_exp import setup_mlflow_experiment
from exp.exp_informer import Exp_Informer

############### MLFlow ##################
tracking_server_ip =                'YOUR_MLFLOW_IP'           # Replace with the exposed IP of Ferran
tracking_service_port =             'YOUR_MLFLOW_PORT'         # Replace with the port number
tracking_server_username =          'YOUR_USERNAME'            # Replace with your username
tracking_server_password =          'YOUR_PASSWORD'            # Replace with your password
experiment_name =                   'YOUR_EXPERIMENT_NAME'
setup_mlflow_experiment(tracking_server_ip, tracking_service_port, tracking_server_username, tracking_server_password, experiment_name)

logged_model = input("Enter the MLFLow Informer run-id: ")
script_dir = os.path.dirname(os.path.abspath(__file__))
artifacts_path = mlflow.artifacts.download_artifacts(logged_model, dst_path=script_dir)
print('Saved artifacts to: ', artifacts_path)

# Load model settings
args_path = os.path.join(artifacts_path, 'informer-triton/args.pth')
setting_path = os.path.join(artifacts_path, 'informer-triton/setting.txt')
args = torch.load(args_path)
args.inverse = True
with open(setting_path, 'r') as f:
    setting = f.read().strip()

print('SEQ LEN: ', args.seq_len)
print('PRED LEN: ', args.pred_len)

# Initialize Informer Experiment
exp = Exp_Informer(args)
num_inf_tests = 1000

# Modified Inference speed test function to record each measurement
def test_inference_speed(exp, setting, num_tests=num_inf_tests):
    inference_times = []
    for _ in range(num_tests):
        start_time = time.time()
        exp.predict(setting)  # Assuming predict method uses the same data every time
        inference_times.append(time.time() - start_time)
    return np.array(inference_times)

# Test inference speed for 1000 times
inference_times = test_inference_speed(exp, setting)

# Save the vector of inference times
parent_dir = os.path.dirname(script_dir)
compare_results_dir = os.path.join(parent_dir, 'compare_results')
informer_dir = os.path.join(compare_results_dir, 'informer')
os.makedirs(informer_dir, exist_ok=True)
np.save(os.path.join(informer_dir, 'inf_speed_vector_1000.npy'), inference_times)

print('\nSuccessfully saved vector of inference times over 1000 tests in compare_results/informer folder!')
