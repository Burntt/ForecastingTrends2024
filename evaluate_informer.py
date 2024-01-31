import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import sys
import mlflow
from models.model import Informer, InformerStack

from sklearn.metrics import mean_squared_error
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from mlflow_utils.mlflow_exp import setup_mlflow_experiment
from exp.exp_informer import Exp_Informer

############### MLFlow ##################
tracking_server_ip =                'YOUR_MLFLOW_IP'           # Replace with the exposed IP of Ferran
tracking_service_port =             'YOUR_MLFLOW_PORT'         # Replace with the port number
tracking_server_username =          'YOUR_USERNAME'            # Replace with your username
tracking_server_password =          'YOUR_PASSWORD'            # Replace with your password
experiment_name =                   'YOUR_EXPERIMENT_NAME'

# Call the function
setup_mlflow_experiment(tracking_server_ip, tracking_service_port, tracking_server_username, tracking_server_password, experiment_name)

# Select Informer model from MLFlow
logged_model = input("Enter the MLFLow Informer run-id: ")

# Load model.
script_dir = os.path.dirname(os.path.abspath(__file__))
artifacts_path = mlflow.artifacts.download_artifacts(logged_model, dst_path=script_dir)
args_path = artifacts_path + '/informer-triton/args.pth'
setting_path = artifacts_path + '/informer-triton/setting.txt'

# Load setting from artifacts and setting
args = torch.load(args_path)

args.inverse = True

print('SEQ LEN: ', args.seq_len)
print('PRED LEN: ', args.pred_len)

with open(setting_path, 'r') as f:
    setting = f.read().strip()

# Do a prediction
exp = Exp_Informer(args)

# Load model from .pt file
pt_model_path = './triton/informer-triton/1/model.pt'

# Call the test method with pt_model_path and load from local folder
exp.test(setting, pt_model_path=pt_model_path)
preds = np.load('./results/' + setting + '/pred.npy')
trues = np.load('./results/' + setting + '/true.npy')

# Check shapes
print('PREDS SHAPE: ', preds.shape)
print('TRUES SHAPE: ', trues.shape)


# Save results for analysis - Get the current script directory and go one level up
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)

# Create or locate the folder 'compare_results'
compare_results_dir = os.path.join(parent_dir, 'compare_results')
os.makedirs(compare_results_dir, exist_ok=True)

# Create 'informer' subfolder within 'compare_results'
informer_dir = os.path.join(compare_results_dir, 'informer')
os.makedirs(informer_dir, exist_ok=True)

# Save preds and trues in 'informer' subfolder
np.save(os.path.join(informer_dir, 'preds.npy'), preds)
np.save(os.path.join(informer_dir, 'trues.npy'), trues)

print('\n Succesfully saved predictions and groundtruth in compare_results folder!!')

