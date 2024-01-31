import json
import os
import requests
import torch
import numpy as np
import sys
import mlflow
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.tools import generate_and_convert_tensors
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

# Select model
logged_model = 'runs:/1d29d6650a9a4512afe51a9dc2e14365/triton'

# Load model.
script_dir = os.path.dirname(os.path.abspath(__file__))
artifacts_path = mlflow.artifacts.download_artifacts(logged_model, dst_path=script_dir)
args_path = artifacts_path + '/informer-triton/args.pth'
setting_path = artifacts_path + '/informer-triton/setting.txt'

# Load setting from artifacts and setting
args = torch.load(args_path)
settings_file_path = artifacts_path + '/informer-triton/setting.txt'
with open(settings_file_path, 'r') as f:
    setting = f.read().strip()

# Do a prediction
Exp = Exp_Informer
exp = Exp(args) 
output, _ = exp.predict(setting, True)

# Generate and convert the tensors
tensor_data = generate_and_convert_tensors(args)

# Now you can access the data for each tensor
example_x_enc = tensor_data["x_enc"]
example_x_mark_enc = tensor_data["x_mark_enc"]
example_x_dec = tensor_data["x_dec"]
example_x_mark_dec = tensor_data["x_mark_dec"]

example_enc_self_mask = tensor_data["enc_self_mask"]
example_dec_self_mask = tensor_data["dec_self_mask"]
example_dec_enc_mask = tensor_data["dec_enc_mask"]


# Get output dimensions, excluding the batch size
output_dim = output.shape[1:]

inference_request = {
    "inputs": [
        {
            "name": "input0",
            "shape": [args.batch_size, args.seq_len, args.enc_in],
            "datatype": "FP32",
            "data": example_x_enc
        },
        {
            "name": "input1",
            "shape": [args.batch_size, args.seq_len, 1],
            "datatype": "FP32",
            "data": example_x_mark_enc
        },
        {
            "name": "input2",
            "shape": [args.batch_size, args.pred_len, args.dec_in],
            "datatype": "FP32",
            "data": example_x_dec
        },
        {
            "name": "input3",
            "shape": [args.batch_size, args.pred_len, 1],
            "datatype": "FP32",
            "data": example_x_mark_dec
        },
        {
            "name": "input4",
            "shape": [args.batch_size, args.seq_len, args.seq_len],
            "datatype": "FP32",
            "data": example_enc_self_mask
        },
        {
            "name": "input5",
            "shape": [args.batch_size, args.pred_len, args.pred_len],
            "datatype": "FP32",
            "data": example_dec_self_mask
        },
        {
            "name": "input6",
            "shape": [args.batch_size, args.pred_len, args.seq_len],
            "datatype": "FP32",
            "data": example_dec_enc_mask
        },
    ],
    "outputs": [
        {
            "name": "output0",
            "datatype": "FP32",
            "shape": [-1] + list(output_dim)
        }
    ]
}

# Coded Endpoint:     https://10.201.0.135:31001/seldon/inf-informer/informer-triton/v2/models/infer 

node_ip = '10.201.0.135'
emissary_ingress_port = '31001'
model_namespace = 'inf-informer'
model_name = 'informer-triton'
tail_url = 'v2/models/infer'

endpoint = "https://{}:{}/seldon/{}/{}/{}".format(node_ip, emissary_ingress_port, model_namespace, model_name, tail_url)
response = requests.post(endpoint, json=inference_request, verify=False)


print('\nCoded Endpoint:                    ', endpoint, '\n')
print('\nResponse:                          ', response)
print('Response HTTP Status Code:           ', response.status_code)
print('Response HTTP Response Body:         ', response.content)
