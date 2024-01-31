import argparse
import os
import torch
import mlflow
import shutil
import logging
import time

from exp.exp_informer import Exp_Informer
from urllib.parse import urlparse
from mlflow_utils.prepare_triton import prepare_traced_model, save_to_triton_flavor
from mlflow_utils import triton_flavor
from mlflow_utils.mlflow_exp import setup_mlflow_experiment

############### Arguments from Shell ##################

parser = argparse.ArgumentParser(description='[Informer] Long Sequences Forecasting')


parser.add_argument('--experiment_name', type=str, default='default_exp', help='Name of the MLFlow experiment')
parser.add_argument('--model', type=str, required=True, default='informer',help='model of experiment, options: [informer, informerstack, informerlight(TBD)]')

parser.add_argument('--data', type=str, required=True, default='ETTh1', help='data')
parser.add_argument('--root_path', type=str, default='./ETDataset/ETT-small/', help='root path of the data file')
parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')    
parser.add_argument('--features', type=str, default='M', help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
parser.add_argument('--freq', type=str, default='h', help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

parser.add_argument('--seq_len', type=int, default=96, help='input sequence length of Informer encoder')
parser.add_argument('--label_len', type=int, default=48, help='start token length of Informer decoder')
parser.add_argument('--pred_len', type=int, default=24, help='prediction sequence length')
# Informer decoder input: concat[start token series(label_len), zero padding series(pred_len)]

parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
parser.add_argument('--c_out', type=int, default=7, help='output size')
parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
parser.add_argument('--s_layers', type=str, default='3,2,1', help='num of stack encoder layers')
parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
parser.add_argument('--factor', type=int, default=5, help='probsparse attn factor')
parser.add_argument('--padding', type=int, default=0, help='padding type')
parser.add_argument('--distil', action='store_false', help='whether to use distilling in encoder, using this argument means not using distilling', default=True)
parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
parser.add_argument('--attn', type=str, default='prob', help='attention used in encoder, options:[prob, full]')
parser.add_argument('--embed', type=str, default='timeF', help='time features encoding, options:[timeF, fixed, learned]')
parser.add_argument('--activation', type=str, default='gelu',help='activation')
parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')
parser.add_argument('--mix', action='store_false', help='use mix attention in generative decoder', default=True)
parser.add_argument('--cols', type=str, nargs='+', help='certain cols from the data files as the input features')
parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')
parser.add_argument('--itr', type=int, default=2, help='experiments times')
parser.add_argument('--train_epochs', type=int, default=6, help='train epochs')
parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
parser.add_argument('--des', type=str, default='test',help='exp description')
parser.add_argument('--loss', type=str, default='mse',help='loss function')
parser.add_argument('--lradj', type=str, default='type1',help='adjust learning rate')
parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
parser.add_argument('--inverse', action='store_true', help='inverse output data', default=False)

parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
parser.add_argument('--gpu', type=int, default=0, help='gpu')
parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
parser.add_argument('--devices', type=str, default='0,1,2,3',help='device ids of multile gpus')

args = parser.parse_args()
args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

############### MLFlow ##################
tracking_server_ip =                'YOUR_MLFLOW_IP'           # Replace with the exposed IP of Ferran
tracking_service_port =             'YOUR_MLFLOW_PORT'         # Replace with the port number
tracking_server_username =          'YOUR_USERNAME'            # Replace with your username
tracking_server_password =          'YOUR_PASSWORD'            # Replace with your password
experiment_name = args.experiment_name

# Call the function
experiment_id = setup_mlflow_experiment(tracking_server_ip, tracking_service_port, tracking_server_username, tracking_server_password, experiment_name)

print("Experiment ID:", experiment_id, '\n')


# Rest

if args.use_gpu and args.use_multi_gpu:
    args.devices = args.devices.replace(' ','')
    device_ids = args.devices.split(',')
    args.device_ids = [int(id_) for id_ in device_ids]
    args.gpu = args.device_ids[0]

data_parser = {
    'Gtrace_5m': {'data':'Gtrace_5m.csv','T':'avg_cpu_usage','M':[10,10,10],'S':[1,1,1],'MS':[10,10,10]},
    'Gtrace_60m': {'data':'Gtrace_60m.csv','T':'avg_cpu_usage','M':[8,8,8],'S':[1,1,1],'MS':[7,7,1]},
}

if args.data in data_parser.keys():
    data_info = data_parser[args.data]
    args.data_path = data_info['data']
    args.target = data_info['T']
    args.enc_in, args.dec_in, args.c_out = data_info[args.features]

args.s_layers = [int(s_l) for s_l in args.s_layers.replace(' ','').split(',')]
args.detail_freq = args.freq
args.freq = args.freq[-1:]

# Check, print, and save a
print('Args in experiment:')
print(args, '\n')


Exp = Exp_Informer

with mlflow.start_run() as run:
    for ii in range(args.itr):
        # setting record of experiments
        setting = '{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_at{}_fc{}_eb{}_dt{}_mx{}_{}_{}'.format(args.model, args.data, args.features, 
                    args.seq_len, args.label_len, args.pred_len,
                    args.d_model, args.n_heads, args.e_layers, args.d_layers, args.d_ff, args.attn, args.factor, 
                    args.embed, args.distil, args.mix, args.des, ii)
        
        # Log parts of setting:
        mlflow.log_param("seq_len", args.seq_len)
        mlflow.log_param("n_heads", args.n_heads)
        mlflow.log_param("enc_lay", args.e_layers)
        mlflow.log_param("pred_len", args.pred_len)
        mlflow.log_param("dec_lay", args.d_layers)

        # Save args
        args_path = os.path.join(args.checkpoints, setting, 'args.pth')
        parent_dir = os.path.dirname(args_path)
        if not os.path.exists(parent_dir):
            os.makedirs(parent_dir)
        torch.save(args, args_path)

        exp = Exp(args) # set experiments
        print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
        train_start_time = time.time()
        model = exp.train(setting)
        train_end_time = time.time()
        mlflow.log_param("training_time", train_end_time - train_start_time)
        print('>>>>>>>end training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
        
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        test_results = exp.test(setting)

        # Log the test results to MLFlow
        mlflow.log_metrics({
            "mae": test_results[0],
            "mse": test_results[1],
            "rmse": test_results[2],
        })

        print('>>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        preds = exp.predict(setting, True)

        # Convert to TorchScript
        device = torch.device("cuda" if args.use_gpu else "cpu")  # determine the device to use
        model = model.to(device)
        traced_model, inputs_examples, input_shapes = prepare_traced_model(model, args)


        # Save to Triton Flavor 
        artifact_path='triton'  
        model_name = "informer"
        version = "1"
        tmp_folder_triton = model_name + '-' + artifact_path
        os.makedirs(tmp_folder_triton, exist_ok=True)
        save_to_triton_flavor(traced_model, inputs_examples, preds, tmp_folder_triton, setting, version)

        # Log with Triton MLFlow plugin to MLFlow server
        triton_flavor.log_model(tmp_folder_triton, artifact_path=artifact_path, registered_model_name=model_name)

        # S3 minio model uri for seldon delpoyment:
        S3_minio_model_uri = mlflow.get_artifact_uri(artifact_path)
        S3_minio_model_uri = S3_minio_model_uri.replace(":", "")
        print('\n\nUse this S3 model uri for your informer Seldon Deployment\n')
        print('s3://' + S3_minio_model_uri + '\n\n')
        
        # Emtpy cache
        torch.cuda.empty_cache()


