import mlflow
import torch
import os

def setup_mlflow_experiment(tracking_server_ip, tracking_service_port, tracking_server_username, tracking_server_password, experiment_name):
    """
    This function sets up an MLflow experiment.

    Parameters:
    tracking_server_ip: The IP of the tracking server.
    tracking_service_port: The port of the tracking service.
    tracking_server_username: The username for the tracking server.
    tracking_server_password: The password for the tracking server.
    experiment_name: The name of the MLflow experiment.

    Returns:
    experiment_id: The ID of the created or existing MLflow experiment.
    """

    # Setup the tracking URI
    tracking_uri = "http://{}:{}@{}:{}".format(tracking_server_username, tracking_server_password, 
                                               tracking_server_ip, tracking_service_port)
    print('\n\ntracking_uri set to:', tracking_uri)
    mlflow.set_tracking_uri(tracking_uri)

    # Check if the experiment exists, if not create a new one
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        experiment_id = mlflow.create_experiment(experiment_name)
        print("Created new experiment with ID:", experiment_id)
    else:
        experiment_id = experiment.experiment_id
        print("Using existing experiment with ID:", experiment_id, '\n\n')
    mlflow.set_experiment(experiment_name)

    return experiment_id


def load_model_artifacts(logged_model: str) -> (dict, str):
    """
    Load model artifacts and settings.

    Parameters:
        logged_model (str): The logged model path for mlflow.

    Returns:
        args (dict): Arguments loaded from args.pth.
        setting (str): Settings loaded from setting.txt.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    artifacts_path = mlflow.artifacts.download_artifacts(logged_model, dst_path=script_dir)
    
    args_path = os.path.join(artifacts_path, 'informer-triton', 'args.pth')
    args = torch.load(args_path)
    
    settings_file_path = os.path.join(artifacts_path, 'informer-triton', 'setting.txt')
    with open(settings_file_path, 'r') as f:
        setting = f.read().strip()

    return args, setting
