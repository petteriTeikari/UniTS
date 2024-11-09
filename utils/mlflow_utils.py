import os
import mlflow


def init_mlflow(tracking_uri: str):
    abs_path = os.path.abspath(os.path.join(os.getcwd(), '..', '..', tracking_uri))
    mlflow.set_tracking_uri(abs_path)
    print('Init ML | tracking_uri = {}'.format(abs_path))


def init_mlflow_experiment(experiment_name: str):
    mlflow.set_experiment(experiment_name)


def log_mlflow_params(args):
    print('Input arguments:')
    for key, value in vars(args).items():
        print(f' {key}: {value}')
        mlflow.log_param(key, value)


def mlflow_log_experiment(epoch_losses: list, avg_adj_f1, checkpoint_path: str):

    a = 1