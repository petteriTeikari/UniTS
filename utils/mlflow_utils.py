import argparse
import os
import pickle

import mlflow
import numpy as np


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


def mlflow_log_metrics(metrics_dict: dict):

    for split in metrics_dict.keys():
        split_out = split # split.replace('outlier_', '')
        scalars = metrics_dict[split]['scalars']['global']
        for key, value in scalars.items():
            if value is not None:
                if isinstance(value, np.ndarray):
                    mlflow.log_metric(f'{split_out}/{key}_lo', value[0])
                    mlflow.log_metric(f'{split_out}/{key}_hi', value[1])
                else:
                    mlflow.log_metric(f'{split_out}/{key}', value)


def mlflow_log_model(checkpoint_path: str):
    mlflow.log_artifact(checkpoint_path, artifact_path='model')


def mlflow_log_results(artifacts, checkpoint_path: str, args: argparse.Namespace):

    base_dir = os.path.dirname(checkpoint_path)
    model_name = args.mlflow_run
    results_path = os.path.join(base_dir, f"outlierDetection_{model_name}.pickle")
    with open(results_path, "wb") as handle:
        pickle.dump(artifacts, handle, protocol=pickle.HIGHEST_PROTOCOL)
    mlflow.log_artifact(results_path, artifact_path='outlier_detection')


def mlflow_log_experiment(epoch_losses: list,
                          artifacts: dict,
                          checkpoint_path: str,
                          args):

    mlflow_log_metrics(metrics_dict=artifacts['results'])
    if epoch_losses is not None:
        # None when training was not on, as in this was a zeroshot, the weights did not change
        mlflow_log_model(checkpoint_path=checkpoint_path)
    mlflow_log_results(artifacts, checkpoint_path=checkpoint_path, args=args)

    # Best train loss
    if epoch_losses is not None:
        loss = epoch_losses[artifacts['best_epoch']]
        mlflow.log_metric('train/loss', loss)

    # copy the data .yaml to MLflow, make the path absolute before
    data_yaml = os.path.abspath(args.task_data_config_path)
    if os.path.exists(data_yaml):
        mlflow.log_artifact(data_yaml, artifact_path='data_provider')

