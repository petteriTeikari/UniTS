import warnings
import numpy as np
from loguru import logger
import torch
from torch import nn
from sklearn.metrics import confusion_matrix

# from src.submodules.UniTS.utils.extra_eval.adjf1 import adjbestf1_with_threshold

from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
from tqdm import tqdm


def get_padding_indices(length_orig: int = 1981, length_padded: int = 2048):
    no_points_pad = length_padded - length_orig  # 67
    start_idx = no_points_pad // 2  # 33
    end_idx = start_idx + length_orig  # 2014
    return start_idx, end_idx


def unpad_glaucoma_PLR(array: np.ndarray, length_PLR: int = 1981):
    start_idx, end_idx = get_padding_indices(
        length_orig=length_PLR, length_padded=array.shape[1]
    )
    array_out = array[:, start_idx:end_idx]
    return array_out


def get_no_of_windows(length_PLR: int = 1981, window_size: int = 512):
    return np.ceil(length_PLR / window_size).astype(int)


def reshape_array(array, window_size: int = 500, length_PLR: int = 1981):
    """
    Reshape the array to the original shape (e.g. from (64,512) to (16,1981))
    array: np.array
        shape: (no_subjects, no_timepoints)
    """
    windows_per_subject = get_no_of_windows(
        length_PLR=length_PLR,
        window_size=window_size,
    )

    dim = len(array.shape)
    if dim == 3:
        array = np.squeeze(array)
        dim = len(array.shape)

    if dim == 2:
        array = np.reshape(
            array,
            (
                array.shape[0] // windows_per_subject,
                array.shape[1] * windows_per_subject,
            ),
        )
    elif dim == 1:
        no_subjects = int(array.shape[0] / windows_per_subject / window_size)
        array = np.reshape(array, (no_subjects, window_size * windows_per_subject))
    else:
        logger.error("Only 1D/2D Reshaping supported now, not {}".format(dim))
        raise ValueError("Only 1D/2D Reshaping supported now, not {}".format(dim))

    array = unpad_glaucoma_PLR(array, length_PLR=length_PLR)
    assert np.isnan(array).sum() == 0, "NaNs in the reshaped array"

    return array


def adjustment(gt, pred):
    anomaly_state = False
    for i in range(len(gt)):
        if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
            anomaly_state = True
            for j in range(i, 0, -1):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
            for j in range(i, len(gt)):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
        elif gt[i] == 0:
            anomaly_state = False
        if anomaly_state:
            pred[i] = 1
    return gt, pred


def get_scalar_outlier_metrics(preds: np.ndarray, gt: np.ndarray):
    metrics = {}

    if len(gt.shape) > 1:
        gt = gt.flatten()
        preds = preds.flatten()

    no_unique_gt = len(np.unique(gt))
    assert no_unique_gt == 2, "You have {} classes, should be binary now"

    no_unique_pred = len(np.unique(preds))
    assert no_unique_pred == 2, "You have {} classes, should be binary now"

    metrics["accuracy"] = accuracy_score(gt, preds)

    (
        metrics["precision"],
        metrics["recall"],
        metrics["f1"],
        metrics["support"],
    ) = precision_recall_fscore_support(gt, preds, average="binary")

    _, fp, _, _ = confusion_matrix(gt.flatten(), preds.flatten()).ravel()
    metrics["fp"] = float(fp) / len(gt.flatten())

    return metrics


def metrics_per_split(
    energy, labels, threshold, window_size: int = 500, length_PLR: int = 1981
):
    metrics_test = {"scalars": {}, "arrays": {}, "arrays_flat": {}}
    metrics_test["arrays"]["trues"] = labels

    pred = (energy > threshold).astype(int)
    metrics_test["arrays"]["pred_mask"] = pred

    pred = pred.flatten()
    labels = labels.flatten()
    gt = labels.astype(int)  # e.g. (32000,)
    assert pred.shape == gt.shape

    # (4) detection adjustment
    gt, pred = adjustment(gt, pred)
    pred = np.array(pred)
    gt = np.array(gt)

    # The vanilla metrics
    metrics_test["scalars"] = get_scalar_outlier_metrics(preds=pred, gt=gt)

    return metrics_test


def get_score(criterion, batch_x, outputs):
    score = torch.mean(criterion(batch_x, outputs), dim=-1)
    score = score.detach().cpu().numpy()
    return score


def forward_three_item_loader(loader, model, device, criterion):
    """
    args:
        batch_x: torch.Tensor
            expected size (batch_size, seq_len, num_features), e.g. (128,100,25) for the demo PSM dataset
        batch_y: torch.Tensor
            expected size (batch_size, seq_len, 1), e.g. (128,100,1) for the demo PSM dataset
    """
    attens_energy = []
    labels = []
    recon = []
    with torch.no_grad():
        for i, (batch_x, batch_y, _) in enumerate(loader):
            batch_x = batch_x.float().to(device)  # e.g (16,100)
            batch_x = batch_x.unsqueeze(2)  # e.g. (16,100,1)
            outputs = model(batch_x, None, None, None)  # e.g. (16,100,1)
            score = get_score(criterion, batch_x, outputs)

            attens_energy.append(score)
            recon.append(outputs.cpu().detach().numpy())
            labels.append(batch_y.cpu().detach().numpy())

    return attens_energy, recon, labels


def forward_two_item_loader(
    loader,
    model,
    device,
    criterion,
    task_id=0,
):
    """
    See test_anomaly_detection() in exp_sup.py
    """

    attens_energy = []
    labels = []
    recon = []
    with torch.no_grad():
        for i, (batch_x, batch_y) in enumerate(tqdm(loader, "attens_energy")):
            batch_x = batch_x.float().to(device)  # e.g. (256, 500, 1)
            # reconstruction
            outputs = model(
                batch_x,
                None,
                None,
                None,
                task_id=task_id,
                task_name="anomaly_detection",
            )
            # criterion
            score = get_score(criterion, batch_x, outputs)

            attens_energy.append(score)
            recon.append(outputs.cpu().detach().numpy())
            labels.append(batch_y.cpu().detach().numpy())

    return attens_energy, recon, labels


def reshape_arrays_to_input_lengths(
    attens_energy, labels, recon, length_PLR: int = 1981, window_size: int = 500
):
    attens = reshape_array(
        attens_energy, window_size=window_size, length_PLR=length_PLR
    )
    recon = reshape_array(recon, window_size=window_size, length_PLR=length_PLR)
    labels = reshape_array(array=labels, window_size=window_size, length_PLR=length_PLR)
    assert attens.shape == recon.shape == labels.shape
    return attens, labels, recon


def get_attens_energy(loader, model, device, criterion):
    example: list = next(iter(loader))
    no_of_items_in_batch = len(example)

    if no_of_items_in_batch == 3:
        # in the Foundation_PLR, there are batch_x, batch_y, and input mask
        # to cater originally for Momemnt's needs with the input mask being able to mask NaN values
        attens_energy, recon, labels = forward_three_item_loader(
            loader, model, device, criterion
        )
    elif no_of_items_in_batch == 2:
        # UniTS has only data and labels
        attens_energy, recon, labels = forward_two_item_loader(
            loader, model, device, criterion
        )
    else:
        logger.error(
            "No of items = {} in dataloader not supported".format(no_of_items_in_batch)
        )
        raise ValueError(
            "No of items = {} in dataloader not supported".format(no_of_items_in_batch)
        )

    attens_energy = np.concatenate(attens_energy, axis=0)
    recon = np.concatenate(recon, axis=0)  # [:,:,0]
    labels = np.concatenate(labels, axis=0)  # [:,:,0]
    attens_energy, labels, recon = reshape_arrays_to_input_lengths(
        attens_energy, labels, recon
    )

    outlier_percentage = 100 * (labels.sum() / labels.size)
    logger.info(f"Outliers masked {outlier_percentage:2f}%")

    return attens_energy, labels, recon


def combine_lists_of_ndarrays(list1: list, list2: list):
    def convert_list_to_array(list):
        return np.vstack(list).flatten()

    array1 = convert_list_to_array(list=list1)
    array2 = convert_list_to_array(list=list2)
    return np.concatenate((array1, array2), axis=0)


def convert_flat_to_2d_array(
    flat_arrays: dict, window_size: int = 500, length_PLR: int = 1981
):
    dict_out = {}
    for key, array in flat_arrays.items():
        dict_out[key] = reshape_array(
            array=array, window_size=window_size, length_PLR=length_PLR, dim=1
        )


def eval_outlier_detection(
    model,
    train_loader,
    test_loader,
    device_id: int = 0,
    device=None,
    task_id: int = None,
    features: str = "S",
    anomaly_ratio: float = 10,
    # these are now annoyingly hard-coded :(
    window_size: int = 500,
    length_PLR: int = 1981,
):
    # https://github.com/thuml/Time-Series-Library/blob/c80b851784184e088cc12afc9472fe55a06d6ada/exp/exp_anomaly_detection.py#L128
    model.eval()
    warnings.simplefilter("ignore")
    # size_average and reduce args will be deprecated, please use reduction='none' instead.
    criterion = nn.MSELoss(reduce=False)
    warnings.resetwarnings()
    metrics_test = {}

    # 1) static on the train set
    logger.info('Getting the train set "attens_energy"')
    train_energy, train_labels, train_recon = get_attens_energy(
        loader=train_loader, model=model, device=device, criterion=criterion
    )  # e.g. (32000,)

    # the test set
    logger.info('Getting the test set "attens_energy"')
    test_energy, test_labels, test_recon = get_attens_energy(
        loader=test_loader, model=model, device=device, criterion=criterion
    )  # e.g. (32000,)

    # 2) find the threshold
    combined_energy = np.concatenate([train_energy, test_energy], axis=0).flatten()
    combined_labels = np.concatenate([train_labels, test_labels], axis=0).flatten()

    # instead of a prior, use the actual anomaly ratio
    anomaly_ratio = 100 * (np.sum(combined_labels) / combined_labels.shape[0])  # 15.625
    threshold = np.percentile(
        combined_energy, 100 - anomaly_ratio
    )  # float threshold, e.g. 0.00389
    logger.info(
        f"Threshold = {threshold:.5f}, at anomaly_percentage = {anomaly_ratio:.2f}%"
    )

    # (3) evaluation on the splits
    logger.info("Getting the evaluation metrics")
    metrics_test["outlier_train"] = metrics_per_split(
        energy=train_energy, labels=train_labels, threshold=threshold
    )
    metrics_test["outlier_test"] = metrics_per_split(
        energy=test_energy, labels=test_labels, threshold=threshold
    )

    # add reconstructions to output dictionary
    metrics_test["outlier_test"]["arrays"]["preds"] = test_recon
    metrics_test["outlier_train"]["arrays"]["preds"] = train_recon

    return metrics_test
