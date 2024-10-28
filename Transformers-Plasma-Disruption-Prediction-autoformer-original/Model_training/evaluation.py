import evaluate
from transformers import Trainer
import torch
import torch.nn.functional as F
import numpy as np
from sklearn import metrics
import wandb
import matplotlib.pyplot as plt
import random
import seaborn as sns
import typing
import pandas as pd
import plotting
import copy
import constants
import dataset_types
import utils
import eval_class.model_eval as me

import warnings
warnings.filterwarnings('once') 

# Constants
COLUMN_NAMES = constants.COLUMN_NAMES
FEATURE_COLUMN_NAMES =  constants.FEATURE_COLUMN_NAMES


def compute_metrics_seq_to_seq(eval_pred):
    """Compute metrics for evaluation.

    Args:
        eval_pred (object): Evaluation predictions.

    Returns:
        metric (object): Metric for evaluation.
    """
    
    logits, labels = eval_pred
    labels = labels.squeeze()
    predictions = np.argmax(logits, axis=-1)

    # create a mask tensor to ignore the -100 tokens
    mask = labels != -100

    # compute the classification report on the valid tokens only
    report = metrics.classification_report(
        y_true=labels[mask],
        y_pred=predictions[mask],
        digits=4,
        output_dict=True,
        zero_division="warn"
    )

    # TODO: rerun with/without weighted avg!!!

    return {
        "f1": report["macro avg"]["f1-score"],
        "accuracy": report["accuracy"],
        "precision": report["macro avg"]["precision"],
        "recall": report["macro avg"]["recall"],
    }


def compute_metrics(eval_pred):
    """Compute metrics for evaluation.

    Args:
        eval_pred (object): Evaluation predictions.

    Returns:
        metric (object): Metric for evaluation.
    """
    
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    # check if labels are a vector of probabilities or just class labels
    if labels[0].size > 1:
        labels = np.argmax(labels, axis=-1)

    report = metrics.classification_report(
        y_true=labels,
        y_pred=predictions,
        digits=4,
        output_dict=True,
    )

    # metrics = evaluate.load("f1")
    # return metrics.compute(predictions=predictions, references=labels)

    return {
        "f1": report["macro avg"]["f1-score"],
        "accuracy": report["accuracy"],
        "precision": report["macro avg"]["precision"],
        "recall": report["macro avg"]["recall"],
    }


def compute_metrics_state_prediction(eval_pred):
    """Compute metrics for evaluation.

    Args:
        eval_pred (object): Evaluation predictions.

    Returns:
        metric (object): Metric for evaluation.
    """

    logits, labels = eval_pred
    predictions = logits[:, :-1, :]  # For regression tasks, the logits are the predictions
    
    mask = labels != -100

    # flatted the labels and predictions
    labels = labels[mask]
    predictions = predictions[mask]

    mse = metrics.mean_squared_error(y_true=labels, y_pred=predictions)
    mae = metrics.mean_absolute_error(y_true=labels, y_pred=predictions)
    r2 = metrics.r2_score(y_true=labels, y_pred=predictions)

    return {
        "mse": mse,
        "mae": mae,
        "r2": r2,
    }


def compute_metrics_after_training(y_true, y_pred, label="", seq_to_seq=False):
    """Compute metrics for evaluation after training is complete.

    Args:
        eval_pred (object): Evaluation predictions.

    Returns:
        metric (object): Metric for evaluation.
    """
    
    labels = y_true
    predictions = y_pred

    f1 = evaluate.load("f1")
    accuracy = evaluate.load("accuracy")
    precision = evaluate.load("precision")
    recall = evaluate.load("recall")

    return {
        ("final_f1" + label): f1.compute(predictions=predictions, references=labels),
        ("final_accuracy" + label): accuracy.compute(predictions=predictions, references=labels),
        ("final_precision" + label): precision.compute(predictions=predictions, references=labels),
        ("final_recall" + label): recall.compute(predictions=predictions, references=labels)
    }


def compute_auc(
        y_true,
        y_pred):
    """Compute the area under the curve (AUC) for a given set of predictions.
    
    Args:
        y_true (np.array): True labels.
        y_pred (np.array): Predicted labels.

    Returns:
        auc (float): Area under the curve (AUC).
        fpr (np.array): False positive rate array based on roc_curve computed thresholds.
        tpr (np.array): True positive rate based on roc_curve computed thresholds.
    """

    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred, pos_label=1)
    auc = metrics.auc(fpr, tpr)

    return auc, fpr, tpr


def draw_probabilities_from_trainer(trainer, test_dataset, seq_to_seq=False):
    """Draw predicted probabilities from the model.
    
    Args:
        trainer (object): Trainer object from huggingface.
        test_dataset (object): Test dataset, of ModelReadyDataset.
        seq_to_seq (bool): Whether the model is seq_to_seq or not.
    
    Returns:
        probs (np.array): Predicted probabilities.
        label_ids (np.array): True labels.
    """
    
    outputs = trainer.predict(test_dataset)
    true_labels = outputs.label_ids
    logits = outputs.predictions
    
    if seq_to_seq:
        probs = F.softmax(torch.tensor(logits), dim=2) 

        mask = (true_labels != -100)
        true_labels = true_labels[mask]
        
        # convert y_pred to np.array and add a dimension at the end
        probs = probs[:, : ,1]
        probs = probs[mask.squeeze(-1)]  # Change here
        probs = probs.tolist()

    else: 
        probs = F.softmax(torch.tensor(logits), dim=1)
        probs = probs[:, 1].tolist()
        true_labels = np.argmax(true_labels, axis=-1)
        
    return probs, true_labels  # Convert to numpy here


def draw_probabilities_seqseq_to_seqlab(outputs, threshold=.5):
    """Draw predicted probabilities from the model only counting when probability surpasses threshold.
    
    Args:
        outputs (object): Trainer object output from huggingface.
        threshold (float): Threshold for drawing labels.
        
    Returns:
        probs (np.array): Predicted probabilities."""
    
    true_labels = []
    predicted_1s= []

    for i in range(outputs.predictions.shape[0]):
        logit = outputs.predictions[i]
        probs = F.softmax(torch.tensor(logit), dim=1)[:, 1]
        labels = outputs.label_ids[i]
        labels = labels[labels>=0] # remove -100
        true_label = np.max(labels) # determine 0 or 1
        predicted_1 = int(np.sum(np.array(probs.tolist())>threshold) > 0) # if at least one prediction is above threshold
        true_labels.append(true_label.tolist())
        predicted_1s.append(predicted_1)
    
    return predicted_1s, true_labels
        

def evaluate_main(trainer, test_dataset, seq_to_seq=False):
    probs, true_labels = draw_probabilities_from_trainer(
        trainer, test_dataset, seq_to_seq=seq_to_seq)
    
    auc, fpr, tpr = compute_auc(
        y_pred=probs, y_true=true_labels)

    plot = plt.plot(fpr, tpr, label="Vanilla labelling ROC curve (area = %0.3f)" % auc)
   
    eval_metrics = compute_metrics_after_training(
        y_pred=probs, y_true=true_labels, seq_to_seq=seq_to_seq)
    
    if seq_to_seq:  
        outputs = trainer.predict(test_dataset)
        probs, true_labels = draw_probabilities_seqseq_to_seqlab(
            outputs, threshold=.5)
        
        eval_metrics = compute_metrics_after_training(
            y_pred=probs, y_true=true_labels, label="_thresh.5")
        wandb.log(eval_metrics)

        auc_1, fpr, tpr = compute_auc(y_pred=probs, y_true=true_labels)

        plot = plt.plot(fpr, tpr, label="Threshold .5 ROC curve (area = %0.3f)" % auc_1)
        
        # plot fpr and tpr as a line plot
        wandb.log({"roc": wandb.Image(plot[0]), "auc": auc})
        
        return
    
    # plot fpr and tpr as a line plot
    wandb.log({"roc": wandb.Image(plot[0]), "auc": auc})

    return


def prediction_time_from_end_positive(probs, threshold):
    """Compute the time from the end of the sequence to the first positive prediction.

    Args:
        probs (np.array): Predicted probabilities.
        threshold (float): Threshold for drawing labels.

    Returns:
        index (int): Index of the first positive prediction.
    """

    exceeds_threshold = (probs > threshold)

    # Find the first index where the condition is True
    index = np.argmax(exceeds_threshold).item()

    # If no probability is above the threshold, return the length of the sequence
    if index == 0 and not np.sum(exceeds_threshold[0]):
        index = probs.shape[0]

    return probs.shape[0] - index


def get_probs_from_seq_to_seq_model(shot, eval_model):
    """Get predicted probabilities from a seq_to_seq model.

    Args:
        shot (ModelReadyDatasetSeqtoSeqDisruption): Shot object.
        eval_model (object): Model to evaluate.

    Returns:
        probs (np.array): Predicted probabilities.
    """

    outputs = eval_model(inputs_embeds=shot["inputs_embeds"].unsqueeze(0).to(torch.float16))
    logits = outputs.logits
    probs = F.softmax(logits.cpu().detach().clone(), dim=2)
    probs = probs.numpy()[0, :, :]

    return probs


def get_probs_from_seq_to_lab_model(shot, eval_model):
    """Get predicted probabilities from a seq_to_lab model.
    
    Args:
        shot (ModelReadyDataset): Shot object.
        eval_model (object): Model to evaluate.
        
    Returns:
        probs (np.array): Predicted probabilities."""        
    
    probs = []
    for t in range(1, len(shot["inputs_embeds"])):
        input = shot["inputs_embeds"].unsqueeze(0)[:, :t, :].to(torch.float16)
        model_outputs = eval_model(
            inputs_embeds=input)
        model_output = model_outputs.logits
        prob = F.softmax(model_output, dim=-1).cpu().detach().numpy()
        probs.append(prob[0])

    probs = np.array(probs)
    return probs


def separate_d_and_nd_shots(val_dataset, seq_to_seq):
    """Separate disruptive and non-disruptive shots from the validation dataset.

    Args:
        val_dataset (ModelReadyDataset): Validation dataset, of ModelReadyDataset.
        seq_to_seq (bool): Whether the model is seq_to_seq or not.

    Returns:
        d_test_shots (ModelReadyDataset): Disruptive shots.
        nd_test_shots (ModelReadyDataset): Non-disruptive shots.
        nd_shot_inds (list): Indices of non-disruptive shots.
        d_shot_inds (list): Indices of disruptive shots.
        max_len (int): Maximum length of the sequences in the validation dataset.
    """

    d_shot_inds = []
    nd_shot_inds = []

    for i in range(len(val_dataset)):
        df = val_dataset[i]
        lab = np.array(df["labels"].cpu().tolist()[-1] if seq_to_seq else df["labels"].cpu()[1])
        if lab > .5:
            d_shot_inds.append(i)
        else:
            nd_shot_inds.append(i)

    max_len = np.max([df["inputs_embeds"].shape[0] for df in val_dataset])

    d_test_shots = val_dataset.subset(d_shot_inds)
    nd_test_shots = val_dataset.subset(nd_shot_inds)

    return d_test_shots, nd_test_shots, nd_shot_inds, d_shot_inds, max_len


def evaluate_main_seq(
        trainer,
        val_dataset,
        standardize_plot_length,
        threshold=.6,
        seq_to_seq=False,):
    """Evaluate custom sequence metrics for the model on the validation dataset.

    Args:
        trainer (object): Trainer object from huggingface.
        val_dataset (ModelReadyDataset): Validation dataset, of ModelReadyDataset.
        threshold (float): Threshold for drawing labels.
    """

    _, _, nd_shot_inds, d_shot_inds, max_len = separate_d_and_nd_shots(val_dataset, seq_to_seq)
    
    prediction_times_from_end = []
    areas_under_threshold = []
    eval_model = trainer.model.eval()

    # choose a few non_disruptive shots to plot

    sampled_nd_shot_inds = random.sample(nd_shot_inds, 5)
    sampled_d_shot_ins = random.sample(d_shot_inds, 5)        
    fig, ax = plotting.set_up_disruptivity_prediction_plot(
        probs_len=max_len, threshold=.5)    

    # loop through successively larger seq -> label windows
    for i in sampled_d_shot_ins:
        disruptive_shot = val_dataset[i]
        probs = get_probs_from_seq_to_lab_model(
            shot = disruptive_shot, eval_model = eval_model)[:, 1]
        probs = utils.moving_average_with_buffer(probs)

        if standardize_plot_length:
            probs = np.interp(
                np.linspace(0, 1, max_len),
                np.linspace(0, 1, len(probs)), probs)
        ax.plot(probs, "r-", alpha=.7)

        prediction_times_from_end.append(
            prediction_time_from_end_positive(probs, threshold))

    # plot non-disruptive shots
    for i in sampled_nd_shot_inds:
        non_disruptive_shot = val_dataset[i]
        nd_probs = get_probs_from_seq_to_lab_model(
            shot = non_disruptive_shot, eval_model = eval_model)[:, 1]
        nd_probs = utils.moving_average_with_buffer(nd_probs)

        if standardize_plot_length:
            nd_probs = np.interp(
                np.linspace(0, 1, max_len), np.linspace(0, 1, len(nd_probs)), nd_probs)
        ax.plot(nd_probs, "b-", alpha=.7)
        areas_under_threshold.append(np.mean(threshold - nd_probs))

    plt.legend(["Disruptive", "Non-disruptive"])
    wandb.log({("disruptivity_plot"): wandb.Image(fig)})
    plt.close()

    wandb.log({"area_under_threshold_mean": np.mean(areas_under_threshold), 
               "prediction_times_from_end_mean": np.mean(prediction_times_from_end)})
    
    return 


def distance_to_disruptivity_curve(
        eval_model, 
        taus,
        eval_dataset,
        seq_to_seq,
        window_length,
        mean_and_std):
    """Compute the distance to the disruptivity curve for a given model and taus.
    
    Args:
        eval_model (transformers.PreTrainedModel): Model to evaluate.
        taus (list): List of taus to evaluate.
        eval_dataset (Dataset): Test dataset.
        seq_to_seq (bool): Whether the model is seq_to_seq or not.
        window_length (int): Window length for smoothing.
        mean_and_std (bool): whether distance should be mean and std or just mean.
    """

    distances = []
    num_non_disruptions = 0

    for shot in eval_dataset:
        shot_len = shot["inputs_embeds"].shape[0]
        shot_tau = taus[shot["machine"]]
        is_disruptive = shot["labels"][1].cpu().detach().numpy() > .5

        if is_disruptive:
            smoothed_curve = produce_smoothed_curve(
                shot_len=shot_len, shot_tau=shot_tau,
                window_length=window_length)
        else:
            if num_non_disruptions >= eval_dataset.num_disruptions:
                continue
            smoothed_curve = np.zeros(shot_len - 1)
            num_non_disruptions += 1

        # make predictions 
        if seq_to_seq:
            logits = eval_model(
                inputs_embeds=shot["inputs_embeds"].unsqueeze(0).to(torch.float16)).logits
            probs = F.softmax(logits, dim=-1).cpu().detach().numpy()[0,:,1]
        else:
            probs = get_probs_from_seq_to_lab_model(shot=shot, eval_model=eval_model)[:, 1]
        
        # calculate distance
        if smoothed_curve.shape[0] > probs.shape[0]:
            smoothed_curve = smoothed_curve[:probs.shape[0]]
        elif smoothed_curve.shape[0] < probs.shape[0]:
            probs = probs[:smoothed_curve.shape[0]]
        distance = np.linalg.norm(probs - smoothed_curve)
        distances.append(distance)

    return np.mean(distances) + np.std(distances) if mean_and_std else np.mean(distances)


def produce_smoothed_curve(shot_len, shot_tau, window_length):
    """Produce a smoothed curve to match the disruptivity curve.

    Args:
        shot_len (int): Length of the shot.
        shot_tau (int): Tau of the shot.
        window_length (int): Window length for smoothing.

    Returns:
        smoothed_curve (np.array): Smoothed curve to match the 
            disruptivity curve.
    """

    curve_to_match = np.zeros(shot_len)
    curve_to_match[-shot_tau:] = 1

    curve_series = pd.Series(curve_to_match)

    # Calculate the moving average with a window size of 10
    smoothed_curve = curve_series.rolling(
        window=window_length,
        min_periods=1,
        center=True
    ).mean()

    return smoothed_curve


def compute_thresholded_statistics(
    test_unrolled_predictions, high_threshold, low_threshold,
    hysteresis
):
    """Compute statistics for thresholded disruptivity warnings. 
    
    In this case, if the model predicts a disruption above the high threshold,
    for a certain number of steps, it is considered a disruption. However, 
    if the prediction falls below the low threshold, the hysteresis counter is 
    reset. 
    
    Args:
        test_unrolled_predictions (dict): Dictionary of unrolled 
            predictions. Keys are "preds" with a list of predictions and 
            label with whether or not it disrupted.
        high_threshold (float): High threshold for disruption.
        low_threshold (float): Low threshold for disruption.
        
    Returns:
        None.
    """

    thresholded_preds = []
    thresholded_labels = []

    for _, val in test_unrolled_predictions.items():
        preds = val["preds"]
        label = val["label"]

        # initialize hysteresis counter
        hysteresis_counter = 0
        pred = 0

        for i in range(len(preds)):
            if preds[i] > high_threshold:
                hysteresis_counter += 1
            elif preds[i] < low_threshold:
                hysteresis_counter = 0

            if hysteresis_counter > hysteresis:
                pred = 1
                break
            
        thresholded_preds.append(pred)
        thresholded_labels.append(label)
    
    return
        

def create_eval_dict(test_dataset, trainer, master_dataset):
    """Save the unrolled probabilities into the eval class created by ENI.
    
    Args:
        test_dataset (AutoformerSeqtoLabelDataset): Test dataset.
        trainer (object): Trainer object from huggingface.
    """

    eval_dict = {}

    sampling_rates = {
        "cmod": .005,
        "east": .1,
        "d3d": .025
    }

    for shot in test_dataset:
        probs = get_probs_from_seq_to_lab_model(shot=shot, eval_model=trainer.model.eval().cpu())
        disruptivity = utils.moving_average_with_buffer(probs[:, 1])
        shot_num = shot["shot"]
        
        # master_ind = [i for i in range(len(master_dataset)) if master_dataset[i]["shot"]==int(shot_num)][0]
        # time = master_dataset[master_ind]["data"]["time"].seconds()
        
        time = np.array(list(range(len(disruptivity)))) * sampling_rates[shot["machine"]] 
        label = shot["labels"][1] > .5
        time_until_disrupt = [np.nan] * len(probs)
        if label:
            time_until_disrupt = max(time) - time
        
        m = shot["machine"]
        m = "D3D" if m == "d3d" else m
        
        eval_dict[f"{m}_{shot_num}"] = {
            "proba_shot": disruptivity,
            "time_untill_disrupt": time_until_disrupt,
            "time_shot": time,
            "label_shot": label,
        }
    
    return eval_dict


def return_eval_params():
    # Necessary inputs
    params_dict = {
        'high_thr':.5,
        'low_thr':.5,
        't_hysteresis':0,
        't_useful':.03
        }

    metrics = [
        'f1_score', 
        'f2_score', 
        'recall_score', 
        'precision_score', 
        'roc_auc_score', 
        'accuracy_score', 
        'confusion_matrix', 
        'tpr', 
        'fpr', 
        'AUC_zhu']
    
    return params_dict, metrics
        



 


