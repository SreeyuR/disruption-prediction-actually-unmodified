import os
import tsaug
import numpy as np
import torch
import data_processing, dataset_types, evaluation
import copy



def augment_training_set(
        train_dataset,
        seq_to_seq,
        balance_positive_and_negative_cases,
        ratio_to_augment,
        data_augmentation_intensity, **kwargs):
    
    """Augment the training set by a tsaug pipeline.

    Args:
        train_dataset (object): Training data.
        seq_to_seq (bool): Whether to use seq_to_seq data.
        balance_positive_and_negative_cases (bool): Whether to balance the positive and negative
            cases in the training set.
        ratio_to_augment (int): Ratio of augmented data to original data.

    Returns:
        aug_train_dataset (object): Augmented training data."""
    
    my_augmenter = (
        tsaug.TimeWarp() * 2 @ (.2 * data_augmentation_intensity)  # time warping 5 times in parallel
        + tsaug.Drift(max_drift=(0.1, 0.3), n_drift_points=(20 * int(data_augmentation_intensity))) @ .3  # then add Gaussian drift
        + tsaug.Quantize(n_levels=[10, 20, 30], per_channel=True) @ .4 # then quantize time series into 10/20/30 levels
        + tsaug.Dropout(size=(5), p=.1, per_channel=True) @ .4  # then drop 10 values
        + tsaug.Dropout(size=(5), fill=0.0, p=.1, per_channel=True) @ .4  # then drop 10 values
        + tsaug.AddNoise(scale=1, per_channel=True) @ .4  # then add random noise
    )
    
    n_original = len(train_dataset)
    n_disruptions = train_dataset.num_disruptions
    n_augment = int(n_original * ratio_to_augment)
    num_disruptions_to_augment = int(n_augment/2)  - n_disruptions
    num_non_disruptions_to_augment = int(n_augment/2) - (n_original - n_disruptions)

    if balance_positive_and_negative_cases:
        ratio_augmented_disruptions = int(num_disruptions_to_augment / (n_disruptions * 2)) + 1 
    else:
        ratio_augmented_disruptions = 1 

    augmented_shots = {}
    augmented_shot_ind = 0

    for i in range(len(train_dataset)):
        lab = train_dataset[i]["labels"][-1] if seq_to_seq else train_dataset[i]["labels"][1]
        
        # now disruptions are coded as (machine_hyperparam["machine"])
        if lab > .5 :
            for j in range(ratio_augmented_disruptions):
                for k in (0, 1):
                    augmented_x = my_augmenter.augment(
                        train_dataset[i]["inputs_embeds"].unsqueeze(0).numpy())
                    augmented_shots[augmented_shot_ind] = {
                        "label": 1,
                        "data": augmented_x[k, :, :],
                        "machine": train_dataset[i]["machine"],
                        "shot": train_dataset[i]["shot"] + "_aug"}
                    augmented_shot_ind += 1
        
        # now non-disruptions are coded as (1 - machine_hyperparam["machine"])
        elif lab < .5:
            if num_non_disruptions_to_augment>0:
                for k in (0, 1):
                    augmented_x = my_augmenter.augment(
                        train_dataset[i]["inputs_embeds"].unsqueeze(0).numpy())
                    augmented_shots[augmented_shot_ind] = {
                        "label": 0,
                        "data": augmented_x[k, :, :],
                        "machine": train_dataset[i]["machine"],
                        "shot": train_dataset[i]["shot"] + "_aug"
                    }
                    augmented_shot_ind += 1
                    num_non_disruptions_to_augment -= 1

        else:
            raise ValueError("Invalid label value.")

    if seq_to_seq:
        DatasetPlaceholder = dataset_types.ModelReadyDatasetSeqtoSeqDisruption
    else:
        DatasetPlaceholder = dataset_types.ModelReadyDataset
    
    augmented_training_data = DatasetPlaceholder(
        shots=[augmented_shots[i] for i in range(augmented_shot_ind)],
        machine_hyperparameters=train_dataset.machine_hyperparameters,
        end_cutoff_timesteps=train_dataset.end_cutoff_timesteps,
        taus=train_dataset.taus,
        max_length=train_dataset.max_length,)
    
    # skip scaling for now...
    # augmented_training_data.variably_scale_with_another_scaler(train_dataset.scaler)

    final_training_data = train_dataset.concat(augmented_training_data)

    return final_training_data


def augment_data_windowing(
    train_dataset, ratio_to_augment,
    seq_to_seq, use_smoothed_tau,
    window_length, **kwargs):
    """Augment a sequence to label dataset by taking smaller
    windows of the data and labelling them according to tau.
    
    Args:
        train_dataset (object): Training data.
        ratio_to_augment (int): Ratio of augmented data to original data.
        seq_to_seq (bool): Whether to use seq_to_seq data.
        use_smoothed_tau (bool): Whether to use smoothed tau.
        window_length (int): Length of window to take in smoothing.
    
    Returns:
        aug_train_dataset (object): Augmented training data."""

    augmented_shots = {}
    augmented_shot_ind = 0

    taus = train_dataset.taus
    
    for j in range(len(train_dataset)):
        train_label = train_dataset[j]["labels"][-1] if seq_to_seq else train_dataset[j]["labels"][1]

        augmented_shots[augmented_shot_ind] = {
            "label": train_label,
            "data": train_dataset[j]["inputs_embeds"],
            "machine": train_dataset[j]["machine"],
            "shot": train_dataset[j]["shot"],
        }
        augmented_shot_ind += 1

        for i in range(ratio_to_augment):
            shot_length = len(train_dataset[j]["inputs_embeds"])
            
            # sample a window within 0 and len train_dataset[j]["inputs_embeds"]
            window_start = np.random.randint(0, shot_length)
            window_end = np.random.randint(window_start, shot_length)
            windowed_inputs_embeds = train_dataset[j]["inputs_embeds"][window_start:window_end, :]
            
            tau_value = taus[train_dataset[j]["machine"]]
            
            # if len(train_dataset[j] - tau_value is within the window, label as 1, else 0
            if train_label < .5:
                label = 0

            elif use_smoothed_tau:
                smoothed_curve = evaluation.produce_smoothed_curve(
                    shot_len=shot_length,
                    shot_tau=tau_value,
                    window_length=window_length
                )
                ind = min(window_end, shot_length - 1)
                label = smoothed_curve[ind]

            elif (shot_length - tau_value) < window_end:
                label = 1
            else:
                label = 0
            
            augmented_shots[augmented_shot_ind] = {
                "label": label,
                "data": windowed_inputs_embeds,
                "machine": train_dataset[j]["machine"],
                "shot": train_dataset[j]["shot"] + "_" + str(window_start) + "_to_" + str(window_end),
            }
            augmented_shot_ind += 1

    if seq_to_seq:
        DatasetPlaceholder = dataset_types.ModelReadyDatasetSeqtoSeqDisruption
    else:
        DatasetPlaceholder = dataset_types.ModelReadyDataset
        
    augmented_training_data = DatasetPlaceholder(
        shots=[augmented_shots[i] for i in range(augmented_shot_ind)],
        machine_hyperparameters=train_dataset.machine_hyperparameters,
        end_cutoff_timesteps=0,
        taus=train_dataset.taus,
        max_length=train_dataset.max_length,
        smooth_v_loop=train_dataset.smooth_v_loop,
        v_loop_smoother=train_dataset.v_loop_smoother,)

    return augmented_training_data


def restricted_data_augmentation_windowing(
    train_dataset,
    seq_to_seq, use_smoothed_tau,
    window_length, context_length):
    """Augment a sequence to label dataset by taking smaller
    windows of the data and labelling them according to tau.
    
    Args:
        train_dataset (object): Training data.
        seq_to_seq (bool): Whether to use seq_to_seq data.
        use_smoothed_tau (bool): Whether to use smoothed tau.
        window_length (int): Length of window to take in smoothing.
        context_length (int): Length of context to use in windowing the data.
    
    Returns:
        aug_train_dataset (object): Augmented training data."""

    augmented_shots = {}
    augmented_shot_ind = 0

    taus = train_dataset.taus
    
    for j in range(len(train_dataset)):
        train_label = train_dataset[j]["labels"][-1] if seq_to_seq else train_dataset[j]["labels"][1]

        shot_length = len(train_dataset[j]["inputs_embeds"])
        window_start = shot_length - context_length

        # if the context is larger than the shot_length, don't discard the shot
        if window_start < 0:
            augmented_shots[augmented_shot_ind] = {
                "label": train_label,
                "data": train_dataset[j]["inputs_embeds"],
                "machine": train_dataset[j]["machine"],
                "shot": train_dataset[j]["shot"],
            }
            augmented_shot_ind += 1
            continue

        while window_start >= 0:
            windowed_inputs_embeds = train_dataset[j]["inputs_embeds"][window_start:window_start+context_length, :]
            window_end = window_start + context_length

            tau_value = taus[train_dataset[j]["machine"]]
            
            # if len(train_dataset[j] - tau_value is within the window, label as 1, else 0
            if train_label < .5:
                label = 0

            elif use_smoothed_tau:
                smoothed_curve = evaluation.produce_smoothed_curve(
                    shot_len=shot_length,
                    shot_tau=tau_value,
                    window_length=window_length
                )
                ind = min(window_end, shot_length - 1)
                label = smoothed_curve[ind]

            elif (shot_length - tau_value) < window_end:
                label = 1
            else:
                label = 0
            
            augmented_shots[augmented_shot_ind] = {
                "label": label,
                "data": windowed_inputs_embeds,
                "machine": train_dataset[j]["machine"],
                "shot": train_dataset[j]["shot"] + "_" + str(window_start) + "_to_" + str(window_end),
            }
            augmented_shot_ind += 1

            window_start -= context_length

    if seq_to_seq:
        DatasetPlaceholder = dataset_types.ModelReadyDatasetSeqtoSeqDisruption
    else:
        DatasetPlaceholder = dataset_types.ModelReadyDataset
        
    augmented_training_data = DatasetPlaceholder(
        shots=[augmented_shots[i] for i in range(augmented_shot_ind)],
        machine_hyperparameters=train_dataset.machine_hyperparameters,
        end_cutoff_timesteps=0,
        taus=train_dataset.taus,
        max_length=train_dataset.max_length,)

    return augmented_training_data
