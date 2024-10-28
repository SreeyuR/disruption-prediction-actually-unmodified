import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
import sys, os
import pickle
import numpy as np
from sklearn.preprocessing import RobustScaler, StandardScaler
import random
import utils
import tsaug
import pandas as pd
import copy
from dataset_types import (
    ModelReadyDataset, ModelReadyDatasetSeqtoSeqDisruption,
    ModelReadyDatasetStatePretraining)
from dataset_augmentation_strategies import (
    augment_data_windowing, augment_training_set, restricted_data_augmentation_windowing,)
import dataset_augmentation_strategies

from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from numpy import dot
from numpy.linalg import norm
import evaluation
import math
import collections


import warnings
warnings.filterwarnings('once') 


def load_data(filename):
    """Load data from a pickle file.

    Args:
        filename (str): Path to the pickle file.

    Returns:
        data (object): Data loaded from the pickle file.
    """
    
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data


def fix_resample_data_issue(
        data,
        keep_jx_uneven_sampling,
        resample_rate=.005):
    """Fix the issue with resampling the data.

    Args:
        data (object): Data to fix.
        resample_rate (int): Rate at which to resample the data.

    Returns:
        data (object): Fixed data.
    """
    
    for i in range(len(data)):
        data[i]["data"].set_index("time", inplace=True)
        if data[i]["machine"] == "d3d":
            data[i]["data"] = data[i]["data"].resample(str(.025) + "S").ffill()

            if not keep_jx_uneven_sampling:
                data[i]["data"] = data[i]["data"].resample(str(resample_rate) + "S").ffill()

        elif data[i]["machine"] == "east":
            data[i]["data"] = data[i]["data"].resample(str(.1) + "S").ffill()

            if not keep_jx_uneven_sampling:
                data[i]["data"] = data[i]["data"].resample(str(resample_rate) + "S").ffill()

        data[i]["data"].reset_index(inplace=True)

    return data


def train_test_val_inds_from_file(
        case_number,
        testing,
        ):
    """Get train, test, and validation indices for a dataset. 
    Separating this function out now so that the same indices 
    can be used for all curriculum steps.

    Args:
        case_number (int): Case number.
        testing (bool): Whether to use a small subset of the data for testing.

    Returns:
        train_inds (list): List of training indices.
        test_inds (list): List of testing indices.
        val_inds (list): List of validation indices.
    """

    train_inds = pd.read_csv(f"train_inds_case{case_number}.csv")["dataset_index"].tolist()
    test_inds = pd.read_csv(f"holdout_inds_case{case_number}.csv")["dataset_index"].tolist()
    val_inds = pd.read_csv(f"val_inds_case{case_number}.csv")["dataset_index"].tolist()

    if testing:
        train_inds = train_inds[:50]
        test_inds = test_inds[:50]
        val_inds = val_inds[:50]

    return train_inds, test_inds, val_inds


def train_test_val_inds(
        dataset,
        new_machine,
        case_number,
        testing,
        causal_matching,
        distance_fn_name,
        taus,
        ):
    """Get train, test, and validation indices for a dataset. 
    Separating this function out now so that the same indices 
    can be used for all curriculum steps.

    Args:
        dataset (str): Name of the dataset.
        new_machine (str): Name of the machine.
        case_number (int): Case number.
        testing (bool): Whether to use a small subset of the data for testing.

    Returns:
        train_inds (list): List of training indices.
        test_inds (list): List of testing indices.
        val_inds (list): List of validation indices.
    """

    temp_train_inds, test_inds = get_train_test_indices_from_Jinxiang_cases(
        dataset=dataset,
        case_number=case_number,
        new_machine=new_machine,
        seed=42)
    
    np.random.shuffle(temp_train_inds)
    train_inds, val_inds = (
        temp_train_inds[:int(len(temp_train_inds) * .85)], 
        temp_train_inds[int(len(temp_train_inds) * .85):]
    )

    if testing:
        train_inds = train_inds[:50]
        test_inds = test_inds[:50]
        val_inds = val_inds[:50]

    return train_inds, test_inds, val_inds


def process_train_test_val_inds(
        dataset,
        train_inds,
        test_inds,
        val_inds,
        data_processing_args,
        **kwargs,):
    """Split dataset into training and testing sets.

    Args:
        dataset (object): Data to split.
        train_inds (list): List of training indices.
        test_inds (list): List of testing indices.
        val_inds (list): List of validation indices.
        end_cutoff (int): Number of shots to remove from the end of the dataset.
        machine_hyperparameters (dict): Dictionary of floats used to scale disruption labels.
        train_size (float): Proportion of data to use for testing.
        new_machine (str): Machine to use for testing.
        case_number (int): Case number to use for training set composition, see Xu, 2021.
        testing (bool): Whether to use a reduced sample for code testing.
        taus (dict): Number of timesteps to use for disruption prediction, separated by machine.
        seq_to_seq (bool): Whether to use a seq-to-seq model.
        data_augmentation (bool): Whether to use data augmentation.
        data_augmentation_windowing (bool): Whether to use data augment using windows of the data.
        data_augmentation_ratio (int): Factor by which to augment data.
        scaling_type (str): Type of scaling to use.
        balance_classes (bool): Whether to balance the classes.
        max_length (int): Maximum length of the dataset.

    Returns:
        train_dataset (object): Training data.
        test_dataset (object): Testing data.
    """
    
    if not utils.check_column_order([df["data"] for i, df in dataset.items()]):
        raise ValueError("Dataframe columns are out of order.")
    
    scale_labels = False
    dataset_type = data_processing_args["dataset_type"]
    data_augmentation = data_processing_args["data_augmentation"]
    data_augmentation_windowing = data_processing_args["data_augmentation_windowing"]

    if dataset_type == "state":
        DatasetPlaceholder = ModelReadyDatasetStatePretraining
        seq_to_seq = False
        scale_labels = True
    elif dataset_type == "seq_to_label":
        DatasetPlaceholder = ModelReadyDataset
        seq_to_seq = False
    elif dataset_type == "seq_to_seq":
        DatasetPlaceholder = ModelReadyDatasetSeqtoSeqDisruption
        seq_to_seq = True
    else:
        raise ValueError("Invalid dataset type.")

    train_dataset = DatasetPlaceholder(
        shots=[dataset[i] for i in train_inds],
        **data_processing_args)
    train_scaler = train_dataset.variably_scale(
        scaling_type=data_processing_args["scaling_type"],
        scale_labels=scale_labels)

    test_dataset = DatasetPlaceholder(
        shots=[dataset[i] for i in test_inds],
        **data_processing_args)
    test_dataset.variably_scale_with_another_scaler(
        train_scaler, scale_labels=scale_labels)
    test_dataset.move_data_to_device()

    val_dataset = DatasetPlaceholder(
        shots=[dataset[i] for i in val_inds],
        **data_processing_args)
    val_dataset.variably_scale_with_another_scaler(
        train_scaler, scale_labels=scale_labels)
    val_dataset.move_data_to_device()
                
    if data_augmentation_windowing and dataset_type != "state":
        train_dataset = augment_data_windowing(
            train_dataset,
            **data_processing_args,
        )
        val_dataset = augment_data_windowing(
            val_dataset,
            **data_processing_args,
        )
    
    if data_augmentation and dataset_type != "state":
        train_dataset = augment_training_set(
            train_dataset,
            seq_to_seq=seq_to_seq,
            **data_processing_args)

    train_dataset.move_data_to_device()

    return train_dataset, test_dataset, val_dataset


def collate_fn_seq_to_label(dataset, pad_from_left, truncate_longer):
    """
    Takes in an instance of Torch Dataset.
    Returns:
     * input_embeds: tensor, size: Batch x (padded) seq_length x embedding_dim
     * label_ids: tensor, size: Batch x 1 x 1 
    """

    output = {}

    if pad_from_left:
        output['inputs_embeds'] = pad_sequence(
            [df["inputs_embeds"].to(dtype=torch.float16) for df in dataset],
            padding_value=-100,
            batch_first=True)
        
        if truncate_longer:
            lens = [len(df["inputs_embeds"].to(dtype=torch.float16)) for df in dataset]
            min_len = min(lens)
            output['inputs_embeds'] = output['inputs_embeds'][:, :min_len, :]
    else:
        # Get sequences
        padded_sequences = pad_from_right(dataset, "inputs_embeds", truncate_longer)
        output['inputs_embeds'] = padded_sequences
        
    output['labels'] = torch.vstack(
        [df["labels"].to(torch.float32) for df in dataset])
    output["attention_mask"] = (output["inputs_embeds"][:, :, 1] != -100).to(torch.long)

    return output


def pad_from_right(batch_df, feature, truncate_longer):
    """Pad sequences starting from the left and heading right instead 
    of from the right and heading left. 
    
    Args:
        batch_df: list of pandas dataframes
        feature: str, name of feature to pad
     
    Returns:
        padded_sequences: tensor, size: Batch x (padded) seq_length x embedding_dim
    """
    
    sequences = [
        df[feature].to(dtype=torch.float16)
        for df in batch_df]
    sequences = [
        torch.flip(sequence, [0]) for sequence in sequences]
    padded_sequences = pad_sequence(
        sequences, padding_value=-100, batch_first=True)
    padded_sequences = torch.flip(padded_sequences, [1]) # along the time dim
    
    if truncate_longer:
        lens = [len(sequence) for sequence in sequences]
        min_len = min(lens)
        padded_sequences = padded_sequences[:, -min_len:, :]

    return padded_sequences


def collate_fn_seq_to_seq(dataset):
    """
    Takes in an instance of Torch Dataset and collates sequences for inputs and outputs.
    Returns:
     * input_embeds: tensor, size: Batch x (padded) seq_length x embedding_dim
     * label_ids: tensor, size: Batch x (padded) seq_length x 1 
    """

    output = {}

    output['inputs_embeds'] = pad_sequence(
        [df["inputs_embeds"].to(dtype=torch.float16) for df in dataset],
        padding_value=-100,
        batch_first=True)
    
    output['labels'] = pad_sequence(
        [df["labels"].to(dtype=torch.long) for df in dataset],
        padding_value=-100,
        batch_first=True)
    
    output["attention_mask"] = (output["inputs_embeds"][:, :, 1] != -100).to(torch.long)

    return output


def collate_fn_seq_to_seq_state(dataset):
    """
    Takes in an instance of Torch Dataset and collates sequences for inputs and outputs.
    Returns:
     * input_embeds: tensor, size: Batch x (padded) seq_length x embedding_dim
     * label_ids: tensor, size: Batch x (padded) seq_length x 1 
    """

    output = {}

    output['inputs_embeds'] = pad_sequence(
        [df["inputs_embeds"].to(dtype=torch.float16) for df in dataset],
        padding_value=-100,
        batch_first=True)
    
    output['labels'] = pad_sequence(
        [df["labels"].to(dtype=torch.float16) for df in dataset],
        padding_value=-100,
        batch_first=True)
    
    output["attention_mask"] = (output["inputs_embeds"][:, :, 1] != -100).to(torch.long)

    return output


Inds = collections.namedtuple("Inds", ["existing", "new", "disr", "nondisr"])

def get_index_sets(dataset, inds, new_machine):
    """Looks through each index given in the dataset and generates the following sets
        1. Existing machines: the indices of existing machines' shots
        2. New machine: the indices of the new machine's shots
        3. Disruptions: the indices of disruptions
        4. Non-disruptions: the indices of non-disruptions
    Args:
        dataset: The dataset
        inds: The indicies to look through in the dataset
    """
    existing_machines = {"cmod", "d3d", "east"}
    existing_machines.remove(new_machine)

    new, existing = set(), set()
    disr, nondisr = set(), set()
    for key in inds:
        v = dataset[key]
        if v["machine"] == new_machine:
            new.add(key)
        else:
            existing.add(key)

        if v["label"] == 0:
            nondisr.add(key)
        else:
            disr.add(key)

    assert len(existing & new) == 0, "Existing and new machines overlap"
    assert len(disr & nondisr) == 0, "Disruptions and non-disr overlap"

    assert len(existing | new) == len(inds)
    assert len(disr | nondisr) == len(inds)

    return Inds(existing, new, disr, nondisr)


def get_train_test_indices_from_Jinxiang_cases(
    dataset, case_number, new_machine, seed, test_percentage=0.15):
    """Get train and test indices for Jinxiang's cases.

    Args:
        dataset (object): Data to split.
        case_number (int): Case number.
        new_machine (str): Name of the new machine.

    Returns:
        train_inds (list): List of indices for the training set.
        test_indices (list): List of indices for the testing set.
    """

    rand = random.Random(seed)

    def take(inds, p=None, N=None):
        assert p or N
        N = math.ceil(p * len(inds)) if p else N
        assert N is not None and N <= len(inds)
        inds = list(inds)
        rand.shuffle(inds)
        return set(inds[:N])

    ix = get_index_sets(dataset, dataset.keys(), new_machine)
    existing, new, disr, non_disr = (
        ix.existing,
        ix.new,
        ix.disr,
        ix.nondisr,
    )

    test_inds = take(new, p=test_percentage)

    # remove test_inds from the other sets
    for s in [new, existing, disr, non_disr]:
        s.difference_update(test_inds)

    train_inds = set()
    if case_number == 1:
        train_inds = (existing & disr) | (new & non_disr) | (take(new & disr, N=20))
    elif case_number == 2:
        train_inds = (existing & disr) | (new & non_disr)
    elif case_number == 3:
        train_inds = (
            (existing & disr) | take(new & non_disr, p=0.5) | take(new & disr, N=20)
        )
    elif case_number == 4:
        train_inds = (new & non_disr) | take(new & disr, N=20)
    elif case_number == 5:
        train_inds = existing | (new & non_disr) | (take(new & disr, N=20))
    elif case_number == 6:
        train_inds = existing
    elif case_number == 7:
        train_inds = (existing & disr) | new
    elif case_number == 8:
        train_inds = existing | new
    elif case_number == 9:
        train_inds = new
    elif case_number == 10:
        train_inds = (existing & disr) | take(new & non_disr, p=0.33) | (new & disr)
    elif case_number == 11:
        train_inds = (
            take(existing & non_disr, p=0.2)
            | take(new & non_disr, p=0.33)
            | (new & disr)
        )
    elif case_number == 12:
        train_inds = take(new & non_disr, p=0.33) | (new & disr)
    elif case_number == 14:  # Will's case 14 where everything is a 12.5% split
        test_inds = take(dataset.keys(), p=0.125)
        train_inds = set(dataset.keys()) - test_inds
    else:
        raise ValueError(f"Case {case_number} not supported")

    assert len(test_inds & train_inds) == 0, "Test and train overlap"
    train_inds, test_inds = list(train_inds), list(test_inds)
    rand.shuffle(train_inds)
    rand.shuffle(test_inds)
    
    # Counting the number of disruptive shots in the train set
    train_disruptive_shots = sum(dataset[index]["label"] == 1 for index in train_inds)
    train_non_disruptive_shots = sum(dataset[index]["label"] == 0 for index in train_inds)
    print(f"Number of disruptive shots in the train set: {train_disruptive_shots}")
    print(f"Number of non-disruptive shots in the train set: {train_non_disruptive_shots}")

    # Counting the number of disruptive shots in the test set
    test_disruptive_shots = sum(dataset[index]["label"] == 1 for index in test_inds)
    test_non_disruptive_shots = sum(dataset[index]["label"] == 0 for index in test_inds)
    print(f"Number of disruptive shots in the test set: {test_disruptive_shots}")
    print(f"Number of non-disruptive shots in the test set: {test_non_disruptive_shots}")

    return train_inds, test_inds



def get_class_weights(train_dataset):
    """Get class weights for the training set.

    Args:
        train_dataset (object): Training set.

    Returns:
        class_weights (list): List of class weights.
    """
    class_counts = {}

    for i in range(len(train_dataset)):
        df = train_dataset[i]
        label = int(df['labels'][len(df['labels']) - 1])
        if label in class_counts.keys():
            class_counts[label] += 1
        else:
            class_counts[label] = 1
    class_weights = [class_counts[key]/sum(class_counts.values()) for key in class_counts.keys()]

    print("class weights: ")
    print(class_weights)

    return class_weights


def get_class_weights_seq_to_seq(train_dataset):
    """Get class weights for the training set.

    Args:
        train_dataset (object): Training set.

    Returns:
        class_weights (list): List of class weights.
    """
    class_counts = {0: 0, 1: 0}

    for i in range(len(train_dataset)):
        df = train_dataset[i]
        ones = torch.sum(df["labels"])
        zeros = len(df["labels"]) - ones
        class_counts[0] += zeros
        class_counts[1] += ones
    class_weights = [class_counts[key]/sum(class_counts.values()) for key in class_counts.keys()]

    print("class weights: ")
    print(class_weights)

    return class_weights


def get_class_weights_main(
        train_dataset,
        testing=False,
        multi_loss=True):
    """Get class weights for the training set.
    
    Args:
        train_dataset (object): Training set.
        testing (bool): Whether or not to use class weights for testing.
        multi_loss (bool): Whether or not to use class weights for multi-loss training.
        
    Returns:
        class_weights (list): List of class weights.
        """

    # if testing:
    #     class_weights = np.array([.5, .5])
    if multi_loss:
        class_weights = get_class_weights_seq_to_seq(train_dataset)
    else:
        class_weights = get_class_weights(train_dataset)
    return class_weights


def add_disruption_column(dataset, taus):
    """Add a disruption column to the dataset, marked by tau.
    
    Args:
        dataset (list): List of shots.
        taus (dict): Dictionary of tau values for each machine.
        
    Returns:
        dataset (list): List of shots with disruption column added.
    """

    for i, shot in dataset.items():
        tau_value = taus[shot['machine']]
        steps_to_mark = min(tau_value, len(shot['data']))
        shot['data']['disrupted'] = 0
        if shot["label"]:
            shot['data'].loc[len(shot['data']) - steps_to_mark:, 'disrupted'] = 1
    return dataset
    