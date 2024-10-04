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
import constants

import warnings
warnings.filterwarnings('once') 


class ModelReadyDataset(Dataset):
    """Torch Dataset for model-ready data.
    
    Args:
        shots (list): List of shots.
        max_length (int): Maximum length of the input sequence.
        
    Attributes:
        input_embeds (list): List of input embeddings.
        labels (list): List of labels.
        shot_inds (list) = List of shot inds  
        shot_lens (list): List of shot lengths.
    """

    def __init__(
            self,
            shots,
            end_cutoff_timesteps,
            machine_hyperparameters,
            taus,
            max_length,
            smooth_v_loop,
            v_loop_smoother,
            shot_num_addendum = "",
            **kwargs,):
        self.inputs_embeds = []
        self.labels = []
        self.shot_inds = []
        self.num_disruptions = 0
        self.machines = []
        self.end_cutoff_timesteps = end_cutoff_timesteps
        self.machine_hyperparameters = machine_hyperparameters
        self.max_length = max_length
        self.smooth_v_loop = smooth_v_loop
        self.v_loop_smoother = v_loop_smoother
        self.taus = taus
        self.shots = []

        for shot in shots:

            # test if the shot's length is between 10 and max_length

            if 22 >= len(shot["data"]) or len(shot["data"]) >= max_length:
                continue

            shot_df = copy.deepcopy(shot['data'])
            s = str(shot["shot"])

            if isinstance(shot_df, pd.DataFrame):
                shot_df.drop(columns=["time"], inplace=True)

                # set smooth_v_loop the first time processing. 
                if smooth_v_loop:
                    shot_df["v_loop"] = shot_df["v_loop"].rolling(v_loop_smoother).mean()
                    first_valid_index = shot_df["v_loop"].first_valid_index()
                    first_valid_mean = shot_df["v_loop"].loc[first_valid_index]
                    shot_df["v_loop"].fillna(first_valid_mean, inplace=True)

            # set machine hyperparameter scaling of 0,1 labels.
            ps = machine_hyperparameters[shot['machine']]
            if 0 < shot["label"] < 1:
                l = max(min(shot["label"], ps[1]), ps[0])
                o = torch.tensor([1 - l, l], dtype=torch.float32)
            else:
                is_disruptive = int(shot['label'])
                o = torch.tensor([ps[1 - is_disruptive], ps[is_disruptive]], dtype=torch.float32)

            shot_end = int(len(shot_df) - end_cutoff_timesteps)
            d = torch.tensor(shot_df[:shot_end].values if isinstance(shot_df, pd.DataFrame) else shot_df[:shot_end], dtype=torch.float32)

            self.inputs_embeds.append(d)
            self.labels.append(o)
            self.num_disruptions += shot["label"]
            self.machines.append(shot['machine'])
            self.shots.append(s)

            
    def check_scaled_data(self, df_scaled, threshold_std=3.0):
        """Check if the data is scaled properly.

        Args:
            df_scaled (numpy array): Scaled data.
            threshold_std (float): Threshold for standard deviation.
        """

        df_scaled = pd.DataFrame(df_scaled)
        stds = df_scaled.std(axis=0)
        for column, std in zip(df_scaled.columns, stds):
            if column == 0:
                print("--------------------")
                print("Checking if data is scaled properly...")
            if std > threshold_std:
                print(f'Column {column} has high standard deviation after scaling: {std}')
        print("--------------------")


    def variably_scale(self, scaling_type='standard', scale_labels=False):
        """Robustly scale the data.
        
        Returns:
            scaler (object): Scaler used to scale the data."""

        if scaling_type == 'robust':
            scaler = RobustScaler()
        elif scaling_type == 'standard':
            scaler = StandardScaler()
        else:
            raise ValueError(f'Invalid scaling type: {scaling_type}')
        
        combined = torch.cat(self.inputs_embeds)
        scaler.fit(combined)

        for i in range(len(self.inputs_embeds)):
            self.inputs_embeds[i] = torch.tensor(
                scaler.transform(self.inputs_embeds[i]), dtype=torch.float32)

        if scale_labels:
            for i in range(len(self.labels)):
                self.labels[i] = torch.tensor(
                    scaler.transform(self.labels[i]), dtype=torch.float32)

        # Usage:
        # Assume df_scaled is your DataFrame after scaling
        self.check_scaled_data(
            torch.cat(self.inputs_embeds).numpy(),
            threshold_std=1.5)
        
        return scaler    
    
    def variably_scale_with_another_scaler(self, scaler, scale_labels=False):
        """Robustly scale the data with another scaler.
        
        Args:
            scaler (object): Scaler to use to scale the data.
        """
        
        for i in range(len(self.inputs_embeds)):
            self.inputs_embeds[i] = torch.tensor(
                scaler.transform(self.inputs_embeds[i]))
            
        if scale_labels:
            for i in range(len(self.labels)):
                self.labels[i] = torch.tensor(
                    scaler.transform(self.labels[i]))
            
        self.check_scaled_data(
            torch.cat(self.inputs_embeds).numpy(),
            threshold_std=1
            )
        return
    
    def move_data_to_device(self):
        """Move data to device.
        
        Args:
            device (object): Device to move data to.
        """

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        for i in range(len(self.inputs_embeds)):
            self.inputs_embeds[i] = self.inputs_embeds[i].to(device)
            self.labels[i] = self.labels[i].to(device)
        
        return
    

    def subset(self, indices):
        """Subset the dataset.

        Args:
            indices (list): List of indices to subset the dataset with.

        Returns:
            subset (ModelReadyDataset): Subset of the dataset."""
        
        # if the type of index is an int, convert it to a list
        if isinstance(indices, int):
            indices = [indices]
        
        subset_inputs_embeds = [self.inputs_embeds[i] for i in indices]
        subset_labels = [self.labels[i] for i in indices]
        subset_machines = [self.machines[i] for i in indices]
        subset_shots = [self.shots[i] for i in indices]

        # Create a new instance of this class with the subset of data
        subset = ModelReadyDataset(
            [],
            end_cutoff_timesteps=self.end_cutoff_timesteps,
            machine_hyperparameters=self.machine_hyperparameters,
            taus=self.taus,
            max_length=self.max_length,
            smooth_v_loop=self.smooth_v_loop,
            v_loop_smoother=self.v_loop_smoother)
        subset.inputs_embeds = subset_inputs_embeds
        subset.labels = subset_labels
        subset.machines = subset_machines
        subset.shots = subset_shots

        return subset
    
    def concat(self, new_dataset):
        """Concatenate this dataset with another dataset.
        
        Args:
            new_dataset (ModelReadyDataset): Dataset to concatenate with.
            
        Returns:
            concat_dataset (ModelReadyDataset): Concatenated dataset."""
    
        concat_dataset = ModelReadyDataset(
            [],
            end_cutoff_timesteps=self.end_cutoff_timesteps,
            machine_hyperparameters=self.machine_hyperparameters,
            taus=self.taus,
            max_length=self.max_length,
            smooth_v_loop=self.smooth_v_loop,
            v_loop_smoother=self.v_loop_smoother)
        concat_dataset.inputs_embeds = self.inputs_embeds + new_dataset.inputs_embeds
        concat_dataset.labels = self.labels + new_dataset.labels
        concat_dataset.machines = self.machines + new_dataset.machines
        concat_dataset.shots = self.shots + new_dataset.shots
        return concat_dataset
    
    def set_shot_indices(self, shot_inds):
        self.shot_inds = shot_inds

    def __len__(self):
        return len(self.inputs_embeds)

    def __getitem__(self, idx):
        return {
            'inputs_embeds': self.inputs_embeds[idx],
            'labels': self.labels[idx],
            'machine': self.machines[idx],
            "shot": self.shots[idx],
        }
    

class ModelReadyDatasetSeqtoSeqDisruption(ModelReadyDataset):
    """Torch Dataset for model-ready data.
    
    Args:
        shots (list): List of shots.
        max_length (int): Maximum length of the input sequence.
        end_cutoff (float): Fraction of the shot to use as the end of the shot.
        end_cutoff_timesteps (int): Number of timesteps to use as the end of the shot.
        machine_hyperparameters (dict): Dictionary of the machine smoothing hyperparameters.
        tau (float): Tau value for prediction window cutoff.
        
    Attributes:
        input_embeds (list): List of input embeddings.
        labels (list): List of labels.
        shot_inds (list) = List of shot inds  
        shot_lens (list): List of shot lengths."""
        
    def __init__(
            self,
            shots,
            end_cutoff_timesteps,
            machine_hyperparameters,
            taus,
            max_length,
            **kwargs,):
        self.inputs_embeds = []
        self.labels = []
        self.shot_inds = []
        self.end_cutoff_timesteps = end_cutoff_timesteps
        self.machine_hyperparameters = machine_hyperparameters
        self.taus = taus
        self.max_length = max_length
        self.num_disruptions = 0
        self.num_disruptive_values = 0
        self.num_disruptive_values_list = []
        self.machines = []
        self.shots = []

        for i, shot in enumerate(shots):
            shot_df = copy.deepcopy(shot['data'])
            s = shot["shot"]

            shot_end = int(len(shot_df) - end_cutoff_timesteps)

            # test if shot_end is between 10 and max_length
            if not 10 <= shot_end < max_length:
                continue

            # set machine hyperparameter scaling of 0,1 labels.
            p = machine_hyperparameters[shot['machine']]
            o_fill = torch.tensor(p[0], dtype=torch.float32)
            o = torch.repeat_interleave(o_fill, shot_end)

            if shot['label']:
                tau_value = taus[shot['machine']]
                steps_to_mark = min(tau_value, shot_end)
                
                # Mark the last 'steps_to_mark' steps as 1
                o[-steps_to_mark:] = p[1]
                self.num_disruptions += 1
                self.num_disruptive_values += steps_to_mark
                self.num_disruptive_values_list.append(steps_to_mark)
            else:
                self.num_disruptive_values_list.append(0)

            o = o.unsqueeze(-1)

            if isinstance(shot_df, pd.DataFrame):
                d = torch.tensor(shot_df[:shot_end].values, dtype=torch.float32)
            else:
                d = shot_df[:shot_end].clone().detach()

            self.inputs_embeds.append(d)
            self.labels.append(o)
            self.machines.append(shot['machine'])
            self.shots.append(s)


    def subset(self, indices):
        """Subset the dataset.

        Args:
            indices (list): List of indices to subset the dataset with.

        Returns:
            subset (ModelReadyDataset): Subset of the dataset."""
        
        # if the type of index is an int, convert it to a list
        if isinstance(indices, int):
            indices = [indices]
        
        subset_inputs_embeds = [self.inputs_embeds[i] for i in indices]
        subset_labels = [self.labels[i] for i in indices]
        subset_machines = [self.machines[i] for i in indices]
        subset_disruptive_values_list = [self.num_disruptive_values_list[i] for i in indices]
        subset_shots = [self.shots[i] for i in indices]

        # Create a new instance of this class with the subset of data
        subset = ModelReadyDatasetSeqtoSeqDisruption(
            [],
            end_cutoff_timesteps=self.end_cutoff_timesteps,
            machine_hyperparameters=self.machine_hyperparameters,
            taus=self.taus,
            max_length=self.max_length)
        
        subset.inputs_embeds = subset_inputs_embeds
        subset.labels = subset_labels
        subset.machines = subset_machines
        subset.shots = subset_shots
        subset.num_disruptive_values = sum(subset_disruptive_values_list)
        subset.num_disruptive_values_list = subset_disruptive_values_list

        return subset
    
    def concat(self, new_dataset):
        """Concatenate this dataset with another dataset.
        
        Args:
            new_dataset (ModelReadyDataset): Dataset to concatenate with.
            
        Returns:
            concat_dataset (ModelReadyDataset): Concatenated dataset."""
    
        concat_dataset = ModelReadyDatasetSeqtoSeqDisruption(
            [],
            end_cutoff_timesteps=self.end_cutoff_timesteps,
            machine_hyperparameters=self.machine_hyperparameters,
            taus=self.taus,
            max_length=self.max_length)
        concat_dataset.inputs_embeds = self.inputs_embeds + new_dataset.inputs_embeds
        concat_dataset.labels = self.labels + new_dataset.labels
        concat_dataset.machines = self.machines + new_dataset.machines
        concat_dataset.num_disruptive_values = self.num_disruptive_values + new_dataset.num_disruptive_values
        concat_dataset.num_disruptive_values_list = self.num_disruptive_values_list + new_dataset.num_disruptive_values_list
        concat_dataset.shots = self.shots + new_dataset.shots

        return concat_dataset

class ModelReadyDatasetStatePretraining(ModelReadyDataset):
    def __init__(
        self,
        shots,
        end_cutoff_timesteps,
        machine_hyperparameters,
        max_length,
        taus,
        **kwargs,):

        self.inputs_embeds = []
        self.labels = []
        self.shot_inds = []
        self.end_cutoff_timesteps = end_cutoff_timesteps
        self.machine_hyperparameters = machine_hyperparameters
        self.taus = taus
        self.shots = []

        self.max_length = max_length
        self.num_disruptions = 0
        self.machines = []
        self.is_disruptive = []

        for i, shot in enumerate(shots):
            shot_df = copy.deepcopy(shot['data'])

            if isinstance(shot_df, pd.DataFrame):
                shot_df.drop(columns=["time"], inplace=True)
            
            shot_end = int(len(shot_df) - end_cutoff_timesteps)

            # test if shot_end is between 10 and max_length
            if not 10 <= shot_end < max_length:
                continue

            if isinstance(shot_df, pd.DataFrame):
                d = torch.tensor(shot_df[:shot_end].values, dtype=torch.float32)
                d_target = torch.tensor(shot_df[1:shot_end].values, dtype=torch.float32)
            elif isinstance(shot_df, np.array):
                d = torch.tensor(shot_df[:shot_end], dtype=torch.float32)
                d_target = torch.tensor(shot_df[1:shot_end+1], dtype=torch.float32)
            elif isinstance(shot_df, torch.tensor):
                d = shot_df[:shot_end].clone().detach()
                d_target = shot_df[1:shot_end].clone().detach()
            else:
                raise ValueError('Unknown data type for shot_df')

            self.inputs_embeds.append(d)
            self.labels.append(d_target)
            self.machines.append(shot['machine'])
            self.is_disruptive.append(int(np.round(shot['label'])))
            self.shots.append(shot['shot'])

    def subselect_feature_columns(self):
        self.labels = [x[:, [0, 1, 2, 4, 8, 9, 10, 11, 12]] for x in self.labels]


class AutoformerReadySeqToLabel(Dataset):
    """Torch Dataset for model-ready data.
    
    Args:
        shots (list): List of shots.
        max_length (int): Maximum length of the input sequence.
        
    Attributes:
        input_embeds (list): List of input embeddings.
        labels (list): List of labels.
        shot_inds (list) = List of shot inds  
        shot_lens (list): List of shot lengths.
    """

    def __init__(
            self,
            shots,
            end_cutoff_timesteps,
            max_length,
            use_smoothed_tau,
            context_length,
            max_lagged_sequence,
            window_length,
            prediction_length,
            shot_num_addendum = "",
            **kwargs,):
        self.past_values = []
        self.future_values = []
        self.past_time_features = []
        self.future_time_features = []
        self.static_categorical_features = []
        self.values = []
        self.shot_inds = []
        self.num_disruptions = 0
        self.end_cutoff_timesteps = end_cutoff_timesteps
        self.max_length = max_length
        self.shots = []
        self.context_length = context_length
        self.max_lagged_sequence = max_lagged_sequence
        self.is_disruption = []

        for shot in shots:
            shot_df = copy.deepcopy(shot['data'])
            s = str(shot["shot"]) + shot_num_addendum
            shot_end = int(len(shot_df) - end_cutoff_timesteps)
            shot_df = shot_df[:shot_end]

            autoformer_data_size =  context_length + max_lagged_sequence + prediction_length
            past_future_cutoff = autoformer_data_size - prediction_length

            if int(len(shot_df)) < autoformer_data_size:
                continue

            if use_smoothed_tau:
                # take the moving average of train_dataset[j]["data"]["disrupted"]
                curve_series = pd.Series(shot_df["disrupted"])

                # Calculate the moving average with a window size of 10
                smoothed_curve = curve_series.rolling(
                    window=window_length,
                    min_periods=1,
                    center=True
                ).mean()
                smoothed_curve.fillna("ffill", inplace=True)
                smoothed_curve.fillna(value=0, inplace=True)
                shot_df["disrupted"] = smoothed_curve

            time_features = shot_df["time"].values

            # cast time_features from a pd.TimeDelta into a float for seconds
            if isinstance(time_features, pd.Timedelta) or np.issubdtype(time_features.dtype, np.timedelta64):
                time_features = time_features.astype('timedelta64[ms]').astype(np.float32) * .001
                time_features = np.round(time_features, 5)
            
            machine_indicators = shot_df[["cmod", "d3d", "east"]].iloc[0].values

            machine_indicator = get_machine_index(shot_df)

            shot_df.drop(columns=["time", "cmod", "d3d", "east"], inplace=True)
            all_values = shot_df.values[-autoformer_data_size:]

            # break them out
            all_values = cast_tensors_well(all_values)  
            past_values = cast_tensors_well(
                all_values).squeeze()[:past_future_cutoff].squeeze()
            past_values[:, 9] = 0 # set the disruption label to 0
            past_time_features = cast_tensors_well(
                time_features[:past_future_cutoff]) # because it's just 1d
            static_categorical_variables = cast_tensors_well(
                machine_indicator, t=torch.long)

            future_values = cast_tensors_well(
                all_values[past_future_cutoff:autoformer_data_size]).squeeze()
            future_time_features = cast_tensors_well(
                time_features[past_future_cutoff:autoformer_data_size])

            # test if the shot's length is between 10 and max_length
            self.past_values.append(past_values)
            self.past_time_features.append(past_time_features)
            self.future_values.append(future_values)
            self.future_time_features.append(future_time_features)
            self.static_categorical_features.append(static_categorical_variables)
            self.shots.append(s)
            self.values.append(all_values)
            self.is_disruption.append(int(np.round(shot['label'])))
            self.num_disruptions += int(np.round(shot['label']))
        
    def __len__(self):
        return len(self.past_values)
    
    def __getitem__(self, idx):
        return {
            "past_values": self.past_values[idx],
            "past_time_features": self.past_time_features[idx].unsqueeze(-1),
            "future_values": self.future_values[idx],
            "future_time_features": self.future_time_features[idx].unsqueeze(-1),
            "static_categorical_features": self.static_categorical_features[idx],
            "is_disruption": self.is_disruption[idx],
        }
    

    def variably_scale(self, scaling_type='standard', scale_labels=False):
        """Robustly scale the data.
        
        Returns:
            scaler (object): Scaler used to scale the data."""

        if scaling_type == 'robust':
            scaler = RobustScaler()
        elif scaling_type == 'standard':
            scaler = StandardScaler()
        else:
            raise ValueError(f'Invalid scaling type: {scaling_type}')
        
        combined = torch.cat(self.values)
        scaler.fit(combined)

        for i in range(len(self.past_values)):
            self.past_values[i] = torch.tensor(
                scaler.transform(self.past_values[i]), dtype=torch.float32)

        for i in range(len(self.future_values)):
            self.future_values[i] = torch.tensor(
                scaler.transform(self.future_values[i]), dtype=torch.float32)
        
        return scaler    
    
    def variably_scale_with_another_scaler(self, scaler, scale_labels=False):
        """Robustly scale the data with another scaler.
        
        Args:
            scaler (object): Scaler to use to scale the data.
        """
        
        for i in range(len(self.past_values)):
            self.past_values[i] = torch.tensor(
                scaler.transform(self.past_values[i]))

        for i in range(len(self.future_values)):
            self.future_values[i] = torch.tensor(
                scaler.transform(self.future_values[i]))
        return
    
    def move_data_to_device(self):
        """Move data to device.
        
        Args:
            device (object): Device to move data to.
        """

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        for i in range(len(self.past_values)):
            self.past_values[i] = self.past_values[i].to(device)
            self.future_values[i] = self.future_values[i].to(device)
            self.past_time_features[i] = self.past_time_features[i].to(device)
            self.future_time_features[i] = self.future_time_features[i].to(device)
        return


def cast_tensors_well(shot_df, t=torch.float32):
    """Cast the shot data to a tensor. Handles several datatypes.
    
    Args:
        shot_df (pandas DataFrame): DataFrame containing the shot data.
        t (torch datatype): Datatype to cast the shot data to.
    
    Returns:
        d (torch tensor): Tensor containing the shot data.
    """

    if isinstance(shot_df, pd.DataFrame):
        d = torch.tensor(shot_df.values, dtype=t)
    elif isinstance(shot_df, np.ndarray):
        d = torch.tensor(shot_df, dtype=t)
    elif isinstance(shot_df, torch.Tensor):
        d = shot_df.clone().detach()
    else:
        raise ValueError('Unknown data type for shot_df')
    return d


def get_machine_index(shot_df):
    """Get the machine index for the shot.
    
    Args:
        shot_df (pandas DataFrame): DataFrame containing the shot data.
        
    Returns:
        machine_indicator (numpy array): Machine indicator for the shot.
    """

    machine_weights = {
    'cmod': 0,
    'd3d': 1,
    'east': 2
}

    machine_indicator = np.array(
        [sum([shot_df[machine].iloc[0] * weight 
            for machine, weight in machine_weights.items()])])
    
    return machine_indicator