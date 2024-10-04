import os

from transformers import TrainingArguments, Trainer, EarlyStoppingCallback
from transformers.utils import logging as transformers_logging
from transformers.trainer_callback import DefaultFlowCallback, ProgressCallback

import torch
from torch.utils.data import DataLoader
import wandb
import numpy as np

# local imports
from custom_callbacks import AllWandbCallback
import custom_callbacks
from models import (
    PlasmaTransformerSeqtoLab, PlasmaTransformerSeqtoSeq,
    StatePredictionPlasmaTransformer, PlasmaAutoformer)
from data_processing import (
    load_data, collate_fn_seq_to_label, collate_fn_seq_to_seq,
    collate_fn_autoformers)
import data_processing
import evaluation, arg_parsing, trainers, plotting, constants, utils
import copy

import random
import logging

from functools import partial

import warnings
warnings.filterwarnings('error', message="RuntimeWarning") 

args, arg_keys = arg_parsing.get_args()

print("--------------------")
print("Hyperparameters:")
print(args)
print("--------------------")

# Create a dictionary comprehension to extract the attribute values from args
arg_values = {key: getattr(args, key) for key in arg_keys}

# Unpack the dictionary to create local variables
locals().update(arg_values)

N_EMBD = constants.N_EMBD

# pyright: reportUndefinedVariable=none

if __name__ == "__main__":

    print("Setting up logging and random seed...")
    transformers_logging.set_verbosity(40)
    logging.basicConfig(level=logging.ERROR, filename="test_trainer.log", filemode="w")

    # set a random seed
    seed = 42

    seq_to_label, seq_to_seq = (model_type == "seq_to_label", model_type == "seq_to_seq")

    print("Loading data...")

    # get the name of the data file
    cwd = os.getcwd()
    f = utils.get_data_filename(cwd)
    data = load_data(f)

    taus = {
        "cmod": tau_cmod,
        "d3d": tau_d3d,
        "east": tau_east,
    }

    data = data_processing.add_disruption_column(data, taus)
    cols = data[0]["data"].columns
    feature_cols = [col for col in data[0]["data"].columns 
                    if not col in ["time", "cmod", "d3d", "east"]]

    disrupted_index = np.where(np.array(feature_cols) == "disrupted")[0][0]
    static_categorical_features = 3
    reduced_categorical_feature = 1 # [0, 1, 2]

    # scientific question: how much autocorrelation, how many lags 
    if lags_sequence_num == 1:
        lags_sequence = [1, 2, 3, 4, 5]
    elif lags_sequence_num == 2:
        lags_sequence = [1, 2, 4, 8, 16]
    elif lags_sequence_num == 3:
        lags_sequence = [1, 4, 16, 32, 64]

    model = PlasmaAutoformer(
        prediction_length=prediction_length,
        context_length=context_length,
        num_time_features=1, 
        num_real_dynamic_features=0, # TODO: put in programmed control, i_prog
        num_static_categorical_features=reduced_categorical_feature, # machine type
        num_static_real_features=0, # TODO: machine hyperparameters, when generating more data
        prediction_input_feature_size=len(feature_cols), # number of values not known at prediction time.
        lags_sequence=lags_sequence,
        num_parallel_samples=5,
        cardinality=[3], # cardinality of the machine type static categorical vars
        static_cat_embedding_dim=[1])

    print("sending model to device...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"Device: {device}")

    if device.type=="cuda" and testing:
        print("Naughty... don't test on GPU! Turning back to testing=False")
        testing = False 

    # make a dictionary of the machine hyperparameters
    machine_hyperparameters = {
        "cmod": [cmod_hyperparameter_non_disruptions, cmod_hyperparameter_disruptions],
        "d3d": [d3d_hyperparameter_non_disruptions, d3d_hyperparameter_disruptions],
        "east": [east_hyperparameter_non_disruptions, east_hyperparameter_disruptions],
    }

    train_inds, test_inds, val_inds = data_processing.train_test_val_inds(
        dataset=data,
        new_machine=new_machine,
        case_number=case_number,
        testing=testing,
        causal_matching=causal_matching,
        distance_fn_name=distance_fn_name,
        taus=taus,
    )

    training_args = TrainingArguments(
        output_dir="test_trainer",
        overwrite_output_dir=True,
        learning_rate=learning_rate,
        evaluation_strategy="steps",
        eval_steps=5,
        save_strategy="steps" if no_save else "epoch",
        save_total_limit=1,
        save_steps=1e9,
        logging_strategy="steps",
        logging_steps=100,
        do_train=True,
        do_predict=True,
        max_steps=max_steps if not testing else 10,
        per_device_train_batch_size=data_augmentation_ratio*2,
        load_best_model_at_end=not no_save,
        dataloader_pin_memory=False,
        metric_for_best_model="f1",
        lr_scheduler_type=lr_scheduler_type,
        warmup_steps=warmup_steps,
        adam_beta1=adam_beta1,
        adam_beta2=adam_beta2,
        )
    
    print("Setting up wandb...")
    utils.set_up_wandb(
        training_args=training_args, model=model, seed=seed, parsed_args=arg_values)

    data_processing_args = {
        "dataset": data,
        "train_inds": train_inds,
        "test_inds": test_inds,
        "val_inds": val_inds,
        "end_cutoff_timesteps": end_cutoff_timesteps,
        "end_cutoff_timesteps_test": end_cutoff_timesteps_test,
        "machine_hyperparameters": machine_hyperparameters,
        "dataset_type": "autoformer",
        "taus": taus,
        "data_augmentation": data_augmentation,
        "data_augmentation_windowing": data_augmentation_windowing,
        "data_augmentation_intensity": data_augmentation_ratio,
        "data_augmentation_ratio": data_augmentation_ratio,
        "scaling_type": scaling_type,
        "balance_classes": balance_classes,
        "max_length": max_length,
        "use_smoothed_tau": use_smoothed_tau,
        "window_length": disruptivity_distance_window_length,
        "context_length": context_length,
        "prediction_length": prediction_length,
        "max_lagged_sequence": max(lags_sequence),
    }

    train_dataset, test_dataset, val_dataset = data_processing.process_train_test_val_inds(
        dataset=data,
        train_inds=train_inds,
        test_inds=test_inds,
        val_inds=val_inds,
        data_processing_args=data_processing_args,) 
    
    train_disruptions, test_disruptions, val_disruptions = utils.print_dataset_info(
        train_dataset, test_dataset, val_dataset, seq_to_label)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    custom_collate = partial(collate_fn_autoformers, pad_from_left=True)

    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate)
    val_dataloader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate)
    test_dataloader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    num_epochs = 2 if testing else num_epochs

    for epoch in range(num_epochs):
        model.train()
        for batch in train_dataloader:
            # Assuming your collate function returns input_ids, attention_mask, etc.
            (past_values, future_values, past_time_features, 
             future_time_features, attention_mask, static_categorical_features) = (
                batch["past_values"], batch["future_values"], batch["past_time_features"],
                batch["future_time_features"], batch["attention_mask"], 
                batch["static_categorical_features"])
            
            (past_values, future_values, past_time_features, 
             future_time_features, attention_mask, static_categorical_features) = (
                past_values.to(device), future_values.to(device), past_time_features.to(device),
                future_time_features.to(device), attention_mask.to(device),
                static_categorical_features.to(device))

            # Forward pass
            outputs = model(
                past_values=past_values,
                future_values=future_values,
                past_time_features=past_time_features,
                future_time_features=future_time_features,
                past_observed_mask=attention_mask,
                static_categorical_features=static_categorical_features,
            )
            
            # Loss computation
            loss = outputs.loss

            # Backward pass
            loss.backward()
            
            # Optimization step
            optimizer.step()
            
            # Zero the gradients
            optimizer.zero_grad()

        print(f'Epoch {epoch+1} finished')

        # evaluate 
        model.eval()
        with torch.no_grad():
            val_preds, val_truth, val_i = trainers.predict_and_compute_autoformer(
                eval_model=model, dataloader=val_dataloader,
                disrupted_index=disrupted_index, plotting_label="val",
                device=device, stopping_i=None
            )
                
            train_preds, train_truth, _ = trainers.predict_and_compute_autoformer(
                eval_model=model, dataloader=train_dataloader,
                disrupted_index=disrupted_index, plotting_label="train",
                device=device, stopping_i=val_i
            )

            holdout_preds, holdout_truth, _ = trainers.predict_and_compute_autoformer(
                eval_model=model, dataloader=test_dataloader,
                disrupted_index=disrupted_index, plotting_label="holdout",
                device=device
            )

    print("~Not~ saving model to wandb... o_o ")
    # print("Saving model to wandb...")
    # trainer.save_model(os.path.join(wandb.run.dir, "model.h5"))

    print("Evaluating...")

    # evaluation.evaluate_main(trainer, val_dataset, seq_to_seq = seq_to_seq) 

    test_unrolled_predictions = evaluation.predict_rolling_autoformer(
        eval_model=model,
        test_dataset=[data[i] for i in test_inds],
        disrupted_index=disrupted_index,
        context_window=context_length,
        max_lagged_sequence=max(lags_sequence),
        prediction_length=prediction_length,
    )

    evaluation.compute_thresholded_statistics(
        test_unrolled_predictions, high_threshold=.66,
        low_threshold=.33, hysteresis=5,
    )

    print("Plotting predictions...")
    plotting.plot_disruptivity_autoformers(
        pred_dict=test_unrolled_predictions,
    )

    # print("Visualizing the attention layers...")
    # for index in (10, 20):
    #     plotting.visualize_attention_weights_main(
    #         val_dataset=val_dataset, index=index, trainer=trainer, num_layers=n_layer) 

    print("Done!")