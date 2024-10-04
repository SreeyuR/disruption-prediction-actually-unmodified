import os

from transformers import TrainingArguments, Trainer, EarlyStoppingCallback
from transformers.utils import logging as transformers_logging

import torch
import wandb

# local imports
from custom_callbacks import AllWandbCallback
import custom_callbacks
from models import PlasmaTransformerSeqtoLab, PlasmaTransformerSeqtoSeq, StatePredictionPlasmaTransformer
from data_processing import load_data, collate_fn_seq_to_label, collate_fn_seq_to_seq
import data_processing
from evaluation import MultiLossExperimentalTrainer
import evaluation, arg_parsing
import utils
import copy

import random
import logging

args, arg_keys = arg_parsing.get_args()

print("--------------------")
print("Hyperparameters:")
print(args)
print("--------------------")

# Create a dictionary comprehension to extract the attribute values from args
arg_values = {key: getattr(args, key) for key in arg_keys}

# Unpack the dictionary to create local variables
locals().update(arg_values)

# pyright: reportUndefinedVariable=none

if __name__ == "__main__":

    print("Setting up logging and random seed...")
    transformers_logging.set_verbosity(40)
    logging.basicConfig(level=logging.ERROR, filename="test_trainer.log", filemode="w")

    # set a random seed
    seed = 42

    state_prediction_model = StatePredictionPlasmaTransformer(
        output_dir = "test_trainer",
        n_head = n_head,
        n_layer = n_layer,
        n_inner = n_inner,
        attn_pdrop = attn_pdrop,
        activation_function = activation_function,
        resid_pdrop = resid_pdrop,
        embd_pdrop = embd_pdrop,
        layer_norm_epsilon = layer_norm_epsilon,
        loss=pretraining_loss,
    )

    ModelPlaceholder = PlasmaTransformerSeqtoLab if seq_to_label else PlasmaTransformerSeqtoSeq

    model = ModelPlaceholder(
        output_dir = "test_trainer",
        n_head = n_head,
        n_layer = n_layer,
        n_inner = n_inner,
        attn_pdrop = attn_pdrop,
        activation_function = activation_function,
        resid_pdrop = resid_pdrop,
        embd_pdrop = embd_pdrop,
        layer_norm_epsilon = layer_norm_epsilon,)
    
    print("sending model to device...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"Device: {device}")

    if device.type=="cuda" and testing:
        print("Naughty... don't test on GPU! Turning back to testing=False")
        testing = False 

    print("Loading data...")

    # get the name of the data file
    cwd = os.getcwd()
    f = utils.get_data_filename(cwd)
    data = load_data(f)

    # make a dictionary of the machine hyperparameters
    machine_hyperparameters = {
        "cmod": cmod_hyperparameter,
        "d3d": d3d_hyperparameter,
        "east": east_hyperparameter,
    }

    state_train_dataset, state_test_dataset, state_val_dataset = data_processing.train_test_val_split(
        dataset=data,
        end_cutoff=end_cutoff,
        end_cutoff_timesteps=end_cutoff_timesteps,
        machine_hyperparameters=machine_hyperparameters,
        dataset_type="state",
        new_machine=new_machine,
        case_number=case_number,
        tau=tau,
        testing=testing,
        data_augmentation=False, # TODO: add data augmentation for states
        data_augmentation_windowing=False,
        data_augmentation_factor=data_augmentation_factor,
        scaling_type=scaling_type,
        tau_additions_switch=tau_additions_switch,) 

    training_args = TrainingArguments(
        output_dir="test_trainer",
        overwrite_output_dir=True,
        learning_rate=learning_rate,
        evaluation_strategy="steps",
        eval_steps=100,
        save_strategy="steps" if no_save else "epoch",
        save_total_limit=1,
        save_steps=1e9,
        logging_strategy="steps",
        logging_steps=1e3,
        do_train=True,
        max_steps=10000, #max_steps if not testing else 10,
        per_device_train_batch_size=batch_size,
        load_best_model_at_end=not no_save,
        dataloader_pin_memory=False,
        metric_for_best_model="f1",
        lr_scheduler_type=lr_scheduler_type,
        warmup_steps=warmup_steps,
        adam_beta1=adam_beta1,
        adam_beta2=adam_beta2,
        )
    
    pretraining_args = copy.deepcopy(training_args)
    pretraining_args.max_steps = 10000 # pretraining_max_steps if not testing else 10
    pretraining_args.metric_for_best_model = "loss"
    pretraining_args.per_device_train_batch_size = pretraining_batch_size

    print("Setting up wandb...")
    utils.set_up_wandb(
        training_args=training_args, model=model, seed=seed)


    print("Training...")

    if not no_save:
        early_stopping = EarlyStoppingCallback(
            early_stopping_patience=10, early_stopping_threshold=0.001)
    else:    
        early_stopping = custom_callbacks.NoSaveEarlyStoppingCallback(
            early_stopping_patience=10, early_stopping_threshold=0.001)

    if seq_to_label + seq_to_seq_vanilla + multi_loss > 1:
        raise ValueError("Only one of seq_to_label, seq_to_seq_vanilla, multi_loss can be True")

    # Set up state_ptretraining Trainer 

    # TODO: setup hyperparams to do differential learning rates. 
    # Continue training all weights, but with different learning rates for transformer and head

    print("------------------------")
    print('training state prediction')
    print("------------------------")

    pretrainer = Trainer(
        model=state_prediction_model,
        args=pretraining_args,
        data_collator=data_processing.collate_fn_seq_to_seq_state,
        train_dataset=state_train_dataset,
        eval_dataset=state_test_dataset,
        compute_metrics=evaluation.compute_metrics_state_prediction,
        callbacks=[AllWandbCallback(prefix="state_pretraining")],
    )

    if torch.cuda.is_available():
        pretrainer.device = torch.device('cuda')

    pretrainer.train()

    sample_ind = random.sample(range(len(state_val_dataset)), 1)[0]

    evaluation.plot_state_predictions(
        input=state_val_dataset[sample_ind],
        pretrainer=pretrainer)
    
    evaluation.visualize_attention_weights(
        inputs=state_val_dataset[sample_ind],
        model=state_prediction_model.eval(),
        layer=n_layer - 1)

    print("Pretraining Done!")


    ## todo: merge this branch back into the main one? and then try pretraining vs not? 


    