import os

from transformers import TrainingArguments, Trainer, EarlyStoppingCallback
from transformers.utils import logging as transformers_logging
from transformers.trainer_callback import DefaultFlowCallback, ProgressCallback

import json
from datetime import datetime 

import torch
import wandb
import numpy as np

# local imports
from custom_callbacks import AllWandbCallback
import custom_callbacks
from models import (
    PlasmaTransformerSeqtoLab, PlasmaTransformerSeqtoSeq,
    StatePredictionPlasmaTransformer)
from data_processing import (
    load_data, collate_fn_seq_to_label)
import data_processing
import evaluation, arg_parsing, trainers, plotting, dataset_types
import utils
import copy

import random
import logging

from functools import partial
import eval_class.model_eval as me

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

wandb.login()

N_EMBD = 12

# pyright: reportUndefinedVariable=none

if __name__ == "__main__":

    print("Setting up logging and random seed...")
    transformers_logging.set_verbosity(40)
    logging.basicConfig(level=logging.ERROR, filename="test_trainer.log", filemode="w")

    # set a random seed
    seed = 42
    utils.set_seed_across_frameworks(seed)

    ModelPlaceholder = PlasmaTransformerSeqtoLab

    model = ModelPlaceholder(
        output_dir = "test_trainer",
        n_head = n_head,
        n_layer = n_layer,
        n_inner = n_inner,
        attn_pdrop = attn_pdrop,
        activation_function = activation_function,
        resid_pdrop = resid_pdrop,
        embd_pdrop = embd_pdrop,
        layer_norm_epsilon = layer_norm_epsilon,
        pretrained_model=None,
        max_length=max_length,
        n_embd=N_EMBD,
        attention_head_at_end=attention_head_at_end,)
    
    print("sending model to device...")
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"Device: {device}")

    if device.type=="cuda" and testing:
        print("Naughty... don't test on GPU! Turning back to testing=False")
        testing = False 

    print("Loading data...")

    # get the name of the data file
    cwd = os.getcwd()
    data_filename="Full_HDL_dataset_unnormalized_no_nan_column_names_w_shot_and_time.pickle"
    folder = f"{os.getcwd()}/Transformers-Plasma-Disruption-Prediction-autoformer-original/Model_training"
    f = utils.get_data_filename(folder, data_filename)
    data = load_data(f)

    if fix_sampling:
        data = data_processing.fix_resample_data_issue(data=data)

    # make a dictionary of the machine hyperparameters
    machine_hyperparameters = {
        "cmod": [cmod_hyperparameter_non_disruptions, cmod_hyperparameter_disruptions],
        "d3d": [d3d_hyperparameter_non_disruptions, d3d_hyperparameter_disruptions],
        "east": [east_hyperparameter_non_disruptions, east_hyperparameter_disruptions],
    }

    taus = {
        "cmod": tau_cmod,
        "d3d": tau_d3d,
        "east": tau_east,
    }

    train_inds, test_inds, val_inds = data_processing.train_test_val_inds_from_file(
        case_number=case_number,
        testing=testing,
    )

    training_args = TrainingArguments(
        output_dir="test_trainer",
        overwrite_output_dir=True,
        learning_rate=learning_rate,
        evaluation_strategy="steps",
        eval_steps=5 if not testing else 1,
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
        
    args_dict = training_args.to_dict()
    args_dict['max_steps'] = pretraining_max_steps if not testing else 10
    args_dict['metric_for_best_model'] = "loss"
    args_dict['per_device_train_batch_size'] = pretraining_batch_size
    pretraining_args = TrainingArguments(**args_dict)
    
    print("Setting up wandb...")
    utils.set_up_wandb(
        training_args=training_args, 
        model=model, seed=seed, parsed_args=arg_values)

    data_processing_args = {
        "dataset": data,
        "train_inds": train_inds,
        "test_inds": test_inds,
        "val_inds": val_inds,
        "end_cutoff_timesteps": end_cutoff_timesteps,
        "end_cutoff_timesteps_test": end_cutoff_timesteps_test,
        "machine_hyperparameters": machine_hyperparameters,
        "dataset_type": "state",
        "taus": taus,
        "use_smoothed_tau": use_smoothed_tau,
        "data_augmentation_windowing": data_augmentation_windowing,
        "data_augmentation_intensity": data_augmentation_ratio,
        "data_augmentation_ratio": data_augmentation_ratio,
        "ratio_to_augment": data_augmentation_ratio,
        "scaling_type": scaling_type,
        "max_length": max_length,
        "window_length": disruptivity_distance_window_length,
        "context_length": data_context_length,
        "smooth_v_loop": smooth_v_loop,
        "v_loop_smoother": v_loop_smoother,
    }

    if pretraining:
        print("------------------------")
        print('training state prediction')
        print("------------------------")

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
            pretrained_model=None,
            max_length=max_length,
            loss=pretraining_loss,
            attention_head_at_end=False,
            n_embd=N_EMBD)
        
        state_train_dataset, state_test_dataset, state_val_dataset = (
            data_processing.process_train_test_val_inds(
                train_inds=train_inds,
                test_inds=test_inds,
                val_inds=val_inds,
                dataset=data,
                data_processing_args=data_processing_args))

        pretrainer = Trainer(
            model=state_prediction_model,
            args=pretraining_args,
            data_collator=data_processing.collate_fn_seq_to_seq_state,
            train_dataset=state_train_dataset,
            eval_dataset=state_test_dataset,
            compute_metrics=evaluation.compute_metrics_state_prediction,
            # callbacks=[AllWandbCallback(prefix="state_pretraining")],
        )
        if torch.backends.mps.is_available():
            pretrainer.device = torch.device("mps")
        elif torch.cuda.is_available():
            pretrainer.device = torch.device("cuda")
        else:
            pretrainer.device = torch.device("cpu")

        # torch.compile(pretrainer.model)
        pretrainer.train()

        print("----------------------------------------")
        print("Done pretraining state prediction model.")
        print("----------------------------------------")

        model.from_pretrained_lm(state_prediction_model)

        sample_ind = random.sample(range(len(state_val_dataset)), 1)[0]
        # plotting.plot_state_predictions(
        #     input=state_val_dataset[sample_ind],
        #     pretrainer=pretrainer)
        
        # plotting.visualize_attention_weights(
        #     inputs=state_val_dataset[sample_ind],
        #     model=state_prediction_model.eval(),
        #     plot_index=1,
        #     layer=n_layer - 1,
        #     model_type="state_prediction",)

    ecs = [8, 6, 4, 2, 0]
    selected_ecs = ecs[:curriculum_steps]

    # reverse the order of selected_ecs
    selected_ecs = selected_ecs[::-1]

    global_step = 0

    for i, ec in enumerate(selected_ecs):

        print("---------------------")
        print(f"Training with end cutoff {ec}")
        print("---------------------")

        data_processing_args["dataset_type"] = "seq_to_label"
        data_processing_args["end_cutoff_timesteps"] = ec
        seq_to_label = True

        train_dataset, test_dataset, val_dataset = data_processing.process_train_test_val_inds(
            train_inds=train_inds,
            test_inds=test_inds,
            val_inds=val_inds,
            dataset=data,
            data_processing_args=data_processing_args) 
        
        train_disruptions, test_disruptions, val_disruptions = utils.print_dataset_info(
            train_dataset, test_dataset, val_dataset, seq_to_label)

        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        allWandbCallback = AllWandbCallback(global_step=global_step, prefix=f"cutoff_{ec}")
        best_model_callback = custom_callbacks.BestModelCallback()
        
        if not vanilla_loss:
            class_proportions = utils.determine_class_proportions(
            train_dataset, inverse_class_weighting, seq_to_label)
            trainer = trainers.ClassImbalanceLossTrainer(
                model=model,
                args=training_args,
                data_collator=partial(
                    collate_fn_seq_to_label,
                    pad_from_left=pad_from_left,
                    truncate_longer=truncate_longer),
                class_weights=class_proportions,
                device=device,
                train_dataset=train_dataset,
                eval_dataset=test_dataset,
                compute_metrics=evaluation.compute_metrics,
                # callbacks=[allWandbCallback],
                optimizers=(optimizer, None),
                regularize_logits=regularize_logits,
                regularize_logits_weight=regularize_logits_weight,
            )

        else:
            trainer = Trainer(
                model=model,
                args=training_args,
                data_collator=partial(
                    collate_fn_seq_to_label,
                    pad_from_left=pad_from_left,
                    truncate_longer=truncate_longer),
                compute_metrics=evaluation.compute_metrics,
                train_dataset=train_dataset,
                eval_dataset=test_dataset,
                # callbacks=[allWandbCallback],
                callbacks=[best_model_callback],
                optimizers=(optimizer, None),
            )

        if torch.backends.mps.is_available():
            trainer.device = torch.device("mps")
        elif torch.cuda.is_available():
            trainer.device = torch.device("cuda")
        else:
            trainer.device = torch.device("cpu")

        # Train the model and get the training state
        training_state = trainer.train()

        # Update the global step counter
        global_step += training_state.global_step

    # print("~Not~ saving model to wandb... o_o ")
    
    print("Saving model to wandb...")
    trainer.save_model(os.path.join(wandb.run.dir, "model.h5"))

    print("Computing wall clock time...")
   # for length in (10, 100, 1000):
   #     utils.compute_wall_clock(eval_model = trainer.model.eval(), eval_dataset = val_dataset, length=length)
    
    print("Evaluating...")

    evaluation.evaluate_main(trainer, test_dataset) 
    evaluation.evaluate_main_seq(
        trainer, test_dataset,
        standardize_plot_length=standardize_disruptivity_plot_length)   

    print("Visualizing the attention layers...")
    sampled_shots = random.sample(range(len(test_dataset)), 5)

    eval_dataset = dataset_types.ModelReadyDataset(
        shots=[data[i] for i in test_inds],
        end_cutoff_timesteps=0,
        machine_hyperparameters=machine_hyperparameters,
        taus=taus,
        max_length=max_length,
        smooth_v_loop=smooth_v_loop,
        v_loop_smoother=v_loop_smoother,
    )
        
    print("Evaluating model performance w the ENI class...")
    if eval_high_thresh < eval_low_thresh:
        eval_low_thresh = eval_high_thresh

    eval_dict = evaluation.create_eval_dict(eval_dataset, trainer=trainer, master_dataset=data)
    eval_dict = utils.move_dict_to_cpu(eval_dict)
    params, metrics = evaluation.return_eval_params()

 #  bound eval_high_thresh between .5 and .75 
    eval_high_thresh = max(0.5, min(eval_high_thresh, 0.6))

    # bound eval_low_thresh between .25 and .5 (change these bounds as needed)
    eval_low_thresh = max(0.4, min(eval_low_thresh, 0.5))

    params['high_thr'] = eval_high_thresh 
    params['low_thr'] = eval_low_thresh 
    params['t_hysteresis'] = eval_hysteresis if eval_hysteresis < 4 else 3

    performance_calculator = me.model_performance()

    val_metrics_report = performance_calculator.eval(
        unrolled_proba=eval_dict,
        metrics=metrics,
        params_dict=params,
    )

    print("Saving evaluation metrics to wandb...")
    wandb.log(val_metrics_report)
    
    # save val_metrics_report dict locally to json
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"case_{case_number} eval_class_{timestamp}.json"

    # with open(filename, 'w') as json_file:
    #     json.dump(val_metrics_report, json_file, indent=4, default=utils.default_serialize)
    eval_disruptions = [i for i in range(len(eval_dataset)) if eval_dataset[i]["labels"][1]>.5][:2]
    eval_non_disruptions = [i for i in range(len(eval_dataset)) if eval_dataset[i]["labels"][1]<.5][:2]
    eval_inds = eval_disruptions + eval_non_disruptions

    for i, index in enumerate(eval_inds):
        plotting.visualize_attention_weights_main(
            val_dataset=eval_dataset, index=index, trainer=trainer, num_layers=n_layer, number=i) 
    
    
    print("Done!")
