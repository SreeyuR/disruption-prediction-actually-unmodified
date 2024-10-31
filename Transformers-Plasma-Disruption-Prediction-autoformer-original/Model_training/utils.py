import glob
import os
import wandb
import socket
import subprocess
import shutil
import numpy as np
import time 
import pandas as pd
import torch
import random
import arg_parsing
# pyright: reportUndefinedVariable=none


args, arg_keys = arg_parsing.get_args()

# Create a dictionary comprehension to extract the attribute values from args
arg_values = {key: getattr(args, key) for key in arg_keys}

# Unpack the dictionary to create local variables
locals().update(arg_values)
    

def return_round_robin_filename():
    """Return the filename for the next log file in the round robin sequence.

    Returns:
        next_log_filename (str): Filename for the next log file in the round robin sequence.
    """
    log_files = glob.glob("test_log_*.log")

    # Sort the files by their creation time
    sorted_log_files = sorted(log_files, key=os.path.getctime, reverse=True)

    if sorted_log_files:
        # Extract the number from the most recent log file
        most_recent_log_number = int(sorted_log_files[0].split("_")[2].split(".")[0])

        # Increment the number modulo 10
        next_log_number = (most_recent_log_number + 1) % 10
    else:
        # If no log files are found, start with 0
        next_log_number = 0
    
    next_log_filename = f"test_log_{next_log_number}.log"
    return next_log_filename


def get_data_filename(folder, filename):
    """
    Retrieves the full path for the specified data file.

    Args:
        folder (str): The name of the folder where the data file is located.
        filename (str): The name of the data file.

    Returns:
        str: Full path to the data file.

    Raises:
        FileNotFoundError: If the data file is not found.
    """
    cwd = os.path.join(os.getcwd(), folder)
    print(f'Current Working Directory: {cwd}')
    print(os.path.join(cwd, filename))
    if os.path.exists(os.path.join(cwd, filename)):
        f = os.path.join(cwd, filename)
    else:
        raise FileNotFoundError("Could not find the data file.")
    return f

def set_up_wandb(model, training_args, seed, parsed_args):
    """Set up wandb for logging.
    
    Args:
        model (PlasmaTransformer): The model to be trained.
        training_args (TrainingArguments): The training arguments.
        seed (int): The random seed.
        
    Returns:
        None
    """
    wandb.init(project="disruption-repdiction-transformer-orig",
               group=parsed_args["sweep_id"],
               name=parsed_args["run_id"],)

    if not check_wandb_connection():
        os.environ["WANDB_MODE"] = "offline"
    sync_offline_wandb_runs()

    # zip model.config and training_args into a single dictionary
    wandb_config = {**vars(model.config), **vars(training_args), **parsed_args}
    wandb.config.update(wandb_config)
    wandb.log({"seed": seed})
    # os.environ["WANDB_LOG_MODEL"] = "end"

    return


def check_wandb_connection(host="api.wandb.ai", port=443, timeout=5):
    """Check if the machine is connected to wandb."""
    try:
        socket.setdefaulttimeout(timeout)
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.connect((host, port))
            return True
    except socket.error:
        return False
    

def sync_offline_wandb_runs():
    """Sync offline wandb runs with the wandb server. Deletes old runs."""
    local_runs = glob.glob("wandb/offline-run-*")
    if local_runs and check_wandb_connection():
        for run_dir in local_runs:
            subprocess.run(["wandb", "sync", run_dir])
            # Uncomment the following line to remove the synced run directory
            shutil.rmtree(run_dir)
    return


def get_last_index_of_longest_list(lst):
    if len(lst) == 0:
        return None, None
    
    max_length_index = np.argmax([len(l) for l in lst])
    last_index = len(lst[max_length_index]) - 1
    
    return max_length_index, last_index


def print_dataset_info(train_dataset, test_dataset, val_dataset, seq_to_label=False):
    """Print information about the datasets.

    Args:
        train_dataset (Dataset): The training dataset.
        test_dataset (Dataset): The testing dataset.
        val_dataset (Dataset): The validation dataset.
        seq_to_label (bool): Whether the dataset is seq_to_label or not.
        
    Returns:
        None
    """

    print("--------------------")
    print("Dataset information:")
    print("Length of train dataset: ", len(train_dataset))
    print("Length of test dataset: ", len(test_dataset))
    print("Length of val dataset: ", len(val_dataset))
    train_disruptions = train_dataset.num_disruptions # np.sum([(train_dataset[i]["labels"][-1].detach().clone().cpu() > .5).tolist() for i in range(len(train_dataset))])
    test_disruptions = test_dataset.num_disruptions # np.sum([(test_dataset[i]["labels"][-1].detach().clone().cpu() > .5).tolist() for i in range(len(test_dataset))])
    val_disruptions = val_dataset.num_disruptions # np.sum([(val_dataset[i]["labels"][-1].detach().clone().cpu() > .5).tolist() for i in range(len(val_dataset))])
    print(f"Number of training disruptions: {train_disruptions}")
    print(f"Number of testing disruptions: {test_disruptions}")
    print(f"Number of validation disruptions: {val_disruptions}")
    print("--------------------")   

    return train_disruptions, test_disruptions, val_disruptions


def check_column_order(dataframes):
    """Check if all dataframes have the same column order.
    
    Args:
        dataframes (list): List of pandas DataFrames to check.
        
    Returns:
        bool: True if all dataframes have the same column order, False otherwise.
    """

    # Get the columns of the first dataframe in the list
    first_df_columns = dataframes[0].columns

    # Compare the columns of the first dataframe with the columns of each other dataframe
    for df in dataframes[1:]:
        if not df.columns.equals(first_df_columns):
            return False

    # If we've made it here, all dataframes have the same column order
    return True



def get_sequence_lengths_by_machine(dataset, train_inds):
    """Get the lengths of the sequences in the dataset, 
        separated by machine and label.

    Args:
        dataset (Dataset): The dataset.
        train_inds (list): The indices of the training examples 
            in the dataset.

    Returns:
        dict: A dictionary containing the lengths of the sequences, separated by machine and label.
    """

    disruptive_lens = {"cmod": [], "d3d": [], "east": []}
    non_disruptive_lens = {"cmod": [], "d3d": [], "east": []}
    
    for i in train_inds:
        if dataset[i]["machine"] == "cmod":
            if dataset[i]["label"]:
                disruptive_lens["cmod"].append(len(dataset[i]["data"]))
            else:
                non_disruptive_lens["cmod"].append(len(dataset[i]["data"]))
        elif dataset[i]["machine"] == "d3d":
            if dataset[i]["label"]:
                disruptive_lens["d3d"].append(len(dataset[i]["data"]))
            else:
                non_disruptive_lens["d3d"].append(len(dataset[i]["data"]))
        elif dataset[i]["machine"] == "east":
            if dataset[i]["label"]:
                disruptive_lens["east"].append(len(dataset[i]["data"]))
            else:
                non_disruptive_lens["east"].append(len(dataset[i]["data"]))
        else:
            pass
    
    dis_means = {}
    non_dis_means = {}

    for key, value in disruptive_lens.items():
        dis_means[key] = np.mean(value)
    
    for key, value in non_disruptive_lens.items():
        non_dis_means[key] = np.mean(value)

    return dis_means, non_dis_means


def determine_class_proportions(
    train_dataset,
    inverse_class_weighting,
    seq_to_label,
):
    """Determine the class proportions for the dataset.
    
    Args:
        train_dataset (Dataset): The training dataset.
        inverse_class_weighting (bool): Whether to use inverse class weighting or not.
        seq_to_label (bool): Whether the dataset is seq_to_label or not.
        
    Returns:
        list: A list containing the class proportions."""
    
    if inverse_class_weighting and seq_to_label:
        n = len(train_dataset)
        d = train_dataset.num_disruptions
        class_proportions = [(n - d) / n - .0001, d / n + .0001] 
    else:
        class_proportions = [1, 1]

    return class_proportions


def compute_wall_clock(eval_model, eval_dataset, length=1):
    # Measure the time taken for a forward pass
    
    times = []
    torch.compile(eval_model)
    for i in range(len(eval_dataset)):
        # sample a random integer the length of eval_dataset
        
        shot = eval_dataset[i]

        if shot["inputs_embeds"].shape[0] < length:
            continue
        
        start_time = time.time()

        # Jinxiang and Will: Put forward pass computation here. 
        with torch.no_grad():
            outputs = eval_model(inputs_embeds=shot["inputs_embeds"].unsqueeze(0).to(torch.float16)[:length])
        
        end_time = time.time()
        time_taken = end_time - start_time
        times.append(time_taken)

        if len(times) > 10:
            break
    
    # log a histogram of times to wandb
    wandb.log({f"wall_clock_mean_length_{length}": np.mean(times)})
    wandb.log({f"wall_clock_time_length_{length}": wandb.Histogram(times)})

    return


def moving_average_with_buffer(probs, buffer_value=0.1):
    """
    Compute a moving average with a starting buffer.
    
    Args:
    - probs (np.array): Original probabilities.
    - buffer_size (int): Size of the buffer.
    - buffer_value (float): Initial value of the buffer.
    
    Returns:
    - averaged_probs (np.array): Probabilities after applying moving average with buffer.
    """

    buffer_size = unrolled_smoothing
    buffer = [buffer_value] * buffer_size
    
    # Extend the original probabilities with the buffer at the start
    extended_probs = buffer + list(probs)
    
    # Compute the moving average
    averaged_probs = []
    threshold_index = int(0.9 * len(probs))  # 90% of the original list length
    
    for i in range(buffer_size, len(extended_probs)):
        # For the last 10%, only compute average if there are at least 2 values
        if i >= threshold_index:
            avg_value = extended_probs[i]
        else:
            avg_value = sum(extended_probs[i-buffer_size:i+1]) / buffer_size
            avg_value = avg_value / 3
        averaged_probs.append(avg_value)
    
    return np.array(averaged_probs)


def set_seed_across_frameworks(seed, tensorflow=False):
    """Set the seed across different numerical frameworks in python.
    
    Args:
        seed (int): The random seed.
        
    Returns:
        None
    """
    np.random.seed(seed)
    random.seed(seed)
    pd.options.mode.chained_assignment = None  # default='warn'
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    if tensorflow:
        import tensorflow as tf
        tf.random.set_seed(seed)
    
    return


def default_serialize(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"Object of type '{type(o).__name__}' is not JSON serializable")


def move_dict_to_cpu(input_dict):
    for key, value in input_dict.items():
        if isinstance(value, torch.Tensor):
            input_dict[key] = value.cpu()
    return input_dict
