import matplotlib.pyplot as plt
import numpy as np
import wandb
import torch
import seaborn as sns

# Constants
COLUMN_NAMES = ['ip_error_normalized', 'Greenwald_fraction', 'q95', 'time',
       'n_equal_1_normalized', 'east', 'cmod', 'd3d', 'beta_p', 'kappa', 'li',
       'lower_gap', 'v_loop']

FEATURE_COLUMN_NAMES =  ['ip_error_normalized', 'Greenwald_fraction', 'q95',
       'n_equal_1_normalized', 'beta_p', 'kappa', 'li',
       'lower_gap', 'v_loop']


def set_up_disruptivity_prediction_plot(
        probs_len,
        threshold,
        lims = [0, 1]):
    
    """Setup the plot that will have a lot of disruptivity predictions on it.
    
    Args:
        probs_len (np.array): Array of probabilities.
        threshold (float): Threshold for predicting disruption.
        title (str): Title of the plot.
    """

    fig, ax1 = plt.subplots()

    ax1.plot([threshold]*probs_len, "c--")
    ax1.set_xlabel('Time (or other common index)')
    ax1.set_ylabel('Disruptivity')

    # set ax1 limits between 0 and 1
    ax1.set_ylim([0, 1])

    plt.title("Predictions of Disruptivity across Holdout Set")
    
    return fig, ax1
    

def visualize_attention_weights(inputs, model, plot_index, model_type="classifier", layer=0, head=0):
    """Visualize the attention weights of a given layer and head of a model.

    Args:
        inputs (dict): Slice of ModelReadyDataset to feed to the model.
        model (transformers.PreTrainedModel): Model to visualize.
        plot_index (int): Index of the slice of ModelReadyDataset to feed to the model.
        model_type (str): Type of model to visualize.
        layer (int): Layer to visualize.
        head (int): Head to visualize.
    """

    # Get the attention weights from the model
    model.eval().cpu()
    with torch.no_grad():
        outputs = model(
            inputs_embeds=inputs["inputs_embeds"].to(torch.float16), 
            output_attentions=True)
        attentions = outputs.attentions

    # Extract the attention weights of the specified layer and head
    attention = attentions[layer][0, head].cpu().detach().numpy()

    # Normalize the attention weights
    attention = attention / np.sum(attention, axis=-1, keepdims=True)

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(attention, cmap="gray_r", vmax=.15, ax=ax)
    ax.set_xlabel("Time slice of input shot")
    ax.set_ylabel("Length of input subsequence as forward prediction is rolled out (decreasing)")
    ax.set_title(f"{model_type} Attention Weights (Layer {layer}, Head {head + 1}), Shot {inputs['shot']}")

    # Log the image to wandb
    wandb.log({(f"attention weights for {model_type} " + str(layer) + f" index {plot_index}"): wandb.Image(fig)})

    # Close the figure to prevent further modifications
    plt.close(fig)


def visualize_attention_weights_main(val_dataset, index, trainer, num_layers, number):
    """Visualize the attention weights of the model. Checks if layers exist.

    Args:
        val_dataset (Dataset): Validation dataset.
        trainer (Trainer): Trainer object.
        num_layers (int): Number of layers in the model.
    """

    visualize_attention_weights(val_dataset[index], trainer.model.eval(), plot_index=number, layer=1)

    if num_layers > 1:
        visualize_attention_weights(val_dataset[index], trainer.model.eval(), plot_index=number, layer=num_layers-1)

    return

def plot_state_predictions(input, pretrainer):
    """Plot the state prediction against the true parameter
    
    Args:
        input: input slice of ModelReadyDataset
        pretrainer: the huggingface trainer
    """
    outputs = pretrainer.predict([input])

    for k in range(input["labels"].shape[1]):
        truth = input["labels"][:, k].cpu().detach().numpy()
        title = COLUMN_NAMES[k]
        prediction_logits = outputs.predictions

        fig, ax1 = plt.subplots()

        ax1.plot(truth, 'r-', label="truth")
        ax1.plot(prediction_logits[0,:,k], "b-", label="predictions")

        ax1.set_xlabel('Time')
        ax1.set_ylabel('Parameter normalized', color='r')

        # set ax1 limits between 0 and 1
        plt.title(title)
        
        wandb.log({("state prediction " + str(k)): wandb.Image(fig)})
