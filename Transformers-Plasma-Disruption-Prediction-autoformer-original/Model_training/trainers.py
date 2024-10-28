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
import evaluation

import warnings
warnings.filterwarnings('once') 

# Constants
COLUMN_NAMES = ['ip_error_normalized', 'Greenwald_fraction', 'q95', 'time',
       'n_equal_1_normalized', 'east', 'cmod', 'd3d', 'beta_p', 'kappa', 'li',
       'lower_gap', 'v_loop']

FEATURE_COLUMN_NAMES =  ['ip_error_normalized', 'Greenwald_fraction', 'q95',
       'n_equal_1_normalized', 'beta_p', 'kappa', 'li',
       'lower_gap', 'v_loop']


class SequentialDiscountedLossTrainer(Trainer):
    """Custom trainer for the model implementing a temporal discounted loss.
        
    Args:
        discount_factor (float): Discount factor for temporal discounted loss.
        device (torch.device): Device to use for training.
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.
        
    Attributes:
        discount_factor (float): Discount factor for temporal discounted loss."""
    
    def __init__(self, device, discount_factor=.9995,
                 max_length=2048, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.discount_factor = discount_factor
        self.device = device
        self.discount_weights = torch.tensor(
            [self.discount_factor ** i for i in range(max_length)]).to(self.device)
 

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")

        # create a mask tensor for padding values
        mask = (labels != -100)
        # select only the valid tokens in the labels tensor
        valid_labels = labels[mask]

        time_steps = labels.size()[1]
        outputs = model(**inputs)
        logits = outputs.logits

        # Create a weight tensor that decays with time
        time_decay_weights = self.discount_weights[:time_steps]
        # Apply the time-decay weights to the logits
        weighted_logits = logits * time_decay_weights.view(1, -1, 1)

        # select only the corresponding logits for the valid tokens
        valid_logits = weighted_logits[mask.expand_as(weighted_logits)].view(-1, weighted_logits.size(-1))

        # Compute the cross-entropy loss using the weighted logits
        loss = F.cross_entropy(valid_logits, valid_labels, reduction='mean')
        
        return (loss, outputs) if return_outputs else loss


class ClassImbalanceLossTrainer(Trainer):
    """Custom trainer for the model implementing a weighted loss for class imbalance.
    
    Args:
        class_weights (np.array): List of weights for each class."""
    def __init__(self,
                device,
                seq_to_seq,
                class_weights,
                regularize_logits,
                regularize_logits_weight,
                *args, **kwargs):
        
        super().__init__(*args, **kwargs)
        
        self.device = device
        self.inverse_class_weights = torch.Tensor(1/np.array(class_weights)).to(self.device)
        self.regularize_logits = regularize_logits
        self.regularize_logits_weight = regularize_logits_weight
        self.seq_to_seq = seq_to_seq
    
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels").to(self.device)
        outputs = model(**inputs)
        logits = outputs.logits

        if self.seq_to_seq:
            logits = logits.permute(0, 2, 1)
            labels = labels.squeeze(-1)
        else:
            labels = labels[:, 1].to(torch.long)
        
        # compute the cross-entropy loss using the weighted logits
        loss = F.cross_entropy(
            input=logits,
            target=labels,
            weight=self.inverse_class_weights,
            reduction='mean',
            ignore_index=-100)

        if self.regularize_logits:
            penalty = self.regularize_logits_fn(inputs, model, self.seq_to_seq)
            combined_loss = loss + self.regularize_logits_weight * penalty
            return (combined_loss, outputs) if return_outputs else combined_loss
        
        return (loss, outputs) if return_outputs else loss
    
    def regularize_logits_fn(self, inputs, model, seq_to_seq):
        """Return a penalty if logits increase non-monotonically or by a lot.
        
        Args:
            inputs (dict): Input dictionary.
            model (object): Model.
            seq_to_seq (bool): Whether the model is a seq to seq model.
        
        Returns:
            torch.Tensor: Penalty."""
        
        if not seq_to_seq:
            probs = []
            for t in range(1, inputs["inputs_embeds"].size()[1], 10):  ## TODO... implement in seq to seq! 
                model_outputs = model(
                    inputs_embeds=inputs["inputs_embeds"][:, :t, :].to(torch.float16),
                    attention_mask=inputs["attention_mask"][:, :t])
                model_output = model_outputs.logits
                prob = F.softmax(model_output, dim=-1)[:, 1]
                probs.append(prob)
            probs = torch.vstack(probs)
        else:
            model_outputs = model(
                inputs_embeds=inputs["inputs_embeds"].unsqueeze(0).to(torch.float16),
                attention_mask=inputs["attention_mask"])
            model_output = model_outputs.logits
            probs = F.softmax(model_output, dim=-1)[:, :, 1]

        # select the probabilities for the positive class
        probs_diff = torch.diff(probs, dim=1)
        non_monotonic_penalty = ((probs_diff > .3).float() + (probs_diff < 0).float()).sum()

        return non_monotonic_penalty

class ComputeTrainingMetricsTrainer(Trainer):
    def _maybe_log_save_evaluate(self, tr_loss, model, trial, epoch, ignore_keys_for_eval):
        if self.control.should_log:

            logs: typing.Dict[str, float] = {}

            # all_gather + mean() to get average loss over all processes
            tr_loss_scalar = self._nested_gather(tr_loss).mean().item()

            # reset tr_loss to zero
            tr_loss -= tr_loss

            logs["loss"] = round(tr_loss_scalar / (self.state.global_step - self._globalstep_last_logged), 4)
            logs["learning_rate"] = self._get_learning_rate()

            self._total_loss_scalar += tr_loss_scalar
            self._globalstep_last_logged = self.state.global_step
            self.store_flos()

            self.log(logs)

        metrics = None
        if self.control.should_evaluate:
            eval_metrics = self.evaluate(ignore_keys=ignore_keys_for_eval)
            sample_train_inds = np.random.choice(len(self.train_dataset), 500)
            train_metrics = self.evaluate(
                    eval_dataset=self.train_dataset.subset(sample_train_inds),
                    ignore_keys=ignore_keys_for_eval,
                    metric_key_prefix=f"train",
                )
            metrics = {**eval_metrics, **train_metrics}
            self._report_to_hp_search(trial, self.state.global_step, metrics)
            
        if self.control.should_save:
            self._save_checkpoint(model, trial, metrics=metrics)
            self.control = self.callback_handler.on_save(self.args, self.state, self.control)

