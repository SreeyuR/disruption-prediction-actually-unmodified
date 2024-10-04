import wandb
from transformers.integrations import WandbCallback, TrainerCallback
import torch
import evaluation


class AllWandbCallback(WandbCallback):
    """WandbCallback that logs all metrics.
    
    Args:
        log_all_metrics (bool): Whether to log all metrics.
    """

    def __init__(self, global_step=0, prefix="", log_all_metrics=True, **kwargs):
        super().__init__(**kwargs)
        self.log_all_metrics = log_all_metrics
        self.prefix = prefix
        self.global_step = global_step

    def on_step_end(self, args, state, control, **kwargs):
        self.global_step += 1
        if self.global_step % 100 == 0:
            if len(state.log_history):
                logs = {f"{self.prefix}_{k}": v for k, v in state.log_history[-1].items()}
                wandb.log(logs, step=self.global_step, commit=True)

    def on_evaluation(self, args, state, control, **kwargs):
        if len(state.log_history):
            print(f"in on_evaluation: global_step={self.global_step}")
            logs = {f"{self.prefix}_{k}": v for k, v in state.log_history[-1].items()}
            wandb.log(logs, step=self.global_step, commit=True)
        super().on_epoch_end(args, state, control, **kwargs)


class NoSaveEarlyStoppingCallback(TrainerCallback):
    def __init__(self, early_stopping_patience, early_stopping_threshold):
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_threshold = early_stopping_threshold
        self.best_metric = None
        self.no_improvement_count = 0

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if state.is_local_process_zero:
            # Change `eval_loss` to the metric of your choice
            current_metric = metrics["eval_f1"]

            if self.best_metric is None or abs(current_metric - self.best_metric) > self.early_stopping_threshold:
                self.best_metric = current_metric
                self.no_improvement_count = 0
            else:
                self.no_improvement_count += 1

            if self.no_improvement_count >= self.early_stopping_patience:
                control.should_training_stop = True
                print(f"Early stopping at epoch {state.epoch} due to no improvement in evaluation metric.")


class EvaluationWithoutSavingCallback(TrainerCallback):
    def __init__(self, trainer):
        super().__init__()
        self._best_metric = None
        self._best_model = None
        self.trainer = trainer

    def on_epoch_end(self, args, state, control, **kwargs):
        metrics = self.trainer.evaluate()
        self.trainer.log(metrics)

        if self.trainer.hp_search_backend is not None:
            self.trainer.hp_search_backend.report_objective(metrics[self.trainer.metric_for_best_model])

            self.trainer._save_checkpoint(self.trainer.model, trial=self.trainer.hp_search_backend.get_trial_id())
            self.trainer.storage.delete_dir(self.trainer.output_dir)


class BestModelCallback(TrainerCallback):
    """Saves the best model to trainer.model without saving it to disk."""
    def __init__(self):
        self.best_metric = float('-inf')  # assuming you want to maximize the metric
        self.best_model = None

    def on_evaluate(self, args, state, control, logs=None, **kwargs):
        # Assuming "metric_name" is the name of your metric
        current_metric = kwargs['metrics']['eval_f1']

        if current_metric is not None and current_metric > self.best_metric:
            self.best_metric = current_metric
            # Copy the current model's state_dict
            self.best_model = {k: v.clone().detach() for k, v in kwargs['model'].state_dict().items()}

    def on_train_end(self, args, state, control, **kwargs):
        if self.best_model:
            kwargs['model'].load_state_dict(self.best_model)