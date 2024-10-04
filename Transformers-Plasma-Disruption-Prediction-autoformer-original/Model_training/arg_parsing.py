import argparse


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_args():
    parser = argparse.ArgumentParser()

    # transformer architecture parameters
    parser.add_argument("--n_head", type=int, default=1)
    parser.add_argument("--n_layer", type=int, default=10)
    parser.add_argument("--n_inner", type=int, default=10)
    parser.add_argument("--attn_pdrop", type=float, default=0.1)
    parser.add_argument("--activation_function", type=str, default="gelu") 
    parser.add_argument("--resid_pdrop", type=float, default=0.1)
    parser.add_argument("--embd_pdrop", type=float, default=0.1)
    parser.add_argument("--add_cross_attention", type=str2bool, default=True)
    parser.add_argument("--layer_norm_epsilon", type=float, default=1e-5)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--pad_from_left", type=str2bool, default=True)
    parser.add_argument("--truncate_longer", type=str2bool, default=True)
    parser.add_argument("--attention_head_at_end", type=str2bool, default=False)

    # autoformer parameters
    parser.add_argument('--context_length', type=int, default=20, help='The context length for the Autoformer model.')
    parser.add_argument('--prediction_length', type=int, default=3, help='The prediction length for the Autoformer model.')
    parser.add_argument(
        '--lags_sequence', type=str, 
        default="1_2_3_4_5_6_7_10_11_12_13_15",
        help='The lags sequence for the Autoformer model.')
    parser.add_argument('--lags_sequence_num', type=int, default=1, help='The lags sequence case num.')

    # training parameters
    parser.add_argument("--max_steps", type=int, default=30000)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--lr_scheduler_type", type=str, default="linear")
    parser.add_argument("--warmup_steps", type=int, default=50)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--optim", type=str, default="adam")
    parser.add_argument("--sequential_discounted_loss", type=str2bool, default=False)
    parser.add_argument("--discount_factor", type=float, default=0.9995)
    parser.add_argument("--testing", type=str2bool, default=True)
    parser.add_argument("--sequential_loss_weight", type=int, default=1)
    parser.add_argument("--inverse_class_weighting", type=str2bool, default=True)
    parser.add_argument("--multi_loss", type=str2bool, default=False)
    parser.add_argument("--no_save", type=str2bool, default=True)
    parser.add_argument("--normalize_loss_by_sequence_length", type=str2bool, default=True)
    parser.add_argument("--data_augmentation", type=str2bool, default=False)
    parser.add_argument("--data_augmentation_windowing", type=str2bool, default=True)
    parser.add_argument("--data_augmentation_ratio", type=int, default=2)
    parser.add_argument("--data_augmentation_intensity", type=float, default=1)
    parser.add_argument("--data_augmentation_factor", type=float, default=1) # deprecated TODO: remove.
    parser.add_argument("--scaling_type", type=str, default="standard")
    parser.add_argument("--pretraining_max_steps", type=int, default=7000)
    parser.add_argument("--pretraining_batch_size", type=int, default=256)
    parser.add_argument("--pretraining_learning_rate", type=float, default=1e-4)
    parser.add_argument("--post_pretraining_transformer_learning_rate", type=float, default=1e-4)
    parser.add_argument("--pretraining_loss", type=str, default="log_cosh")
    parser.add_argument("--pretraining", type=str2bool, default=True)
    parser.add_argument("--curriculum_steps", type=int, default=2)
    parser.add_argument("--balance_classes", type=str2bool, default=False)
    parser.add_argument("--regularize_logits", type=str2bool, default=True)
    parser.add_argument("--regularize_logits_weight", type=float, default=0.1)
    parser.add_argument("--model_type", type=str, default="seq_to_label")
    parser.add_argument("--vanilla_loss", type=str2bool, default=True)
    parser.add_argument("--causal_matching", type=str2bool, default=False)
    parser.add_argument("--distance_fn_name", type=str, default="pearsons")
    parser.add_argument("--disruptivity_distance_window_length", type=int, default=8)
    parser.add_argument("--disruptivity_distance_mean_and_std", type=str2bool, default=True)
    parser.add_argument("--num_epochs", type=int, default=200)

    # data parameters
    parser.add_argument("--cmod_hyperparameter_non_disruptions", type=float, default=.1)
    parser.add_argument("--cmod_hyperparameter_disruptions", type=float, default=.9)
    parser.add_argument("--d3d_hyperparameter_non_disruptions", type=float, default=.1)
    parser.add_argument("--d3d_hyperparameter_disruptions", type=float, default=.9)
    parser.add_argument("--east_hyperparameter_non_disruptions", type=float, default=.05)
    parser.add_argument("--east_hyperparameter_disruptions", type=float, default=.92)
    parser.add_argument("--end_cutoff", type=float) 
    parser.add_argument("--seed", type=int, default=67)
    parser.add_argument("--end_cutoff_timesteps", type=int, default=8)
    parser.add_argument("--end_cutoff_timesteps_test", type=int, default=8)
    parser.add_argument("--test_cmod_only", type=str2bool, default=False)
    parser.add_argument("--new_machine", type=str, default="cmod")
    parser.add_argument("--case_number", type=int, default=6)
    parser.add_argument("--tau_cmod", type=int, default=12)
    parser.add_argument("--tau_d3d", type=int, default=35)
    parser.add_argument("--tau_east", type=int, default=75)
    parser.add_argument("--standardize_disruptivity_plot_length", type=str2bool, default=True)
    parser.add_argument("--use_smoothed_tau", type=str2bool, default=True)
    parser.add_argument("--alex_method", type=str, default=False)
    parser.add_argument('--data_context_length', type=int, default=100)
    parser.add_argument('--fix_sampling', type=str2bool, default=True)
    parser.add_argument("--keep_jx_uneven_sampling", type=str2bool, default=True)
    parser.add_argument("--smooth_v_loop", type=str2bool, default=True)
    parser.add_argument("--v_loop_smoother", type=int, default=10)

    #eval parameters
    parser.add_argument("--eval_high_thresh", type=float, default=.8)
    parser.add_argument("--eval_low_thresh", type=float, default=.5)
    parser.add_argument("--eval_hysteresis", type=int, default=2)
    parser.add_argument("--unrolled_smoothing", type=int, default=10)
    parser.add_argument("--sweep_id", type=str, default=None)
    parser.add_argument("--run_id", type=str, default=None)

    args = parser.parse_args()

    # create a list of the keys of args
    args_keys = list(vars(args).keys())

    return args, args_keys