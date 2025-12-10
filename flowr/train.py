import argparse
import warnings

import torch

import flowr.scriptutil as util
from flowr.data.data_info import GeneralInfos as DataInfos

warnings.filterwarnings(
    "ignore", category=UserWarning, message="TypedStorage is deprecated"
)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


################################ DEFAULTS ################################
DEFAULT_D_MODEL = 384
DEFAULT_POCKET_D_MODEL = 256
DEFAULT_N_LAYERS = 12
DEFAULT_POCKET_N_LAYERS = 6
DEFAULT_D_MESSAGE = 64
DEFAULT_D_EDGE = 128
DEFAULT_N_COORD_SETS = 128
DEFAULT_N_ATTN_HEADS = 32
DEFAULT_D_MESSAGE_HIDDEN = 96
DEFAULT_COORD_NORM = "length"
DEFAULT_EMB_SIZE = 64

DEFAULT_MAX_ATOMS = 183
DEFAULT_MAX_ATOMS_POCKET = 600

DEFAULT_EPOCHS = 200
DEFAULT_LR = 2e-4
DEFAULT_BATCH_COST = 512
DEFAULT_ACC_BATCHES = 1
DEFAULT_GRADIENT_CLIP_VAL = 10.0
DEFAULT_COORD_LOSS_WEIGHT = 1.0
DEFAULT_TYPE_LOSS_WEIGHT = 1.0
DEFAULT_BOND_LOSS_WEIGHT = 2.0
DEFAULT_CHARGE_LOSS_WEIGHT = 1.0
DEFAULT_HYBRIDIZATION_LOSS_WEIGHT = 1.0
DEFAULT_INTERACTION_LOSS_WEIGHT = 10.0
DEFAULT_AFFINITY_LOSS_WEIGHT = None
DEFAULT_DOCKING_LOSS_WEIGHT = None
DEFAULT_DISTANCE_LOSS_WEIGHT_LIG = None
DEFAULT_DISTANCE_LOSS_WEIGHT_LIG_POCKET = None
DEFAULT_SMOOTH_DISTANCE_LOSS_WEIGHT_LIG = None
DEFAULT_SMOOTH_DISTANCE_LOSS_WEIGHT_LIG_POCKET = None
PLDDT_CONFIDENCE_LOSS_WEIGHT = None
DEFAULT_CONFIDENCE_LOSS_WEIGHT = 0.0
DEFAULT_CONFIDENCE_GEN_STEPS = 20
DEFAULT_CATEGORICAL_STRATEGY = "uniform-sample"
DEFAULT_LR_SCHEDULE = "constant"
DEFAULT_LR_GAMMA = 0.998
DEFAULT_WARM_UP_STEPS = 2000
DEFAULT_BUCKET_COST_SCALE = "linear"

DEFAULT_N_VALIDATION_MOLS = 64  # 64 holo only data has 64 samples, apo-holo has 51
DEFAULT_NUM_INFERENCE_STEPS = 100
DEFAULT_CAT_SAMPLING_NOISE_LEVEL = 1.0
DEFAULT_COORD_NOISE_STD_DEV = 0.2
DEFAULT_COORD_NOISE_SCALE = 0.01
DEFAULT_COORD_SAMPLING_STRATEGY = "continuous"
DEFAULT_POCKET_COORD_NOISE_STD = 0.0
DEFAULT_TYPE_DIST_TEMP = 1.0
DEFAULT_TIME_ALPHA = 2.0
DEFAULT_TIME_BETA = 1.0
DEFAULT_ROTATION_ALIGNMENT = False
DEFAULT_PERMUTATION_ALIGNMENT = False

DEFAULT_SAMPLE_SCHEDULE = "linear"  # "log" or "linear"
DEFAULT_CORRECTOR_ITERS = 0


################################ MAIN SCRIPT ################################


def main(args):
    # Set some useful torch properties
    # Float32 precision should only affect computation on A100 and should in theory be a lot faster than the default setting
    # Increasing the cache size is required since the model will be compiled seperately for each bucket
    assert (
        args.pocket_noise is not None
    ), "pocket_noise must be set; choose from [apo, fix, random]"

    torch.set_float32_matmul_precision("high")
    torch._dynamo.config.cache_size_limit = util.COMPILER_CACHE_SIZE
    print(f"Set torch compiler cache size to {torch._dynamo.config.cache_size_limit}")

    util.disable_lib_stdout()
    util.configure_fs()

    print("Building model vocabs...")
    vocab = util._build_vocab()
    vocab_charges = util._build_vocab_charges()
    vocab_pocket_atoms = util._build_vocab_pocket_atoms()
    vocab_pocket_res = util._build_vocab_pocket_res()
    if args.add_feats:
        print("Including hybridization features...")
        vocab_hybridization = util._build_vocab_hybridization()
        vocab_aromatic = None  # util._build_vocab_aromatic()
    else:
        vocab_hybridization = None
        vocab_aromatic = None
    print("Vocabs complete.")

    print("Loading dataset statistics...")
    statistics = util.build_data_statistic(args)
    dataset_info = DataInfos(statistics, vocab, args)
    atom_types_distribution = dataset_info.atom_types.float()
    bond_types_distribution = dataset_info.edge_types.float()
    print("Dataset statistics complete.")

    print("Loading datamodule...")
    dm = util.build_dm(
        args,
        vocab,
        vocab_charges=vocab_charges,
        vocab_hybridization=vocab_hybridization,
        vocab_aromatic=vocab_aromatic,
        vocab_pocket_atoms=vocab_pocket_atoms,
        vocab_pocket_res=vocab_pocket_res,
        atom_types_distribution=atom_types_distribution,
        bond_types_distribution=bond_types_distribution,
    )
    print("Datamodule complete.")

    print("Building equinv model...")
    model = util.build_model(
        args,
        dm,
        dataset_info,
        train_mols=dm.train_mols,
        vocab=vocab,
        vocab_charges=vocab_charges,
        vocab_hybridization=vocab_hybridization,
        vocab_aromatic=vocab_aromatic,
        vocab_pocket_atoms=vocab_pocket_atoms,
        vocab_pocket_res=vocab_pocket_res,
    )
    if args.self_condition:
        print("Building model with self-conditioning...")
    else:
        print("Building model without self-conditioning...")
    print("Model building complete.")

    print("Fitting datamodule to model...")
    ckpt_path = None
    if args.load_ckpt is not None:
        print("Loading from checkpoint ...")

        ckpt_path = args.load_ckpt
        # ckpt = torch.load(ckpt_path)
        # if ckpt["optimizer_states"][0]["param_groups"][0]["lr"] != args.lr:
        #     print("Changing learning rate ...")
        #     ckpt["optimizer_states"][0]["param_groups"][0]["lr"] = args.lr
        #     ckpt["optimizer_states"][0]["param_groups"][0]["initial_lr"] = args.lr
        #     ckpt_path = (
        #         "lr" + "_" + str(args.lr) + "_" + os.path.basename(args.load_ckpt)
        #     )
        #     ckpt_path = os.path.join(
        #         os.path.dirname(args.load_ckpt),
        #         f"retraining_with_lr{args.lr}.ckpt",
        #     )
        #     torch.save(ckpt, ckpt_path)

    # model = torch.compile(model)
    trainer = util.build_trainer(args, model=model)
    trainer.fit(
        model,
        datamodule=dm,
        ckpt_path=ckpt_path if args.load_ckpt is not None else None,
    )
    print("Training complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    ################################ SETUP ################################
    parser.add_argument("--exp_name", type=str, default="train")
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--num_nodes", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--load_ckpt", type=str, default=None)
    parser.add_argument("--load_pretrained_ckpt", type=str, default=None)
    parser.add_argument("--lora_finetuning", action="store_true")
    parser.add_argument("--save_dir", type=str, default=".")
    parser.add_argument("--val_check_epochs", type=int, default=None)
    parser.add_argument("--val_check_interval", type=float, default=0.5)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--trial_run", action="store_true")

    ################################ MODEL ################################
    parser.add_argument(
        "--arch",
        type=str,
        default="pocket",
        choices=["transformer", "pocket", "pocket_flex"],
    )
    parser.add_argument("--d_model", type=int, default=DEFAULT_D_MODEL)
    parser.add_argument("--pocket_d_model", type=int, default=DEFAULT_POCKET_D_MODEL)
    parser.add_argument("--pocket_fixed_equi", action="store_true")
    parser.add_argument("--n_layers", type=int, default=DEFAULT_N_LAYERS)
    parser.add_argument("--pocket_n_layers", type=int, default=DEFAULT_POCKET_N_LAYERS)
    parser.add_argument("--d_message", type=int, default=DEFAULT_D_MESSAGE)
    parser.add_argument("--d_edge", type=int, default=DEFAULT_D_EDGE)
    parser.add_argument("--n_coord_sets", type=int, default=DEFAULT_N_COORD_SETS)
    parser.add_argument("--n_attn_heads", type=int, default=DEFAULT_N_ATTN_HEADS)
    parser.add_argument(
        "--d_message_hidden", type=int, default=DEFAULT_D_MESSAGE_HIDDEN
    )
    parser.add_argument("--coord_norm", type=str, default=DEFAULT_COORD_NORM)
    parser.add_argument("--emb_size", type=int, default=DEFAULT_EMB_SIZE)
    parser.add_argument("--max_atoms", type=int, default=DEFAULT_MAX_ATOMS)
    parser.add_argument(
        "--max_atoms_pocket", type=int, default=DEFAULT_MAX_ATOMS_POCKET
    )
    parser.add_argument("--use_lig_pocket_rbf", action="store_true")
    parser.add_argument("--use_sphcs", action="store_true")
    parser.add_argument("--use_rbf", action="store_true")
    parser.add_argument("--use_distances", action="store_true")
    parser.add_argument("--use_crossproducts", action="store_true")
    parser.add_argument("--add_feats", action="store_true")
    parser.add_argument("--remove_hs", action="store_true")
    parser.add_argument("--remove_aromaticity", action="store_true")

    ################################ DATALOADING ################################
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--data_paths", type=str, nargs="+", default=None)
    parser.add_argument("--dataset_weights", type=float, nargs="+", default=None)

    parser.add_argument("--dataset", type=str)
    parser.add_argument("--use_smol", action="store_true")
    parser.add_argument("--batch_cost", type=int, default=DEFAULT_BATCH_COST)
    parser.add_argument("--val_batch_cost", type=int, default=DEFAULT_BATCH_COST)
    parser.add_argument("--use_bucket_sampler", action="store_true")
    parser.add_argument(
        "--bucket_cost_scale", type=str, default=DEFAULT_BUCKET_COST_SCALE
    )
    parser.add_argument("--use_adaptive_sampler", action="store_true")
    parser.add_argument("--use_weighted_sampler", action="store_true")
    parser.add_argument("--acc_batches", type=int, default=DEFAULT_ACC_BATCHES)

    ################################ TRAINING ################################
    parser.add_argument("--train_confidence", action="store_true")
    parser.add_argument(
        "--confidence_loss_weight", type=float, default=DEFAULT_CONFIDENCE_LOSS_WEIGHT
    )
    parser.add_argument(
        "--confidence_gen_steps", type=int, default=DEFAULT_CONFIDENCE_GEN_STEPS
    )
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--weight_decay", type=float, default=1e-12)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--ligand_lr", type=float, default=None)
    parser.add_argument("--pocket_lr", type=float, default=None)
    parser.add_argument(
        "--gradient_clip_val", type=float, default=DEFAULT_GRADIENT_CLIP_VAL
    )
    parser.add_argument(
        "--coord_loss_weight", type=float, default=DEFAULT_COORD_LOSS_WEIGHT
    )
    parser.add_argument(
        "--type_loss_weight", type=float, default=DEFAULT_TYPE_LOSS_WEIGHT
    )
    parser.add_argument(
        "--bond_loss_weight", type=float, default=DEFAULT_BOND_LOSS_WEIGHT
    )
    parser.add_argument(
        "--charge_loss_weight", type=float, default=DEFAULT_CHARGE_LOSS_WEIGHT
    )
    parser.add_argument(
        "--hybridization_loss_weight",
        type=float,
        default=DEFAULT_HYBRIDIZATION_LOSS_WEIGHT,
    )
    parser.add_argument(
        "--distance_loss_weight_lig",
        type=float,
        default=DEFAULT_DISTANCE_LOSS_WEIGHT_LIG,
    )
    parser.add_argument(
        "--affinity_loss_weight",
        type=float,
        default=DEFAULT_AFFINITY_LOSS_WEIGHT,
    )
    parser.add_argument(
        "--docking_loss_weight",
        type=float,
        default=DEFAULT_DOCKING_LOSS_WEIGHT,
    )
    parser.add_argument(
        "--plddt_confidence_loss_weight",
        type=float,
        default=PLDDT_CONFIDENCE_LOSS_WEIGHT,
    )
    parser.add_argument(
        "--distance_loss_weight_lig_pocket",
        type=float,
        default=DEFAULT_DISTANCE_LOSS_WEIGHT_LIG_POCKET,
    )
    parser.add_argument(
        "--smooth_distance_loss_weight_lig",
        type=float,
        default=DEFAULT_SMOOTH_DISTANCE_LOSS_WEIGHT_LIG,
    )
    parser.add_argument(
        "--smooth_distance_loss_weight_lig_pocket",
        type=float,
        default=DEFAULT_SMOOTH_DISTANCE_LOSS_WEIGHT_LIG_POCKET,
    )
    parser.add_argument(
        "--interaction_loss_weight",
        type=float,
        default=DEFAULT_INTERACTION_LOSS_WEIGHT,
    )
    parser.add_argument("--use_fourier_time_embed", action="store_true")
    parser.add_argument("--use_t_loss_weights", action="store_true")
    parser.add_argument("--lr_schedule", type=str, default=DEFAULT_LR_SCHEDULE)
    parser.add_argument("--lr_gamma", type=float, default=DEFAULT_LR_GAMMA)
    parser.add_argument("--warm_up_steps", type=int, default=DEFAULT_WARM_UP_STEPS)
    parser.add_argument("--use_ema", action="store_true")
    parser.add_argument("--ema_decay", type=float, default=0.998)
    parser.add_argument("--self_condition", action="store_true", default=False)
    parser.add_argument("--inpaint_self_condition", action="store_true", default=False)
    parser.add_argument("--no_coord_skip_connect", action="store_true")
    parser.add_argument("--coord_update_every_n", type=int, default=3)

    ################################ INTERPOLATION ################################
    parser.add_argument("--pocket_noise", default=None, type=str)
    parser.add_argument("--separate_pocket_interpolation", action="store_true")
    parser.add_argument("--separate_interaction_interpolation", action="store_true")
    parser.add_argument("--interaction_fixed_time", type=float, default=None)
    parser.add_argument("--scale_coords", action="store_true")
    parser.add_argument("--flow_interactions", action="store_true")
    parser.add_argument("--predict_interactions", action="store_true")
    parser.add_argument("--predict_affinity", action="store_true")
    parser.add_argument("--predict_docking_score", action="store_true")
    parser.add_argument("--interaction_inpainting", action="store_true")
    parser.add_argument("--scaffold_inpainting", action="store_true")
    parser.add_argument(
        "--graph_inpainting",
        default=None,
        type=str,
        choices=["conformer", "random", "harmonic"],
    )
    parser.add_argument("--mixed_uncond_inpaint", action="store_true")
    parser.add_argument("--func_group_inpainting", action="store_true")
    parser.add_argument("--fragment_inpainting", action="store_true")
    parser.add_argument("--max_fragment_cuts", type=int, default=3)
    parser.add_argument("--substructure_inpainting", action="store_true")
    parser.add_argument("--substructure", type=str, default=None)
    parser.add_argument("--linker_inpainting", action="store_true")
    parser.add_argument("--core_inpainting", action="store_true")
    parser.add_argument("--use_cosine_scheduler", action="store_true")
    parser.add_argument(
        "--categorical_strategy", type=str, default=DEFAULT_CATEGORICAL_STRATEGY
    )
    parser.add_argument("--split_continuous_discrete_time", action="store_true")

    ################################ SAMPLING/INTERPOLATION ################################
    parser.add_argument(
        "--n_validation_mols", type=int, default=DEFAULT_N_VALIDATION_MOLS
    )
    parser.add_argument(
        "--num_inference_steps", type=int, default=DEFAULT_NUM_INFERENCE_STEPS
    )
    parser.add_argument(
        "--cat_sampling_noise_level",
        type=float,
        default=DEFAULT_CAT_SAMPLING_NOISE_LEVEL,
    )
    parser.add_argument(
        "--coord_noise_std_dev", type=float, default=DEFAULT_COORD_NOISE_STD_DEV
    )
    parser.add_argument("--coord_noise_schedule", type=str, default="standard")
    parser.add_argument(
        "--coord_noise_scale", type=float, default=DEFAULT_COORD_NOISE_SCALE
    )
    parser.add_argument(
        "--coord_sampling_strategy",
        type=str,
        default=DEFAULT_COORD_SAMPLING_STRATEGY,
    )
    parser.add_argument(
        "--use_sde_simulation",
        action="store_true",
    )
    parser.add_argument(
        "--sample_schedule",
        type=str,
        default=DEFAULT_SAMPLE_SCHEDULE,
        choices=["log", "linear"],
    )
    parser.add_argument(
        "--pocket_coord_noise_std",
        type=float,
        default=DEFAULT_POCKET_COORD_NOISE_STD,
    )
    parser.add_argument("--type_dist_temp", type=float, default=DEFAULT_TYPE_DIST_TEMP)
    parser.add_argument("--time_alpha", type=float, default=DEFAULT_TIME_ALPHA)
    parser.add_argument("--time_beta", type=float, default=DEFAULT_TIME_BETA)
    parser.add_argument("--mixed_uniform_beta_time", action="store_true")
    parser.add_argument(
        "--rotation_alignment", action="store_true", default=DEFAULT_ROTATION_ALIGNMENT
    )
    parser.add_argument(
        "--permutation_alignment",
        action="store_true",
        default=DEFAULT_PERMUTATION_ALIGNMENT,
    )
    parser.add_argument("--corrector_iters", type=int, default=DEFAULT_CORRECTOR_ITERS)

    ################################ DEFAULTS ################################
    parser.set_defaults(
        trial_run=False,
        use_ema=True,
        # self_condition=True,
    )

    args = parser.parse_args()
    main(args)
