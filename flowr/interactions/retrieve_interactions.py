import argparse
from pathlib import Path

import dill as pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm

from flowr.util.pocket import PROLIF_INTERACTIONS, BindingInteractions


def args():
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--data-path",
        type=str,
        help="Path to the data directory",
        required=True,
    )
    argparser.add_argument(
        "--state",
        type=str,
        help="train or test",
        required=True,
    )
    args = argparser.parse_args()
    return args


interaction_decoder = {i: k for i, k in enumerate(PROLIF_INTERACTIONS)}


def retrieve_interactions(
    data_path: str, state: str = "train", all_states: bool = False
):
    data_path = Path(data_path) / (state + ".smol")
    bytes_data = data_path.read_bytes()

    no_interaction = 0
    interactions_dict = {interaction: 0 for interaction in PROLIF_INTERACTIONS}
    for mol_bytes in tqdm(pickle.loads(bytes_data)):
        obj = pickle.loads(mol_bytes)
        interactions = BindingInteractions.from_bytes(
            obj["interactions"], interaction_types=PROLIF_INTERACTIONS
        )
        for row in interactions.array:
            for col in row:
                if np.count_nonzero(col) == 0:
                    no_interaction += 1
                    continue
                else:
                    interaction = np.argmax(col, axis=-1)
                    interactions_dict[interaction_decoder[interaction.item()]] += 1
    print(
        f"Sparsity of interactions: {(1 - sum(interactions_dict.values()) / no_interaction) * 100}%"
    )
    return interactions_dict


def _plot_interactions(
    interactions_dict: dict,
    data_path: Path,
    state: str,
    f_size: int = 30,
    palette: dict = None,
):
    """
    Plot the interactions with a custom color palette and global fontsize.
    """
    sns.set_theme(style="whitegrid")
    if palette is None:
        palette = {"train": "#D5D653", "val": "#DFC6F6", "test": "#857693"}

    if state == "all":
        all_interactions = list(PROLIF_INTERACTIONS)  # Preserve given order

        total_train = sum(interactions_dict["train"].values())
        total_val = sum(interactions_dict["val"].values())
        total_test = sum(interactions_dict["test"].values())
        train_freq = [
            interactions_dict["train"].get(inter, 0) / total_train
            for inter in all_interactions
        ]
        val_freq = [
            interactions_dict["val"].get(inter, 0) / total_val
            for inter in all_interactions
        ]
        test_freq = [
            interactions_dict["test"].get(inter, 0) / total_test
            for inter in all_interactions
        ]

        data = {
            "interaction": all_interactions * 3,
            "frequency": train_freq + val_freq + test_freq,
            "split": (
                ["train"] * len(all_interactions)
                + ["val"] * len(all_interactions)
                + ["test"] * len(all_interactions)
            ),
        }
        df = pd.DataFrame(data)

        plt.figure(figsize=(15, 10))
        ax = sns.barplot(
            data=df,
            x="interaction",
            y="frequency",
            hue="split",
            palette=palette,
            alpha=0.8,
        )
        ax.set_xlabel("")
        ax.set_ylabel("Normalized Frequency", fontsize=f_size)
        ax.tick_params(axis="x", labelsize=f_size, rotation=90)
        ax.tick_params(axis="y", labelsize=f_size)
        ax.legend(fontsize=f_size)
        plt.tight_layout()
        plt.savefig(data_path / "all_states_interactions_plot.png", dpi=300)
        plt.show()

    else:
        interactions = list(PROLIF_INTERACTIONS)  # Preserve ordering
        counts = [interactions_dict.get(inter, 0) for inter in interactions]
        total_count = sum(counts)
        normalized_counts = [count / total_count for count in counts]
        color = palette.get(state, None) if palette is not None else None

        plt.figure(figsize=(8, 6))
        ax = sns.barplot(x=interactions, y=normalized_counts, color=color, alpha=0.8)
        ax.set_xlabel("Interactions", fontsize=f_size)
        ax.set_ylabel("Normalized Frequency", fontsize=f_size)
        ax.tick_params(axis="x", labelsize=f_size, rotation="vertical")
        ax.tick_params(axis="y", labelsize=f_size)
        plt.tight_layout()
        plt.savefig(data_path / f"{state}_interactions_plot.png", dpi=300)
        plt.show()


if __name__ == "__main__":
    args = args()
    if args.state == "all":
        interactions_dict = {}
        for state in ["train", "val", "test"]:
            interactions_dict[state] = retrieve_interactions(args.data_path, state)
        _plot_interactions(interactions_dict, Path(args.data_path), state="all")
    else:
        interactions_dict = retrieve_interactions(args.data_path, args.state)
        _plot_interactions(interactions_dict, Path(args.data_path), args.state)
