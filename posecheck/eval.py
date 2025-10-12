import argparse

from posecheck.data.datasets import *

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--dataset", type=str, default="crossdocked_test")
    args = args.parse_args()

    dataset_name = args.dataset

    # Load the dataset
    types = ["score_only", "minimized", "redocked"]

    for method in [
        "ar",
        "targetdiff",
        "pocket2mol",
        "cvae",
        "diffsbdd",
        "flag",
        "decompdiff",
        "pocket2mol",
    ]:
        for type in types:
            dataset = RedockedData(method, type)
            try:
                dataset.all_strain_energy
            except:
                pass

            try:
                dataset.all_interactions
            except:
                pass

            try:
                dataset.all_clashes
            except:
                pass
