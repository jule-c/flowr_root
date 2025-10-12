import glob
import os
from typing import List, Tuple

import datamol as dm
import pandas as pd
import prolif as plf
import torch
from tqdm import tqdm

from posecheck.utils.clashes import count_clashes_list
from posecheck.utils.interactions import generate_interaction_df
from posecheck.utils.loading import (get_ids_to_pockets, get_pdbqt_mol,
                               load_mols_from_sdf, load_protein_from_pdb,
                               read_pdbqt)
from posecheck.utils.strain import get_strain_energy

# get directory of this file
DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(DIR, "../../data/")

POCKET_DIR = os.path.join(DATA_DIR, "crossdocked/")
PDBQT_DIR = os.path.join(DATA_DIR, "crossdocked_pdbqt/")


class BasePosesDataset(object):
    """Base class for loading protein-ligand complexes for PoseCheck."""

    def __init__(
        self,
        data_dir: str = DATA_DIR,
        split_path: str = os.path.join(DATA_DIR, "split_by_name.pt"),
        pocket_dir: str = POCKET_DIR,
        pdbqt_dir: str = PDBQT_DIR,
        protein_data=None,
    ):
        """
        Initializes the BasePosesDataset class.

        Args:
            data_dir (str, optional): The directory where the data is located. Defaults to DATA_DIR.
            split_path (str, optional): The path to the file that contains the train, validation, and test splits. Defaults to os.path.join(DATA_DIR, "split_by_name.pt").
            pocket_dir (str, optional): The directory where the protein pocket data is located. Defaults to POCKET_DIR.
            pdbqt_dir (str, optional): The directory where the PDBQT files are located. Defaults to PDBQT_DIR.
            protein_data (optional): Preloaded protein data. If None, the protein data will be loaded from the proteins.pt file in the data directory. Defaults to None.
        """
        self.data_dir = data_dir
        self.pocket_dir = pocket_dir
        self.pdbqt_dir = pdbqt_dir
        split_path = os.path.join(data_dir, "crossdocked_split_by_name.pkl")
        self.names = get_ids_to_pockets(split_path)

        if protein_data is None:
            proteins_path = os.path.join(data_dir, "proteins.pt")
            if os.path.exists(proteins_path):
                self.proteins = torch.load(proteins_path)
            else:
                raise Exception("No proteins found, preprocess first")

        else:
            self.proteins = protein_data

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}"

    def __len__(self):
        return len(self.names)

    def get_pdb_path(self, idx: int):
        return glob.glob(
            self.pocket_dir + f"{self.names[idx]}*" + "*.pdb", recursive=True
        )[0]

    def get_pdbqt_path(self, idx: int):
        return glob.glob(
            self.pdbqt_dir + f"{self.names[idx]}*" + "*.pdbqt", recursive=True
        )[0]

    def prepare_pdb(self, idx: int):
        # pdb_path = glob.glob(self.pocket_dir + f"{name}*" + "*.pdb", recursive=True)[0]
        # return load_protein_from_pdb(pdb_path)
        return self.proteins[idx]

    def prepare_sdf(self, name: str, idx: int = 0):
        """Implement in child class. This should be the only class that needs defining for a new dataset."""
        raise NotImplementedError

    def __getitem__(self, idx) -> Tuple[plf.Molecule, plf.Molecule]:
        """Returns a tuple of protein and ligand structures.

        Args:
            idx (int): The index of the pocket.

        Returns:
            Tuple[pd.AtomGroup, pybel.Molecule]: The protein and ligand structures.
        """

        # Select pocket
        name = self.names[idx]

        # Load protein and ligand
        prot = self.prepare_pdb(idx)
        lig = self.prepare_sdf(name, idx)

        return prot, lig

    def get_interaction_df(self, idx: int) -> pd.DataFrame:
        """Returns a dataframe of interactions between the protein and ligand.

        Args:
            idx (int): The index of the pocket.

        Returns:
            pd.DataFrame: The interactions between the protein and ligand.
        """
        try:
            prot, lig = self.__getitem__(idx)
            interaction_df = generate_interaction_df(prot, lig)
            return interaction_df
        except Exception as e:
            print(f"Error with {idx}: {str(e)}")
            return None

    @property
    def all_interactions(self) -> List[pd.DataFrame]:
        """Returns a list of interaction DataFrames for all pockets in this dataset.

        Calculates the interactions if they have not already been calculated.

        List will contain None values if there was an error with the pocket.

        Returns:
            List[pd.DataFrame]: A list of interaction DataFrames.
        """
        interactions_path = (
            self.data_dir
            + f"/benchmarks/preprocessed/{self.__repr__()}_interactions.pt"
        )

        if not os.path.exists(interactions_path):
            print("Calculating interactions...")
            dataset_name = self.__repr__()
            interactions = [
                self.get_interaction_df(idx)
                for idx in tqdm(range(len(self)), desc=dataset_name)
            ]
            torch.save(interactions, interactions_path)
        else:
            interactions = torch.load(interactions_path)

        # Count number of None values
        n_nones = sum([1 for i in interactions if i is None])
        print(f"{self.__repr__()}: Number of None values: {n_nones}")

        return interactions

    @property
    def all_clashes(self):
        clashes_path = (
            self.data_dir + f"/benchmarks/clashes/{self.__repr__()}_clashes.pt"
        )

        if not os.path.exists(clashes_path):
            print("Calculating clashes...")
            dataset_name = self.__repr__()
            clashes = [
                count_clashes_list(*self.__getitem__(idx), target=self.names[idx][:4])
                for idx in tqdm(range(len(self)), desc=dataset_name)
            ]
            # clashes = torch.stack(torch.tensor(clashes))
            torch.save(clashes, clashes_path)
        else:
            clashes = torch.load(clashes_path)

        # Count number of None values
        # n_nones = sum([1 for i in interactions if i is None])
        # print(f"{self.__repr__()}: Number of None values: {n_nones}")

        return clashes

    @property
    def all_strain_energy(self):
        energies_path = (
            self.data_dir + f"/benchmarks/energies/{self.__repr__()}_energies.pt"
        )

        if not os.path.exists(energies_path):
            mols = []
            for idx in tqdm(range(len(self))):
                mols.extend(self.load_mols(idx))

            energies = [get_strain_energy(mol) for mol in tqdm(mols)]

            torch.save(energies, energies_path)
        else:
            energies = torch.load(energies_path)

        return energies


# class CrossdockedDataset(BasePosesDataset):
#     def __init__(
#         self,
#         data_dir: str = DATA_DIR
#         split_path: str = "/Users/charlie/projects/poses_benchmark/data/split_by_name.pt",
#     ):
#         super().__init__(data_dir=data_dir, split_path=split_path)
#         self.ligand_dir = data_dir + "/crossdocked/"

#     def prepare_sdf(self, name: str, idx: int = 0):
#         sdf_path = glob.glob(self.ligand_dir + f"{name}*" + "*.sdf", recursive=True)[0]
#         return 1(sdf_path)


# class ZINCSamples(BasePosesDataset):
#     def __init__(
#         self,
#         data_dir: str = "/Users/charlie/projects/poses_benchmark/data",
#         split_path: str = "/Users/charlie/projects/poses_benchmark/data/split_by_name.pt",
#     ):
#         super().__init__(data_dir=data_dir, split_path=split_path)
#         self.ligand_dir = data_dir + "/benchmarks/zinc/"

#     def prepare_sdf(self, name: str, idx: int = 0):
#         sdf_path = glob.glob(self.ligand_dir + f"{name}*" + "*.sdf", recursive=True)[0]
#         return load_mols_from_sdf(sdf_path)


# class TargetDiffSamples(BasePosesDataset):
#     def __init__(
#         self,
#         data_dir: str = "/Users/charlie/projects/poses_benchmark/data",
#         split_path: str = "/Users/charlie/projects/poses_benchmark/data/split_by_name.pt",
#     ):
#         super().__init__(data_dir=data_dir, split_path=split_path)

#         self.ligand_dir = data_dir + "/benchmarks/targetdiff/"
#         self.data = torch.load(
#             os.path.join(self.ligand_dir, "targetdiff_vina_docked.pt")
#         )

#     def prepare_sdf(self, name: str, idx: int = 0):
#         samples = self.data[idx]

#         mols = [sample["mol"] for sample in samples]
#         tmp_path = self.ligand_dir + "tmp.sdf"
#         dm.to_sdf(mols, tmp_path)

#         lig = load_mols_from_sdf(tmp_path)
#         os.remove(tmp_path)

#         return lig


# class BenchmarkModelSamples(BasePosesDataset):
#     def __init__(
#         self,
#         model_name: str,  # cvae, p2m or sbdd
#         data_dir: str = DATA_DIR,
#         split_path: str = os.path.join(DATA_DIR, "split_by_name.pt"),
#     ):
#         super().__init__(data_dir=data_dir, split_path=split_path)

#         self.name = model_name
#         self.ligand_dir = data_dir + f"/benchmarks/{model_name}/"

#     def __repr__(self) -> str:
#         return f"{self.name}"

#     def prepare_sdf(self, name: str, idx: int = 0):
#         sample_dir = self.ligand_dir + f"pocket_{idx}/"

#         # Remove tmp.sdf incase it exists
#         tmp_path = sample_dir + "tmp.sdf"
#         if os.path.exists(tmp_path):
#             os.remove(tmp_path)

#         # Read all SDFs in the folder and make a single temp SDF with all molecules
#         sdf_paths = glob.glob(sample_dir + "*.sdf", recursive=True)

#         # If no SDFs found, return None
#         if len(sdf_paths) == 0:
#             print("No SDFs found for", name, idx, "returning None")
#             return None

#         # Read all SDFs and make a single temp SDF with all molecules
#         mols = []
#         for sdf_path in sdf_paths:
#             try:
#                 mols.append(dm.read_sdf(sdf_path)[0])
#             except:
#                 pass

#         dm.to_sdf(mols, tmp_path)

#         # Load the temp SDF
#         ligs = load_mols_from_sdf(tmp_path)
#         os.remove(tmp_path)

#         return ligs


class DockedMolsDataset(BasePosesDataset):
    def __init__(
        self,
        name: str,
        docked: bool,
        data_dir: str = DATA_DIR,
        split_path: str = os.path.join(DATA_DIR, "split_by_name.pt"),
    ):
        """Dataset for loading docked molecules from dataset:

        Args:
            name (str): Name of dataset
            docked (bool): Whether to load docked or generated molecules
            data_dir (str, optional): [description]. Defaults to DATA_DIR.
            split_path (str, optional): [description]. Defaults to os.path.join(DATA_DIR, "split_by_name.pt").
        """

        super().__init__(data_dir=data_dir, split_path=split_path)
        self.name = name
        self.docked = docked

        self.ligand_dir = (
            data_dir + "/benchmarks/targetdiff/"
        )  # NOTE samples taken from targetdiff

        #
        self.data = torch.load(os.path.join(self.ligand_dir, f"{name}_vina_docked.pt"))

    def __repr__(self) -> str:
        return f"{self.name}_{'docked' if self.docked else 'generated'}"

    def load_generated_mols(self, idx: int = 0):
        if self.name != "crossdocked_test":
            return [sample["mol"] for sample in self.data[idx]]
        else:
            return [self.data[idx]["mol"]]

    def load_docked_mols(self, idx: int = 0):
        if self.name != "crossdocked_test":
            return [
                get_pdbqt_mol(sample["vina"]["dock"][0]["pose"])
                for sample in self.data[idx]
            ]
        else:
            return [
                get_pdbqt_mol(self.data[idx]["vina"]["dock"][0]["pose"])
            ]  # Because crossdocked_test only has 1 sample

    def load_mols(self, idx: int):
        """Loads molecules for a target directly without computing interactions or preparing ligand with plip.

        Args:
            idx (int, optional): id. Defaults to 0.

        Returns:
            List[Chem.Mol]: List of molecules.
        """
        if self.docked:
            return self.load_docked_mols(idx)
        elif not self.docked:
            return self.load_generated_mols(idx)

    def prepare_sdf(self, name: str, idx: int = 0):
        mols = self.load_mols(idx)

        # Remove None values
        mols = [dm.add_hs(mol, add_coords=True) for mol in mols if mol is not None]

        tmp_path = self.ligand_dir + "tmp.sdf"
        dm.to_sdf(mols, tmp_path)

        try:
            lig = load_mols_from_sdf(tmp_path)
            os.remove(tmp_path)
            return lig
        except Exception as e:
            print(e)
            return []


class DiffSBDDSamples(BasePosesDataset):
    def __init__(
        self,
        docked,
        diffsbdd_dir="/home/cch57/projects/poses_benchmark/data/benchmarks/diffsbdd/",
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.name = "diffsbdd"
        self.docked = docked
        self.ligand_dir = diffsbdd_dir

    def __repr__(self) -> str:
        return f"{self.name}_{'docked' if self.docked else 'generated'}"

    def load_generated_mols(self, idx: int = 0):
        name = self.names[idx]
        pdbqt_files = glob.glob(self.ligand_dir + f"*{name}*" + "*.pdbqt")
        pdbqt_files = [
            pdbqt_file for pdbqt_file in pdbqt_files if "out" not in pdbqt_file
        ]
        generated_mols = [read_pdbqt(pdbqt_file) for pdbqt_file in pdbqt_files]
        return generated_mols

    def load_docked_mols(self, idx: int = 0):
        name = self.names[idx]
        pdbqt_files = glob.glob(self.ligand_dir + f"*{name}*" + "*.pdbqt")
        pdbqt_files = [
            pdbqt_file for pdbqt_file in pdbqt_files if "out" not in pdbqt_file
        ]

        docked_paths = [
            pdbqt_file.split(".")[0] + "_out.pdbqt" for pdbqt_file in pdbqt_files
        ]
        docked_mols = [read_pdbqt(pdbqt_file) for pdbqt_file in docked_paths]
        return docked_mols

    def load_mols(self, idx: int = 0):
        if self.docked:
            return self.load_docked_mols(idx)
        elif not self.docked:
            return self.load_generated_mols(idx)

    def prepare_sdf(self, name: str, idx: int = 0):
        mols = self.load_mols(idx)

        # Remove None values
        mols = [dm.add_hs(mol, add_coords=True) for mol in mols if mol is not None]

        tmp_path = self.ligand_dir + "tmp.sdf"
        dm.to_sdf(mols, tmp_path)

        try:
            lig = load_mols_from_sdf(tmp_path)
            os.remove(tmp_path)
            return lig
        except Exception as e:
            print(e)
            return []


class GraphBPSamples(BasePosesDataset):
    def __init__(self, data_dir: str = DATA_DIR, *args, **kwargs):
        super().__init__(data_dir=data_dir, *args, **kwargs)
        self.name = "graphbp"
        self.data_dir = os.path.join(data_dir, "benchmarks", self.name)

        self.sdf_files = [
            glob.glob(
                os.path.join(self.data_dir, f"*{name[:4].replace('-', '_')}*.sdf")
            )
            for name in self.names
        ]

    def load_mols(self, idx: int = 0):
        target_sdfs = self.sdf_files[idx]

        mols = []
        for sdf in target_sdfs:
            try:
                mols.append(dm.read_sdf(sdf)[0])
            except:
                print(f"Failed to load {sdf} adding None")
                mols.append(None)

        # ! NOTE in the middle of refactoring to remove self.docked
        # if self.docked:
        #     return self.load_docked_mols(idx)
        # elif not self.docked:
        #     return self.load_generated_mols(idx)

        return mols

    def prepare_sdf(self, name: str, idx: int = 0):
        raise NotImplementedError


class FLAGSamples(BasePosesDataset):
    def __init__(self, data_dir: str = DATA_DIR, *args, **kwargs):
        super().__init__(data_dir=data_dir, *args, **kwargs)
        self.name = "flag"
        # self.data_dir = os.path.join(data_dir, 'benchmarks', self.name)
        self.data_dir
        self.ligand_dir = data_dir + "/benchmarks/flag/"
        self.docked = False

        self.sdf_files = [
            glob.glob(
                os.path.join(self.ligand_dir, f"sample_{i}", "generated_mols.sdf")
            )
            for i in range(len(self.names))
        ]

    def load_mols(self, idx: int = 0):
        # TODO Make for generated too
        mols = dm.read_sdf(self.sdf_files[idx][0])

        return mols

    def prepare_sdf(self, name: str, idx: int = 0):
        mols = self.load_mols(idx)

        mols = [dm.add_hs(mol, add_coords=True) for mol in mols if mol is not None]

        # Remove None values
        # mols = [dm.add_hs(mol, add_coords=True) for mol in mols if mol is not None]

        tmp_path = self.ligand_dir + "tmp.sdf"
        dm.to_sdf(mols, tmp_path)

        try:
            lig = load_mols_from_sdf(tmp_path)
            os.remove(tmp_path)
            return lig
        except Exception as e:
            print(e)
            return []


DECOMPDIFF_DIR = "/home/cch57/rds/rds-liolab-8eFRJ5nImu8/data/crossdocked"


class DecompDiffSamples(BasePosesDataset):
    def __init__(self, data_dir: str = DECOMPDIFF_DIR, *args, **kwargs):
        super().__init__(data_dir=data_dir, *args, **kwargs)
        self.name = "decompdiff"
        # self.data_dir = os.path.join(data_dir, 'benchmarks', self.name)
        self.data_dir  # = data_dir
        self.ligand_dir = data_dir + "/benchmarks/flag/"
        self.docked = False

        # self.sdf_files = [glob.glob(os.path.join(self.ligand_dir, f'sample_{i}', "generated_mols.sdf")) for i in range(len(self.names))]

    def load_mols(self, idx: int = 0):
        # get all directories with 'sampling_drift' in the name
        name = self.names[idx][:4]

        paths = glob.glob(
            os.path.join(self.data_dir, "sampling_drift" + f"*{name}*", "*.pt")
        )

        if len(paths) == 0:
            print(f"No paths found for {name}")
            return []
        else:
            mols_path = glob.glob(
                os.path.join(self.data_dir, "sampling_drift" + f"*{name}*", "*.pt")
            )[0]

        data = torch.load(mols_path)

        mols = [sample["mol"] for sample in data]
        mols

        return mols

    def prepare_sdf(self, name: str, idx: int = 0):
        mols = self.load_mols(idx)

        mols = [dm.add_hs(mol, add_coords=True) for mol in mols if mol is not None]

        # Remove None values
        # mols = [dm.add_hs(mol, add_coords=True) for mol in mols if mol is not None]

        tmp_path = self.ligand_dir + "tmp.sdf"
        dm.to_sdf(mols, tmp_path)

        try:
            lig = load_mols_from_sdf(tmp_path)
            os.remove(tmp_path)
            return lig
        except Exception as e:
            print(e)
            return []


def load_docking_data():
    print("Loading docking data...")

    DOCKING_DIR = "/home/cch57/projects/poses_benchmark/data/docking_results"

    models = os.listdir(DOCKING_DIR)
    dfs = []

    for model in models:
        docking_files = glob.glob(os.path.join(DOCKING_DIR, model, "*.pt"))
        data = [torch.load(f) for f in docking_files]

        for i, target in enumerate(data):
            for j, score in enumerate(target["raw"]):
                try:
                    if len(data[i]["raw"][j]["redocked"]["mols"]) == 0:
                        data[i]["raw"][j]["redocked"]["mols"] = [None] * 8
                except:
                    print("hit None - skipping")

        df = pd.DataFrame(
            {
                "generated": [
                    score["raw"][index]["score_only"]["affinity"]
                    for score in data
                    for index in range(len(score["raw"]))
                    if score["raw"][index]["score_only"] is not None
                ],
                "minimized": [
                    score["raw"][index]["minimized"]["min_affinity"]
                    for score in data
                    for index in range(len(score["raw"]))
                    if score["raw"][index]["score_only"] is not None
                ],
                "redocked_best": [
                    score["raw"][index]["redocked"]["affinities"][0]
                    for score in data
                    for index in range(len(score["raw"]))
                    if score["raw"][index]["score_only"] is not None
                ],
                "redocked_all": [
                    score["raw"][index]["redocked"]["affinities"]
                    for score in data
                    for index in range(len(score["raw"]))
                    if score["raw"][index]["score_only"] is not None
                ],
                # mols
                "generated_mol": [
                    score["raw"][index]["score_only"]["mol"]
                    for score in data
                    for index in range(len(score["raw"]))
                    if score["raw"][index]["score_only"] is not None
                ],
                "minimized_mol": [
                    score["raw"][index]["minimized"]["mol"]
                    for score in data
                    for index in range(len(score["raw"]))
                    if score["raw"][index]["score_only"] is not None
                ],
                "redocked_best_mol": [
                    score["raw"][index]["redocked"]["mols"][0]
                    for score in data
                    for index in range(len(score["raw"]))
                    if score["raw"][index]["score_only"] is not None
                ],
                #'redocked_all_mol' : [score['raw'][index]['redocked']['mols'] for score in data for index in range(len(score['raw'])) if score['raw'][index]['score_only'] is not None],
                # rmsds
                "min_rmsd": [
                    score["raw"][index]["minimized"]["min_rmsd"]
                    for score in data
                    for index in range(len(score["raw"]))
                    if score["raw"][index]["score_only"] is not None
                ],
                "target": [
                    score["receptor_path"].split("/")[-1].split(".")[0][:4]
                    for score in data
                    for _ in range(len(score["raw"]))
                    if score["raw"][_]["score_only"] is not None
                ],
            }
        )

        df["method"] = model

        dfs.append(df)

    df = pd.concat(dfs)

    print("Done")
    return df


# DOCKING_DATA = load_docking_data()
DOCKING_DATA = torch.load(os.path.join(DATA_DIR, "docking_data.pt"))
PROTEIN_DATA = torch.load(os.path.join(DATA_DIR, "proteins.pt"))


class RedockedData(BasePosesDataset):
    def __init__(
        self, name: str,
                type: str,
                data_dir: str = DATA_DIR,
                pdbqt_dir: str = PDBQT_DIR,
                docking_data=DOCKING_DATA,
                protein_data=PROTEIN_DATA
    ):
        super().__init__(protein_data=protein_data, data_dir=data_dir, pdbqt_dir=pdbqt_dir)

        self.name = name
        self.type = type
        self.data = docking_data[docking_data["method"] == name]
        assert len(self.data) > 0, "No data found"

    def __len__(self):
        return len(self.names)

    def __repr__(self) -> str:
        return f"{self.name}_{self.type}"

    def load_mols(self, idx: int):
        name = self.names[idx][:4]

        if self.type == "score_only":
            mols = self.data[self.data["target"] == name]["generated_mol"].to_list()
        elif self.type == "minimized":
            mols = self.data[self.data["target"] == name]["minimized_mol"].to_list()
        elif self.type == "redocked":
            mols = self.data[self.data["target"] == name]["redocked_best_mol"].to_list()
        else:
            raise ValueError("Invalid type")

        return [mol for mol in mols if mol is not None]

    def prepare_sdf(self, name: str, idx: int = 0):
        mols = self.load_mols(idx)

        # Remove None values
        mols = [dm.add_hs(mol, add_coords=True) for mol in mols if mol is not None]

        tmp_path = f"tmp_ligands_{self.name}.sdf"
        dm.to_sdf(mols, tmp_path)

        try:
            lig = load_mols_from_sdf(tmp_path)
            os.remove(tmp_path)
            return lig
        except Exception as e:
            print(e)
            os.remove(tmp_path)
            return []
