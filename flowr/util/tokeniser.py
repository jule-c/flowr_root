from __future__ import annotations

import pickle
import threading
from abc import ABC, abstractmethod
from typing import Optional, Union

import flowr.util.functional as smolF

tokenT = Union[str, int]
indicesT = Union[list[int], list[list[int]]]


PICKLE_PROTOCOL = 4


# *** Util functions ***


def _check_unique(obj_list, name="objects"):
    if len(obj_list) != len(set(obj_list)):
        raise RuntimeError(f"{name} cannot contain duplicates")


def _check_type_all(obj_list, exp_type, name="list"):
    for obj in obj_list:
        if not isinstance(obj, exp_type):
            raise TypeError(f"all objects in {name} must be instances of {exp_type}")


# *** Tokeniser Interface ***


class Tokeniser(ABC):
    """Interface for tokeniser classes"""

    @abstractmethod
    def tokenise(self, sentences: list[str]) -> Union[list[str], list[int]]:
        pass

    @classmethod
    @abstractmethod
    def from_vocabulary(cls, vocab: Vocabulary) -> Tokeniser:
        pass


# *** Vocabulary Implementations ***


class Vocabulary:
    """Vocabulary class which maps tokens <--> indices"""

    def __init__(self, tokens: list[tokenT]):
        _check_unique(tokens, "tokens list")
        _check_type_all(tokens, tokenT, "tokens list")

        token_idx_map = {token: idx for idx, token in enumerate(tokens)}
        idx_token_map = {idx: token for idx, token in enumerate(tokens)}

        self.token_idx_map = token_idx_map
        self.idx_token_map = idx_token_map

        # Just to be certain that vocab objects are thread safe
        self._vocab_lock = threading.Lock()

        # So that we can save this object without assuming the above dictionaries are ordered
        self._tokens = tokens

    @property
    def size(self) -> int:
        return len(self)

    def __len__(self) -> int:
        with self._vocab_lock:
            length = len(self.token_idx_map)

        return length

    def contains(self, token: tokenT) -> bool:
        with self._vocab_lock:
            contains = token in self.token_idx_map

        return contains

    def tokens_from_indices(self, indices: list[int]) -> list[tokenT]:
        _check_type_all(indices, int, "indices list")
        with self._vocab_lock:
            tokens = [self.idx_token_map[idx] for idx in indices]

        return tokens

    def indices_from_tokens(
        self, tokens: list[tokenT], one_hot: Optional[bool] = False
    ) -> list[int]:
        _check_type_all(tokens, tokenT, "tokens list")

        with self._vocab_lock:
            indices = [self.token_idx_map[token] for token in tokens]

        if not one_hot:
            return indices

        one_hots = smolF.one_hot_encode(indices, len(self)).tolist()
        return one_hots

    def to_bytes(self) -> bytes:
        with self._vocab_lock:
            obj_bytes = pickle.dumps(self._tokens, protocol=PICKLE_PROTOCOL)

        return obj_bytes

    @staticmethod
    def from_bytes(data: bytes) -> Vocabulary:
        tokens = pickle.loads(data)
        return Vocabulary(tokens)


atom_encoder = {
    "H": 0,
    "Li": 1,
    "B": 2,
    "C": 3,
    "N": 4,
    "O": 5,
    "F": 6,
    "Na": 7,
    "Mg": 8,
    "Al": 9,
    "Si": 10,
    "P": 11,
    "S": 12,
    "Cl": 13,
    "K": 14,
    "Ca": 15,
    "Ti": 16,
    "V": 17,
    "Cr": 18,
    "Mn": 19,
    "Fe": 20,
    "Co": 21,
    "Ni": 22,
    "Cu": 23,
    "Zn": 24,
    "Ge": 25,
    "As": 26,
    "Se": 27,
    "Br": 28,
    "Zr": 29,
    "Mo": 30,
    "Ru": 31,
    "Rh": 32,
    "Pd": 33,
    "Ag": 34,
    "Cd": 35,
    "In": 36,
    "Sn": 37,
    "Sb": 38,
    "Te": 39,
    "I": 40,
    "Ba": 41,
    "Nd": 42,
    "Gd": 43,
    "Yb": 44,
    "Pt": 45,
    "Au": 46,
    "Hg": 47,
    "Pb": 48,
    "Bi": 49,
}
atom_decoder = {v: k for k, v in atom_encoder.items()}
atomic_nb = [
    1,
    3,
    5,
    6,
    7,
    8,
    9,
    11,
    12,
    13,
    14,
    15,
    16,
    17,
    19,
    20,
    22,
    23,
    24,
    25,
    26,
    27,
    28,
    29,
    30,
    32,
    33,
    34,
    35,
    40,
    42,
    44,
    46,
    47,
    48,
    49,
    50,
    51,
    53,
    56,
    60,
    64,
    70,
    78,
    79,
    80,
    82,
    83,
]


# pocket_atom_names = [
#     "N",
#     "CA",
#     "C",
#     "O",
#     "CB",
#     "CG",
#     "CD1",
#     "CD2",
#     "H",
#     "HA",
#     "HB3",
#     "HB2",
#     "HG",
#     "HD11",
#     "HD12",
#     "HD13",
#     "HD21",
#     "HD22",
#     "HD23",
#     "HA3",
#     "HA2",
#     "OG",
#     "CG1",
#     "CG2",
#     "HB",
#     "HG11",
#     "HG12",
#     "HG13",
#     "HG21",
#     "HG22",
#     "HG23",
#     "CE1",
#     "CE2",
#     "CZ",
#     "HD1",
#     "HD2",
#     "HE1",
#     "HE2",
#     "HZ",
#     "CD",
#     "OE1",
#     "OE2",
#     "HG3",
#     "HG2",
#     "CE",
#     "NZ",
#     "HD3",
#     "HE3",
#     "HZ1",
#     "HZ2",
#     "HZ3",
#     "OG1",
#     "HG1",
#     "ND1",
#     "NE2",
#     "OD1",
#     "ND2",
#     "SD",
#     "NE",
#     "NH1",
#     "NH2",
#     "HH22",
#     "HH21",
#     "HH12",
#     "HH11",
#     "HE",
#     "OD2",
#     "NE1",
#     "CE3",
#     "CZ2",
#     "CZ3",
#     "CH2",
#     "HH2",
#     "HE22",
#     "HE21",
#     "HB1",
#     "OH",
#     "HH",
#     "SG",
#     "1HA",
#     "2HA",
#     "3HA",
#     "H1",
#     "H2",
#     "OXT",
#     "HXT",
#     "CH3",
#     "1H",
#     "2H",
#     "3H",
#     "H3",
#     "1HX3",
#     "2HX2",
#     "HG4",
#     "CX",
#     "OQ1",
#     "OQ2",
#     "HQ2",
#     "HA1",
#     "HZ11",
#     "HZ21",
#     "HZ31",
#     "HE11",
#     "HE12",
#     "HD",
#     "P",
#     "O1P",
#     "O2P",
#     "O3P",
#     "H2P",
#     "H3P",
#     "HZ4",
#     "HE4",
#     "1HXT",
#     "2HXT",
#     "3HXT",
#     "OD",
#     "C5'",
#     "C4'",
#     "C2'",
#     "N1",
#     "C2",
#     "C3",
#     "O3",
#     "C4",
#     "C5",
#     "C6",
#     "OP1",
#     "OP2",
#     "OP3",
#     "OP4",
#     "H5'2",
#     "H5'3",
#     "H4'",
#     "H2'1",
#     "H2'2",
#     "H2'3",
#     "H6",
#     "HP2",
#     "HP3",
#     "HE23",
#     "D",
#     "DG",
#     "DE22",
#     "DE21",
#     "DD21",
#     "DD22",
#     "DD1",
#     "DH",
#     "DZ1",
#     "DZ3",
#     "DG1",
#     "DA",
#     "DB2",
#     "DB3",
#     "DG2",
#     "DG3",
#     "DD2",
#     "DE1",
#     "DE2",
#     "DZ",
#     "DD23",
#     "DD13",
#     "DD12",
#     "DD11",
#     "DD3",
#     "DB1",
#     "DB",
#     "DG11",
#     "DG12",
#     "DG13",
#     "DG21",
#     "DG22",
#     "DG23",
#     "DA2",
#     "DA3",
#     "DZ2",
#     "DE3",
#     "HG24",
#     "HA4",
#     "HH13",
#     "HH23",
#     "HH24",
#     "SE",
#     "HH1",
#     "DH2",
#     "HB4",
#     "HD14",
#     "HD24",
#     "D3",
#     "D2",
#     "HH31",
#     "HH32",
#     "HH33",
#     "HD4",
#     "4HXT",
#     "HE13",
#     "HE14",
#     "HE24",
#     "DE",
#     "DH11",
#     "DH12",
#     "DH21",
#     "DH22",
#     "H4",
# ]

pocket_atom_names = ["H", "C", "O", "N", "S", "P", "Se", "F"]

pocket_residue_names = [
    "LEU",
    "GLU",
    "TRP",
    "ASN",
    "VAL",
    "GLN",
    "ARG",
    "SER",
    "GLY",
    "THR",
    "TYR",
    "HIS",
    "ASP",
    "LYS",
    "ILE",
    "ALA",
    "CYS",
    "PRO",
    "PHE",
    "MET",
    "CME",
    "LLP",
    "OAS",
    "SGB",
    "CSD",
    "SEP",
    "OCY",
    "TIS",
    "SCY",
    "OCS",
    "QPA",
    "KPI",
    "PHD",
    "MEN",
    "SUN",
    "TPO",
    "CSO",
    "YCM",
    "UNK",
    "ALY",
    "SVX",
    "PTR",
    "KCX",
    "HOX",
    "PCA",
    "00C",
    "DM0",
    "XCN",
    "SXE",
    "HYP",
    "CSX",
    "CSS",
    "ORN",
    "MLY",
    "2CO",
    "NEP",
    "YOF",
    "SNN",
    "ACE",
    "NMA",
]

pocket_atom_name_encoder = {
    atom: idx + 2 for idx, atom in enumerate(pocket_atom_names)
}  # add 1 to account for padding and 1 for LIG. Add len(res_name_encoder) to account for residues
pocket_atom_name_decoder = {v: k for k, v in pocket_atom_name_encoder.items()}
pocket_res_name_encoder = {
    res: idx + 2 for idx, res in enumerate(pocket_residue_names)
}  # add 1 to account for padding and 1 for LIG
pocket_res_name_decoder = {v: k for k, v in pocket_res_name_encoder.items()}
