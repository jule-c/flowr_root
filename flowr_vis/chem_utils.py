"""
Shared chemistry utilities for FLOWR visualization server and worker.
====================================================================
Contains RDKit-based molecule helpers, property computation, and
structural alert definitions used by both ``server.py`` (CPU frontend)
and ``worker.py`` (GPU worker).
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

# ---------------------------------------------------------------------------
# RDKit imports
# ---------------------------------------------------------------------------
try:
    from rdkit import Chem, DataStructs
    from rdkit.Chem import AllChem, Descriptors

    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False

# ---------------------------------------------------------------------------
# Structural alert SMARTS (PAINS-like filters)
# ---------------------------------------------------------------------------
_STRUCTURAL_ALERT_SMARTS = [
    "*1[O,S,N]*1",
    "[S,C](=[O,S])[F,Br,Cl,I]",
    "[CX4][Cl,Br,I]",
    "[#6]S(=O)(=O)O[#6]",
    "[$([CH]),$(CC)]#CC(=O)[#6]",
    "[$([CH]),$(CC)]#CC(=O)O[#6]",
    "n[OH]",
    "[$([CH]),$(CC)]#CS(=O)(=O)[#6]",
    "C=C(C=O)C=O",
    "n1c([F,Cl,Br,I])cccc1",
    "[CH1](=O)",
    "[#8][#8]",
    "[C;!R]=[N;!R]",
    "[N!R]=[N!R]",
    "[#6](=O)[#6](=O)",
    "[#16][#16]",
    "[#7][NH2]",
    "C(=O)N[NH2]",
    "[#6]=S",
    "[$([CH2]),$([CH][CX4]),$(C([CX4])[CX4])]=[$([CH2]),$([CH][CX4]),$(C([CX4])[CX4])]",
    "C1(=[O,N])C=CC(=[O,N])C=C1",
    "C1(=[O,N])C(=[O,N])C=CC=C1",
    "a21aa3a(aa1aaaa2)aaaa3",
    "a31a(a2a(aa1)aaaa2)aaaa3",
    "a1aa2a3a(a1)A=AA=A3=AA=A2",
    "c1cc([NH2])ccc1",
    "[Hg,Fe,As,Sb,Zn,Se,se,Te,B,Si,Na,Ca,Ge,Ag,Mg,K,Ba,Sr,Be,Ti,Mo,Mn,Ru,Pd,Ni,Cu,Au,Cd,"
    + "Al,Ga,Sn,Rh,Tl,Bi,Nb,Li,Pb,Hf,Ho]",
    "I",
    "OS(=O)(=O)[O-]",
    "[N+](=O)[O-]",
    "C(=O)N[OH]",
    "C1NC(=O)NC(=O)1",
    "[SH]",
    "[S-]",
    "c1ccc([Cl,Br,I,F])c([Cl,Br,I,F])c1[Cl,Br,I,F]",
    "c1cc([Cl,Br,I,F])cc([Cl,Br,I,F])c1[Cl,Br,I,F]",
    "[CR1]1[CR1][CR1][CR1][CR1][CR1][CR1]1",
    "[CR1]1[CR1][CR1]cc[CR1][CR1]1",
    "[CR2]1[CR2][CR2][CR2][CR2][CR2][CR2][CR2]1",
    "[CR2]1[CR2]cc[CR2][CR2][CR2]1",
    "[CH2R2]1N[CH2R2][CH2R2][CH2R2][CH2R2][CH2R2]1",
    "[CH2R2]1N[CH2R2][CH2R2][CH2R2][CH2R2][CH2R2][CH2R2]1",
    "C#C",
    "[OR2,NR2]@[CR2]@[CR2]@[OR2,NR2]@[CR2]@[CR2]@[OR2,NR2]",
    "[$([N+R]),$([n+R]),$([N+]=C)][O-]",
    "[#6]=N[OH]",
    "[#6]=NOC=O",
    "[#6](=O)[CX4,CR0X3,O][#6](=O)",
    "c1ccc2c(c1)ccc(=O)o2",
    "[O+,o+,S+,s+]",
    "N=C=O",
    "[NX3,NX4][F,Cl,Br,I]",
    "c1ccccc1OC(=O)[#6]",
    "[CR0]=[CR0][CR0]=[CR0]",
    "[C+,c+,C-,c-]",
    "N=[N+]=[N-]",
    "C12C(NC(N1)=O)CSC2",
    "c1c([OH])c([OH,NH2,NH])ccc1",
    "P",
    "[N,O,S]C#N",
    "C=C=O",
    "[Si][F,Cl,Br,I]",
    "[SX2]O",
    "[SiR0,CR0](c1ccccc1)(c2ccccc2)(c3ccccc3)",
    "O1CCCCC1OC2CCC3CCCCC3C2",
    "N=[CR0][N,n,O,S]",
    "[cR2]1[cR2][cR2]([Nv3X3,Nv4X4])[cR2][cR2][cR2]1[cR2]2[cR2][cR2][cR2]([Nv3X3,Nv4X4])[cR2][cR2]2",
    "C=[C!r]C#N",
    "[cR2]1[cR2]c([N+0X3R0,nX3R0])c([N+0X3R0,nX3R0])[cR2][cR2]1",
    "[cR2]1[cR2]c([N+0X3R0,nX3R0])[cR2]c([N+0X3R0,nX3R0])[cR2]1",
    "[cR2]1[cR2]c([N+0X3R0,nX3R0])[cR2][cR2]c1([N+0X3R0,nX3R0])",
    "[OH]c1ccc([OH,NH2,NH])cc1",
    "c1ccccc1OC(=O)O",
    "[SX2H0][N]",
    "c12ccccc1(SC(S)=N2)",
    "c12ccccc1(SC(=S)N2)",
    "c1nnnn1C=O",
    "s1c(S)nnc1NC=O",
    "S1C=CSC1=S",
    "C(=O)Onnn",
    "OS(=O)(=O)C(F)(F)F",
    "N#CC[OH]",
    "N#CC(=O)",
    "S(=O)(=O)C#N",
    "N[CH2]C#N",
    "C1(=O)NCC1",
    "S(=O)(=O)[O-,OH]",
    "NC[F,Cl,Br,I]",
    "C=[C!r]O",
    "[NX2+0]=[O+0]",
    "[OR0,NR0][OR0,NR0]",
    "C(=O)O[C,H1].C(=O)O[C,H1].C(=O)O[C,H1]",
    "[CX2R0][NX3R0]",
    "c1ccccc1[C;!R]=[C;!R]c2ccccc2",
    "[NX3R0,NX4R0,OR0,SX2R0][CX4][NX3R0,NX4R0,OR0,SX2R0]",
    "[s,S,c,C,n,N,o,O]~[n+,N+](~[s,S,c,C,n,N,o,O])(~[s,S,c,C,n,N,o,O])~[s,S,c,C,n,N,o,O]",
    "[s,S,c,C,n,N,o,O]~[nX3+,NX3+](~[s,S,c,C,n,N])~[s,S,c,C,n,N]",
    "[*]=[N+]=[*]",
    "[SX3](=O)[O-,OH]",
    "N#N",
    "F.F.F.F",
    "[R0;D2][R0;D2][R0;D2][R0;D2]",
    "[cR,CR]~C(=O)NC(=O)~[cR,CR]",
    "C=!@CC=[O,S]",
    "[#6,#8,#16][#6](=O)O[#6]",
    "c[C;R0](=[O,S])[#6]",
    "c[SX2][C;!R]",
    "C=C=C",
    "c1nc([F,Cl,Br,I,S])ncc1",
    "c1ncnc([F,Cl,Br,I,S])c1",
    "c1nc(c2c(n1)nc(n2)[F,Cl,Br,I])",
    "[#6]S(=O)(=O)c1ccc(cc1)F",
    "[15N]",
    "[13C]",
    "[18O]",
    "[34S]",
]

_STRUCTURAL_ALERTS = (
    [
        p
        for p in (Chem.MolFromSmarts(s) for s in _STRUCTURAL_ALERT_SMARTS)
        if p is not None
    ]
    if RDKIT_AVAILABLE
    else []
)

# ---------------------------------------------------------------------------
# Property definitions
# ---------------------------------------------------------------------------
CONT_PROPERTIES_RDKIT = (
    {
        "MolWt": Descriptors.MolWt,
        "LogP": Descriptors.MolLogP,
        "FractionCSP3": Descriptors.FractionCSP3,
        "TPSA": Descriptors.TPSA,
    }
    if RDKIT_AVAILABLE
    else {}
)

DISC_PROPERTIES_RDKIT = (
    {
        "NumHAcceptors": Descriptors.NumHAcceptors,
        "NumHDonors": Descriptors.NumHDonors,
        "NumHeteroatoms": Descriptors.NumHeteroatoms,
        "NumRotatableBonds": Descriptors.NumRotatableBonds,
        "NumHeavyAtoms": Descriptors.HeavyAtomCount,
        "NumAliphaticCarbocycles": Descriptors.NumAliphaticCarbocycles,
        "NumAliphaticHeterocycles": Descriptors.NumAliphaticHeterocycles,
        "NumAliphaticRings": Descriptors.NumAliphaticRings,
        "NumAromaticCarbocycles": Descriptors.NumAromaticCarbocycles,
        "NumAromaticHeterocycles": Descriptors.NumAromaticHeterocycles,
        "NumSaturatedCarbocycles": Descriptors.NumSaturatedCarbocycles,
        "NumSaturatedHeterocycles": Descriptors.NumSaturatedHeterocycles,
        "NumAromaticRings": Descriptors.NumAromaticRings,
        "RingCount": Descriptors.RingCount,
        "NumAlerts": lambda mol: sum(
            1 for alert in _STRUCTURAL_ALERTS if mol.HasSubstructMatch(alert)
        ),
    }
    if RDKIT_AVAILABLE
    else {}
)

ALL_PROPERTY_NAMES = (
    list(CONT_PROPERTIES_RDKIT.keys())
    + list(DISC_PROPERTIES_RDKIT.keys())
    + ["NumChiralCenters"]
)


# ---------------------------------------------------------------------------
# Molecule helpers
# ---------------------------------------------------------------------------


def mol_to_atom_info(mol) -> List[Dict[str, Any]]:
    """Extract per-atom info (idx, symbol, coords, etc.) from an RDKit mol."""
    if mol is None:
        return []
    conf = mol.GetConformer() if mol.GetNumConformers() > 0 else None
    atoms = []
    for atom in mol.GetAtoms():
        info: Dict[str, Any] = {
            "idx": atom.GetIdx(),
            "symbol": atom.GetSymbol(),
            "atomicNum": atom.GetAtomicNum(),
            "degree": atom.GetDegree(),
            "formalCharge": atom.GetFormalCharge(),
            "isAromatic": atom.GetIsAromatic(),
            "numHs": atom.GetTotalNumHs(),
        }
        if conf is not None:
            pos = conf.GetAtomPosition(atom.GetIdx())
            info["x"] = round(pos.x, 4)
            info["y"] = round(pos.y, 4)
            info["z"] = round(pos.z, 4)
        atoms.append(info)
    return atoms


def mol_to_bond_info(mol) -> List[Dict[str, Any]]:
    """Extract per-bond info from an RDKit mol."""
    if mol is None:
        return []
    return [
        {
            "idx": b.GetIdx(),
            "beginAtomIdx": b.GetBeginAtomIdx(),
            "endAtomIdx": b.GetEndAtomIdx(),
            "bondType": str(b.GetBondType()),
            "isAromatic": b.GetIsAromatic(),
        }
        for b in mol.GetBonds()
    ]


def mol_to_sdf_string(mol) -> str:
    """Convert an RDKit mol to an SDF/Mol block string."""
    if mol is None:
        return ""
    return Chem.MolToMolBlock(mol)


def mol_to_smiles(mol) -> str:
    """Convert an RDKit mol to a SMILES string."""
    if mol is None:
        return ""
    try:
        return Chem.MolToSmiles(mol)
    except Exception:
        return ""


def read_ligand_mol(filepath: str):
    """Read a ligand molecule from SDF/MOL/MOL2/PDB format."""
    if not RDKIT_AVAILABLE:
        return None
    ext = Path(filepath).suffix.lower()
    if ext in (".sdf", ".mol"):
        supplier = Chem.SDMolSupplier(filepath, removeHs=False)
        mols = [m for m in supplier if m is not None]
        return mols[0] if mols else None
    elif ext == ".mol2":
        return Chem.MolFromMol2File(filepath, removeHs=False)
    elif ext == ".pdb":
        return Chem.MolFromPDBFile(filepath, removeHs=False)
    return None


def compute_all_properties(mol) -> Dict[str, Any]:
    """Compute all continuous + discrete RDKit properties for a molecule."""
    if mol is None:
        return {}
    props: Dict[str, Any] = {}
    for name, func in CONT_PROPERTIES_RDKIT.items():
        try:
            props[name] = round(func(mol), 3)
        except Exception:
            props[name] = None
    for name, func in DISC_PROPERTIES_RDKIT.items():
        try:
            props[name] = int(func(mol))
        except Exception:
            props[name] = None
    try:
        props["NumChiralCenters"] = len(
            Chem.FindMolChiralCenters(mol, includeUnassigned=True)
        )
    except Exception:
        props["NumChiralCenters"] = None
    return props


def mol_to_morgan_fp(mol, radius: int = 2, n_bits: int = 2048) -> Optional[np.ndarray]:
    """Compute Morgan fingerprint as a numpy bit vector.

    Uses the modern ``rdFingerprintGenerator`` API (RDKit ≥ 2022.09).
    Falls back to the legacy ``AllChem.GetMorganFingerprintAsBitVect``
    for older RDKit versions.
    """
    if mol is None:
        return None
    try:
        from rdkit.Chem import rdFingerprintGenerator

        gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
        fp = gen.GetFingerprintAsNumPy(mol)
        return fp.astype(np.int8)
    except (ImportError, AttributeError):
        # Fallback for older RDKit versions without rdFingerprintGenerator
        try:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
            arr = np.zeros(n_bits, dtype=np.int8)
            DataStructs.ConvertToNumpyArray(fp, arr)
            return arr
        except Exception:
            return None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# SDF affinity property crawler
# ---------------------------------------------------------------------------
import math
import re as _re

# Mapping from assay-type keywords (case-insensitive) to canonical p-value names.
# Keys are regex patterns matched against SDF property tag names.
# Use lookarounds that treat underscores as separators (unlike \b which
# considers _ a word character).  `(?<![A-Za-z0-9])` = no alphanumeric before,
# `(?![A-Za-z0-9])` = no alphanumeric after.  This ensures e.g. "IC50_nM",
# "r_i_Kd", "r_exp_dg" all match correctly.
_WB_L = r"(?<![A-Za-z0-9])"  # left word boundary (underscore-safe)
_WB_R = r"(?![A-Za-z0-9])"  # right word boundary (underscore-safe)

_AFFINITY_ASSAY_PATTERNS: List[tuple] = [
    # Direct p-converted values (already -log10 scale)
    (_re.compile(_WB_L + r"pIC50" + _WB_R, _re.I), "pIC50", None),
    (_re.compile(_WB_L + r"pKi" + _WB_R, _re.I), "pKi", None),
    (_re.compile(_WB_L + r"pKd" + _WB_R, _re.I), "pKd", None),
    (_re.compile(_WB_L + r"pEC50" + _WB_R, _re.I), "pEC50", None),
    # Raw values (need unit-aware conversion to p-values)
    (_re.compile(_WB_L + r"IC50" + _WB_R, _re.I), "pIC50", "IC50"),
    (_re.compile(_WB_L + r"Ki" + _WB_R, _re.I), "pKi", "Ki"),
    (_re.compile(_WB_L + r"Kd" + _WB_R, _re.I), "pKd", "Kd"),
    (_re.compile(_WB_L + r"EC50" + _WB_R, _re.I), "pEC50", "EC50"),
    # Experimental dG (binding free energy in kcal/mol)
    (
        _re.compile(
            _WB_L + r"(?:exp[_\s]*)?dG" + _WB_R + r"|delta[_\s]*G|binding[_\s]*energy",
            _re.I,
        ),
        "dG",
        "dG",
    ),
]

# Unit multiplier → molar. E.g. "nM" → 1e-9
_UNIT_TO_MOLAR: Dict[str, float] = {
    "M": 1.0,
    "mM": 1e-3,
    "uM": 1e-6,
    "µM": 1e-6,
    "um": 1e-6,
    "nM": 1e-9,
    "nm": 1e-9,
    "pM": 1e-12,
    "pm": 1e-12,
    "fM": 1e-15,
}

# Regex to extract units from a property-tag name.  Matches e.g. "(nM)", "[uM]", "nM", "µM".
# Bare 'M' is anchored so it can't match inside words like "Method".
_UNIT_RE = _re.compile(
    r"[\(\[]*\s*(fM|pM|nM|[uµ]M|mM|(?<![A-Za-z])M(?![A-Za-z]))\s*[\)\]]*"
)


def _detect_unit_from_tag(tag_name: str) -> Optional[float]:
    """Try to extract a concentration unit from an SDF tag name.

    Returns the multiplier to convert to molar, or None if undetected.
    """
    m = _UNIT_RE.search(tag_name)
    if m:
        unit_str = m.group(1)
        # Normalise: case-sensitive lookup after mapping common variants
        for key, val in _UNIT_TO_MOLAR.items():
            if unit_str.lower() == key.lower():
                return val
    return None


def _to_pvalue(raw_val: float, unit_multiplier: float) -> Optional[float]:
    """Convert a raw concentration value + unit multiplier to a p-value.

    p-value = -log10(value_in_molar)
    """
    molar = raw_val * unit_multiplier
    if molar <= 0:
        return None
    return -math.log10(molar)


def crawl_sdf_affinity(filepath: str) -> Optional[Dict[str, Any]]:
    """Crawl an SDF file for affinity/potency properties on the first molecule.

    Scans all SDF property tags for recognisable assay types (IC50, Ki, Kd,
    EC50, pIC50, pKi, pKd, pEC50, dG) and attempts to detect units from the
    tag name.

    Returns
    -------
    dict or None
        ``{"p_label": "pIC50", "p_value": 7.23, "raw_tag": "IC50 (nM)",
           "raw_value": 58.9, "assay_type": "IC50", "unit": "nM"}``
        or None if no affinity property is confidently identified.
    """
    ext = Path(filepath).suffix.lower()
    if ext not in (".sdf", ".mol"):
        return None  # Only SDF/MOL files carry properties
    if not RDKIT_AVAILABLE:
        return None

    try:
        supplier = Chem.SDMolSupplier(filepath, removeHs=False, sanitize=False)
    except Exception:
        return None

    mol = None
    for m in supplier:
        if m is not None:
            mol = m
            break
    if mol is None:
        return None

    # Collect all property names on this molecule
    prop_names = list(mol.GetPropsAsDict().keys())
    if not prop_names:
        return None

    # Try each assay pattern against each property tag
    for pattern, p_label, raw_assay in _AFFINITY_ASSAY_PATTERNS:
        for tag in prop_names:
            if not pattern.search(tag):
                continue

            # Read the value
            try:
                raw_val = float(mol.GetProp(tag))
            except (ValueError, TypeError, KeyError):
                continue

            # Already a p-value (pIC50, pKi, etc.)
            if raw_assay is None:
                return {
                    "p_label": p_label,
                    "p_value": round(raw_val, 4),
                    "raw_tag": tag,
                    "raw_value": raw_val,
                    "assay_type": p_label,  # e.g. "pIC50"
                    "unit": None,
                }

            # dG (kcal/mol) → pKd via dG = RT ln(Kd) → Kd = exp(dG / RT)
            if raw_assay == "dG":
                RT = 0.592  # kcal/mol at 298K
                try:
                    kd_molar = math.exp(raw_val / RT)
                    if kd_molar <= 0:
                        continue
                    p_val = -math.log10(kd_molar)
                    return {
                        "p_label": "pKd",
                        "p_value": round(p_val, 4),
                        "raw_tag": tag,
                        "raw_value": round(raw_val, 4),
                        "assay_type": "dG (kcal/mol)",
                        "unit": "kcal/mol",
                    }
                except (OverflowError, ValueError):
                    continue

            # Raw concentration → p-value
            unit_mult = _detect_unit_from_tag(tag)
            if unit_mult is None:
                # Try common heuristic: very small values (< 1) → likely molar;
                # values 0.001–1000 → likely µM; values > 1000 → likely nM.
                # We SKIP if we can't clearly determine units (per the spec).
                continue

            p_val = _to_pvalue(raw_val, unit_mult)
            if p_val is None:
                continue

            # Determine display unit string
            unit_str = None
            for key, val in _UNIT_TO_MOLAR.items():
                if abs(val - unit_mult) < 1e-20:
                    unit_str = key
                    break

            return {
                "p_label": p_label,
                "p_value": round(p_val, 4),
                "raw_tag": tag,
                "raw_value": round(raw_val, 6),
                "assay_type": raw_assay,
                "unit": unit_str,
            }

    return None
