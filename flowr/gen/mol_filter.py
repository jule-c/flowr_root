"""
Molecular property filtering for on-the-fly ligand filtering during generation.

Two filter types are provided:
    1. ``PropertyFilter``  – uses RDKit descriptors (fast, no model needed).
    2. ``ADMEFilter``      – uses pre-trained ML models (arbitrary callables).

Both share the same range-checking logic via ``MolFilter``.

Usage from the CLI (see ``generate_from_pdb.py``):
    --property_filter molwt:200:500 num_h_donors:1:5 logp::3.5
    --adme_filter clearance:0:50:/path/to/model.pt herg:0:0.5:/path/to/model.pt
"""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional

from rdkit import Chem
from rdkit.Chem import Descriptors

import flowr.util.rdkit as smolRD

# ---------------------------------------------------------------------------
# Property registry – maps *all* accepted aliases to the canonical dict key
# ---------------------------------------------------------------------------

# Continuous properties
_CONT_PROPERTY_SUBSET = {
    "MolWt": Descriptors.MolWt,
    "LogP": Descriptors.MolLogP,
    "FractionCSP3": Descriptors.FractionCSP3,
    "TPSA": Descriptors.TPSA,
}

# Discrete properties (all except NumAlerts)
_DISC_PROPERTY_SUBSET = {
    k: v for k, v in smolRD.DISC_PROPERTIES_RDKIT.items() if k != "NumAlerts"
}

# Merge into a single registry
RDKIT_PROPERTY_REGISTRY: Dict[str, Callable] = {
    **_CONT_PROPERTY_SUBSET,
    **_DISC_PROPERTY_SUBSET,
}

# Build a normalised alias lookup: lowercase + strip underscores -> canonical key
_ALIAS_MAP: Dict[str, str] = {}


def _normalise(name: str) -> str:
    """Lowercase and strip underscores / hyphens for fuzzy matching."""
    return name.lower().replace("_", "").replace("-", "")


# Populate alias map from canonical names
for _canonical in RDKIT_PROPERTY_REGISTRY:
    _ALIAS_MAP[_normalise(_canonical)] = _canonical

# Extra hand-written aliases for convenience
_EXTRA_ALIASES: Dict[str, str] = {
    # MolWt
    "molweight": "MolWt",
    "molecularweight": "MolWt",
    "mw": "MolWt",
    # LogP
    "mollogp": "LogP",
    "alogp": "LogP",
    # FractionCSP3
    "fractioncsp3": "FractionCSP3",
    "fsp3": "FractionCSP3",
    "csp3": "FractionCSP3",
    "fraccsp3": "FractionCSP3",
    # TPSA
    "topologicalpsa": "TPSA",
    "polarsurfacearea": "TPSA",
    # Discrete
    "numhacceptors": "NumHAcceptors",
    "numhydrogenacceptors": "NumHAcceptors",
    "hacceptors": "NumHAcceptors",
    "hba": "NumHAcceptors",
    "numhdonors": "NumHDonors",
    "numhydrogendonors": "NumHDonors",
    "hdonors": "NumHDonors",
    "hbd": "NumHDonors",
    "numheteroatoms": "NumHeteroatoms",
    "heteroatoms": "NumHeteroatoms",
    "numrotatatablebonds": "NumRotatableBonds",
    "rotatablebonds": "NumRotatableBonds",
    "nrot": "NumRotatableBonds",
    "numheavyatoms": "NumHeavyAtoms",
    "heavyatoms": "NumHeavyAtoms",
    "nha": "NumHeavyAtoms",
    "numaliphaticcarbocycles": "NumAliphaticCarbocycles",
    "numaliphaticheterocycles": "NumAliphaticHeterocycles",
    "numaliphaticrings": "NumAliphaticRings",
    "numromaticcarbocycles": "NumAromaticCarbocycles",
    "numaromaticcarbocycles": "NumAromaticCarbocycles",
    "numaromaticheterocycles": "NumAromaticHeterocycles",
    "numsaturatedcarbocycles": "NumSaturatedCarbocycles",
    "numsaturatedheterocycles": "NumSaturatedHeterocycles",
    "numaromaticrings": "NumAromaticRings",
    "aromaticrings": "NumAromaticRings",
    "ringcount": "RingCount",
    "numrings": "RingCount",
    "rings": "RingCount",
    "numchiralcenters": "NumChiralCenters",
    "chiralcenters": "NumChiralCenters",
}
for _alias, _canon in _EXTRA_ALIASES.items():
    _ALIAS_MAP[_normalise(_alias)] = _canon


def resolve_property_name(name: str) -> str:
    """Resolve a user-supplied property name to its canonical registry key.

    Matching is case- and underscore-insensitive.  Raises ``ValueError`` if
    no match is found.
    """
    norm = _normalise(name)
    canonical = _ALIAS_MAP.get(norm)
    if canonical is None:
        available = sorted(set(_ALIAS_MAP.values()))
        raise ValueError(
            f"Unknown property '{name}'. Available properties: {available}"
        )
    return canonical


# ---------------------------------------------------------------------------
# Data classes for filter criteria
# ---------------------------------------------------------------------------


@dataclass
class PropertyCriterion:
    """A single property filter: property name + optional min / max bounds."""

    name: str  # canonical property name
    min_val: Optional[float] = None
    max_val: Optional[float] = None

    def in_range(self, value: float) -> bool:
        if self.min_val is not None and value < self.min_val:
            return False
        if self.max_val is not None and value > self.max_val:
            return False
        return True

    def __repr__(self) -> str:
        lo = self.min_val if self.min_val is not None else "-∞"
        hi = self.max_val if self.max_val is not None else "∞"
        return f"{self.name} ∈ [{lo}, {hi}]"


@dataclass
class ADMECriterion:
    """A single ADME filter: property name + range + path to compiled model."""

    name: str
    min_val: Optional[float] = None
    max_val: Optional[float] = None
    model_path: Optional[str] = None

    def in_range(self, value: float) -> bool:
        if self.min_val is not None and value < self.min_val:
            return False
        if self.max_val is not None and value > self.max_val:
            return False
        return True

    def __repr__(self) -> str:
        lo = self.min_val if self.min_val is not None else "-∞"
        hi = self.max_val if self.max_val is not None else "∞"
        return f"ADME({self.name}) ∈ [{lo}, {hi}] model={self.model_path}"


# ---------------------------------------------------------------------------
# Abstract base filter
# ---------------------------------------------------------------------------


class MolFilter(ABC):
    """Base class for molecular filters applied during generation."""

    @abstractmethod
    def __call__(
        self,
        mols: List[Chem.rdchem.Mol],
        **kwargs,
    ) -> List[Chem.rdchem.Mol]:
        """Return the subset of *mols* that pass all criteria."""

    @abstractmethod
    def __len__(self) -> int:
        """Number of active criteria."""

    @abstractmethod
    def __repr__(self) -> str: ...

    @property
    @abstractmethod
    def active(self) -> bool:
        """Whether any criteria are configured."""


# ---------------------------------------------------------------------------
# RDKit property filter
# ---------------------------------------------------------------------------


class PropertyFilter(MolFilter):
    """Filter molecules by RDKit-calculable properties.

    Parameters
    ----------
    criteria : list[PropertyCriterion]
        Each entry specifies a property name and its allowed range.

    Example
    -------
    >>> pf = PropertyFilter.from_strings(["molwt:200:500", "num_h_donors:1:5"])
    >>> filtered = pf(mol_list)
    """

    def __init__(self, criteria: List[PropertyCriterion] | None = None):
        self.criteria: List[PropertyCriterion] = criteria or []
        # Pre-resolve the RDKit callables for speed
        self._callables: Dict[str, Callable] = {}
        for c in self.criteria:
            if c.name not in RDKIT_PROPERTY_REGISTRY:
                raise ValueError(
                    f"Property '{c.name}' is not in the RDKit property registry."
                )
            self._callables[c.name] = RDKIT_PROPERTY_REGISTRY[c.name]

    # -- construction helpers ------------------------------------------------

    @classmethod
    def from_strings(cls, specs: List[str]) -> "PropertyFilter":
        """Parse CLI-style ``"name:min:max"`` strings into a filter.

        Omitted bounds are treated as *no bound* (i.e. ``-inf`` / ``+inf``).
        Examples: ``"molwt:200:500"``, ``"logp::3.5"``, ``"num_h_donors:1:"``.
        """
        criteria: List[PropertyCriterion] = []
        for spec in specs:
            parts = spec.split(":")
            if len(parts) < 2 or len(parts) > 3:
                raise ValueError(
                    f"Property filter spec must be 'name:min:max', got '{spec}'"
                )
            raw_name = parts[0]
            canonical = resolve_property_name(raw_name)
            min_val = float(parts[1]) if len(parts) > 1 and parts[1] else None
            max_val = float(parts[2]) if len(parts) > 2 and parts[2] else None
            criteria.append(PropertyCriterion(canonical, min_val, max_val))
        return cls(criteria)

    # -- filtering -----------------------------------------------------------

    def __call__(
        self,
        mols: List[Chem.rdchem.Mol],
        **kwargs,
    ) -> List[Chem.rdchem.Mol]:
        if not self.criteria:
            return mols

        kept: List[Chem.rdchem.Mol] = []
        for mol in mols:
            if mol is None:
                continue
            if self._passes(mol):
                kept.append(mol)
        return kept

    def _passes(self, mol: Chem.rdchem.Mol) -> bool:
        for criterion in self.criteria:
            try:
                value = self._callables[criterion.name](mol)
            except Exception:
                return False
            if not criterion.in_range(value):
                return False
        return True

    # -- dunder helpers ------------------------------------------------------

    def __len__(self) -> int:
        return len(self.criteria)

    def __repr__(self) -> str:
        inner = ", ".join(str(c) for c in self.criteria)
        return f"PropertyFilter([{inner}])"

    @property
    def active(self) -> bool:
        return len(self.criteria) > 0


# ---------------------------------------------------------------------------
# ADME / ML-model filter
# ---------------------------------------------------------------------------


class ADMEFilter(MolFilter):
    """Filter molecules by predictions from pre-trained ML models.

    This class provides the *interface* for plugging in arbitrary ADME /
    ADMET models.  Each criterion carries a ``model_path`` that points to a
    compiled (or serialised) model artifact.  The actual model loading and
    inference are delegated to ``load_adme_model`` and ``predict_adme`` which
    should be implemented once concrete model formats are decided.

    Parameters
    ----------
    criteria : list[ADMECriterion]
        Each entry specifies property name, range, and model path.

    Example
    -------
    >>> af = ADMEFilter.from_strings(
    ...     ["clearance:0:50:/models/clearance.pt",
    ...      "herg:0:0.5:/models/herg.pt"]
    ... )
    >>> filtered = af(mol_list)
    """

    def __init__(self, criteria: List[ADMECriterion] | None = None):
        self.criteria: List[ADMECriterion] = criteria or []
        self._models: Dict[str, object] = {}
        for c in self.criteria:
            if c.model_path and c.model_path not in self._models:
                self._models[c.model_path] = self._load_model(c.model_path)

    # -- construction helpers ------------------------------------------------

    @classmethod
    def from_strings(cls, specs: List[str]) -> "ADMEFilter":
        """Parse CLI-style ``"name:min:max:model_path"`` strings.

        The model_path is required for each ADME criterion.
        Omitted bounds use empty strings: ``"clearance::50:/path/model.pt"``.
        """
        criteria: List[ADMECriterion] = []
        for spec in specs:
            parts = spec.split(":")
            if len(parts) != 4:
                raise ValueError(
                    f"ADME filter spec must be 'name:min:max:model_path', got '{spec}'"
                )
            name = parts[0]
            min_val = float(parts[1]) if parts[1] else None
            max_val = float(parts[2]) if parts[2] else None
            model_path = parts[3] if parts[3] else None
            criteria.append(ADMECriterion(name, min_val, max_val, model_path))
        return cls(criteria)

    # -- model loading / inference (override points) -------------------------

    @staticmethod
    def _load_model(model_path: str) -> object:
        """Load a pre-trained ADME model from *model_path*.

        Override or monkey-patch this method once the model format is decided.
        Currently returns a placeholder that will raise at inference time if
        actually called, serving as a clear integration point.
        """
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"ADME model not found: {model_path}")

        # ----- INTEGRATION POINT -----
        # Replace the block below with actual model loading logic, e.g.:
        #   import torch
        #   model = torch.jit.load(model_path)
        #   model.eval()
        #   return model
        warnings.warn(
            f"ADME model loading is not yet implemented. "
            f"Returning stub for '{model_path}'. "
            f"Override ADMEFilter._load_model() with your loader.",
            stacklevel=2,
        )

        class _StubModel:
            """Placeholder – replace with real inference."""

            def __init__(self, path: str):
                self.path = path

            def __call__(self, mol: Chem.rdchem.Mol) -> float:
                raise NotImplementedError(
                    f"ADME inference not implemented for model at '{self.path}'. "
                    "Implement ADMEFilter._predict() or replace _load_model()."
                )

        return _StubModel(model_path)

    @staticmethod
    def _predict(model: object, mol: Chem.rdchem.Mol) -> float:
        """Run a single-molecule prediction through *model*.

        Override this with real inference logic.  The default implementation
        assumes the model object is callable with a single mol argument.
        """
        # ----- INTEGRATION POINT -----
        # E.g. for a torch model you might do:
        #   feats = featurise(mol)
        #   with torch.no_grad():
        #       return model(feats).item()
        return model(mol)

    # -- filtering -----------------------------------------------------------

    def __call__(
        self,
        mols: List[Chem.rdchem.Mol],
        **kwargs,
    ) -> List[Chem.rdchem.Mol]:
        if not self.criteria:
            return mols

        kept: List[Chem.rdchem.Mol] = []
        for mol in mols:
            if mol is None:
                continue
            if self._passes(mol):
                kept.append(mol)
        return kept

    def _passes(self, mol: Chem.rdchem.Mol) -> bool:
        for criterion in self.criteria:
            model = self._models.get(criterion.model_path)
            if model is None:
                warnings.warn(
                    f"No model loaded for ADME criterion '{criterion.name}', skipping."
                )
                continue
            try:
                value = self._predict(model, mol)
            except NotImplementedError:
                raise
            except Exception as e:
                warnings.warn(
                    f"ADME prediction failed for '{criterion.name}': {e}. "
                    "Molecule will be excluded."
                )
                return False
            if not criterion.in_range(value):
                return False
        return True

    # -- dunder helpers ------------------------------------------------------

    def __len__(self) -> int:
        return len(self.criteria)

    def __repr__(self) -> str:
        inner = ", ".join(str(c) for c in self.criteria)
        return f"ADMEFilter([{inner}])"

    @property
    def active(self) -> bool:
        return len(self.criteria) > 0


# ---------------------------------------------------------------------------
# Composite convenience
# ---------------------------------------------------------------------------


class MolFilterPipeline:
    """Chain multiple ``MolFilter`` instances and apply them sequentially."""

    def __init__(self, filters: List[MolFilter] | None = None):
        self.filters = [f for f in (filters or []) if f.active]

    def __call__(
        self,
        mols: List[Chem.rdchem.Mol],
        **kwargs,
    ) -> List[Chem.rdchem.Mol]:
        for filt in self.filters:
            mols = filt(mols, **kwargs)
        return mols

    @property
    def active(self) -> bool:
        return len(self.filters) > 0

    def __repr__(self) -> str:
        return f"MolFilterPipeline({self.filters})"
