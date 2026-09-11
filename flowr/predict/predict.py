import warnings

import lightning as L
import torch
from rdkit import Chem

from flowr.util.device import dict_to_device, get_device, get_device_string


def _perceive_from_3d(mol):
    """Sanitise and re-perceive stereochemistry from the 3D structure.

    ``GeometricMol.to_rdkit()`` returns an unsanitised molecule, and RDKit then writes
    stereogenic double bonds to the SDF with the "either" flag (3). Every reader takes
    that flag as an explicit statement that the configuration is unknown, so the
    stereochemistry the coordinates encode is lost on round-trip (e.g. the E/Z of a
    tamoxifen-like C=C). Sanitising and re-perceiving from the coordinates - which are
    the input coordinates - clears the flag and keeps the SDF faithful.

    Falls back to the unperceived molecule if sanitisation fails, so a result is never
    lost to this step.
    """
    try:
        perceived = Chem.Mol(mol)
        Chem.SanitizeMol(perceived)
        if perceived.GetNumConformers() > 0 and perceived.GetConformer().Is3D():
            Chem.AssignStereochemistryFrom3D(perceived)
        return perceived
    except Exception:
        return mol


def input_ligand_to_rdkit(system, builder=None):
    """Return the ligand that was submitted for scoring as an RDKit molecule.

    Affinity prediction does not generate a ligand: the model is conditioned on the
    real ligand at t ~= 1 and only predicts its affinity. The molecule written to the
    SDF must therefore be the ligand the user supplied - otherwise a predicted
    affinity would be attributed to a structure the model never scored.

    Two sources are tried, both of which describe the *input* ligand:
      1. ``ligand.orig_mol`` - the untouched input molecule (hydrogens included,
         coordinates in the input frame), stored during featurisation.
      2. The featurised ligand itself. ``complex_transform`` only one-hot encodes the
         input graph and shifts the complex to the pocket centre of mass, so undoing
         that shift recovers the input molecule in the input coordinate frame.

    Args:
        system: PocketComplex the affinity was predicted for.
        builder: MolBuilder providing the vocabularies needed for source (2).

    Returns:
        rdkit.Chem.Mol | None: The input ligand, or None if it cannot be recovered.
    """
    ligand = system.ligand

    orig_mol = getattr(ligand, "orig_mol", None)
    if orig_mol is not None:
        try:
            mol = orig_mol.to_rdkit()
        except Exception:
            mol = None
        if mol is not None:
            return _perceive_from_3d(mol)

    # orig_mol is only populated when hydrogens are stripped from the ligand, so fall
    # back to the featurised representation of the same molecule, which is a lossless
    # (one-hot) encoding of the input graph.
    if builder is not None:
        try:
            coords = ligand.coords
            com = getattr(system, "com", None)
            if com is not None:
                coords = coords + com.to(coords.device)
            mol = ligand._copy_with(coords=coords).to_rdkit(
                builder.vocab,
                builder.vocab_charges,
                builder.vocab_hybridization,
            )
        except Exception:
            mol = None
        if mol is not None:
            return _perceive_from_3d(mol)

    return None


def predict_affinity_batch(
    args,
    model,
    prior,
    posterior,
    noise_scale: float = 0.1,
    eps=1e-4,
    seed: int = 42,
    device=None,
):
    """Predict ligand affinity for a batch of protein-ligand complexes.

    The returned molecules are the *input* ligands - the ones the model was
    conditioned on and therefore the ones the predicted affinities belong to -
    annotated with the predicted (and, if known, experimental) affinity values.

    Args:
        args: Configuration object containing generation parameters.
        model: Trained flow-based molecular generation model.
        prior: Prior distribution samples (noise) for ligand generation.
        posterior: Target data containing ground truth ligand-pocket complexes.
        device (str, optional): PyTorch device for computation. Defaults to auto-detected device.
        save_traj (bool, optional): Whether to save generation trajectory. Defaults to False.
        iter (str, optional): Iteration identifier for seed. Defaults to 0.

    Returns:
        list[rdkit.Chem.Mol]: Input ligand molecules as RDKit Mol objects with affinity annotations
    """
    # Use auto-detected device if not specified
    if device is None:
        device = get_device()

    # Seed for reproducibility
    torch.manual_seed(seed)
    L.seed_everything(args.seed)

    # Get Ligand prior
    lig_prior = model.builder.extract_ligand_from_complex(prior)
    lig_prior["interactions"] = prior["interactions"]
    lig_prior["fragment_mask"] = prior["fragment_mask"]
    lig_prior = dict_to_device(lig_prior, device)

    # Get ligand and add noise to ligand coordinates
    lig_data = model.builder.extract_ligand_from_complex(posterior)
    lig_data["interactions"] = posterior["interactions"]
    lig_data["fragment_mask"] = posterior["fragment_mask"]
    lig_data["fragment_mode"] = posterior["fragment_mode"]
    noise = torch.randn_like(lig_data["coords"])
    lig_data["coords"] = lig_data["coords"] + noise * noise_scale
    lig_data = dict_to_device(lig_data, device)

    # Get pocket data
    pocket_data = model.builder.extract_pocket_from_complex(posterior)
    pocket_data["interactions"] = posterior["interactions"]
    pocket_data["complex"] = posterior["complex"]
    pocket_data = dict_to_device(pocket_data, device)

    # Build starting times for the integrator
    ## In affinity mode use the provided reference data and only add a bit of noise to the coordinates
    lig_times_cont = torch.ones(posterior["coords"].size(0), device=device) - eps
    lig_times_disc = torch.ones(posterior["coords"].size(0), device=device) - eps
    pocket_times = torch.ones(pocket_data["coords"].size(0), device=device)
    prior_times = [lig_times_cont, lig_times_disc, pocket_times]

    # Run generation N times
    output = model._predict_affinity(
        lig_prior,
        ligand_data=lig_data,
        pocket_data=pocket_data,
        times=prior_times,
    )

    # Attach affinity predictions as properties if present
    assert "affinity" in output, "Affinity predictions not found in output"
    affinity = output["affinity"]

    # Collect the scored ligands.
    # NOTE: the molecules written out must be the ligands that were scored, i.e. the
    # input ligands. Decoding the model's own atom-type and bond predictions
    # (model._generate_mols) is a lossy round-trip of the input here - it can drop
    # stereochemistry or even yield a different molecular graph - which would attach
    # the predicted affinity to a molecule the model never saw.
    systems = posterior["complex"]
    decoded_ligs = None  # only decoded if an input ligand cannot be recovered

    # Ensure affinity is a dict-like object with keys: pic50, pki, pkd, pec50
    gen_ligs = []
    for idx, system in enumerate(systems):
        metadata = system.metadata
        system_id = metadata.get("system_id")

        mol = input_ligand_to_rdkit(system, builder=getattr(model, "builder", None))
        if mol is None:
            # Should not happen for any supported input, but do not drop the result
            # silently: fall back to the model's reconstruction and warn that the
            # written structure may not be the one that was scored.
            if decoded_ligs is None:
                # The decode of the ligand the affinity is being predicted FOR.
                # Repairing it would silently change the structure the score is
                # reported against, so it is left exactly as the model decoded it.
                decoded_ligs = model._generate_mols(output, valence_repair=False)
            mol = decoded_ligs[idx] if idx < len(decoded_ligs) else None
            warnings.warn(
                f"Could not recover the input ligand for system {system_id}. Falling "
                "back to the model's reconstruction - the structure written to the "
                "SDF may differ from the ligand the affinity was predicted for.",
                RuntimeWarning,
                stacklevel=2,
            )
        if mol is None:
            warnings.warn(
                f"No ligand structure could be written for system {system_id}. "
                "Skipping this system.",
                RuntimeWarning,
                stacklevel=2,
            )
            continue

        mol.SetProp("_Name", str(system_id))
        if "pic50" in metadata:
            exp_pic50 = metadata["pic50"]
            mol.SetProp("exp_pic50", str(exp_pic50))
        if "pkd" in metadata:
            exp_pkd = metadata["pkd"]
            mol.SetProp("exp_pkd", str(exp_pkd))
        if "pki" in metadata:
            exp_pki = metadata["pki"]
            mol.SetProp("exp_pki", str(exp_pki))
        if "pec50" in metadata:
            exp_pec50 = metadata["pec50"]
            mol.SetProp("exp_pec50", str(exp_pec50))

        for key in ["pic50", "pki", "pkd", "pec50"]:
            value = affinity[key]
            # If value is a tensor, get the scalar for this molecule
            if hasattr(value, "detach"):
                val = value[idx].item() if value.ndim > 0 else value.item()
            else:
                val = value[idx] if isinstance(value, (list, tuple)) else value
            mol.SetProp(key, str(val))

        gen_ligs.append(mol)

    return gen_ligs
