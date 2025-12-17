import lightning as L
import torch

from flowr.util.device import dict_to_device, get_device, get_device_string


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

    Args:
        args: Configuration object containing generation parameters.
        model: Trained flow-based molecular generation model.
        prior: Prior distribution samples (noise) for ligand generation.
        posterior: Target data containing ground truth ligand-pocket complexes.
        device (str, optional): PyTorch device for computation. Defaults to auto-detected device.
        save_traj (bool, optional): Whether to save generation trajectory. Defaults to False.
        iter (str, optional): Iteration identifier for seed. Defaults to 0.

    Returns:
        list[rdkit.Chem.Mol]: Generated ligand molecules as RDKit Mol objects with affinity annotations
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

    # Generate RDKit molecules
    gen_ligs = model._generate_mols(output)

    # Attach affinity predictions as properties if present
    assert "affinity" in output, "Affinity predictions not found in output"
    affinity = output["affinity"]
    # Ensure affinity is a dict-like object with keys: pic50, pki, pkd, pec50
    for idx, mol in enumerate(gen_ligs):
        complex = posterior["complex"][idx]
        metadata = complex.metadata
        system_id = metadata["system_id"]

        if mol is None:
            mol = complex.ligand.orig_mol.to_rdkit()
            gen_ligs[idx] = mol
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

    return gen_ligs
