from pathlib import Path

import torch

import flowr.util.rdkit as smolRD
from flowr.util.device import dict_to_device, get_device


def generate_molecules(
    args,
    model,
    prior,
    device=None,
    save_traj=False,
    iter="",
):
    """
    Generate molecules

    This function performs molecular generation by flowing from a prior distribution to the
    target data distribution
    Args:
        args: Configuration object containing generation parameters including:
            - arch: Model architecture ("flowr" or "transformer")
            - integration_steps: Number of ODE integration steps for generation
            - ode_sampling_strategy: Strategy for ODE sampling ("euler", "heun", etc.)
            - solver: ODE solver type for numerical integration
            - corrector_iters: Number of corrector iterations for predictor-corrector methods
            - save_dir: Directory path for saving generated structures
        model: Trained flow-based molecular generation model with methods:
            - _generate(): Core generation method using flow matching
            - _generate_mols(): Convert generated coordinates to RDKit molecules
        prior: Prior distribution samples (noise) for ligand generation
        posterior: Target data containing ground truth ligand-pocket complexes
        device (str, optional): PyTorch device for computation. Defaults to auto-detected device.
        save_traj (bool, optional): Whether to save generation trajectory. Defaults to False.
        iter (str, optional): Iteration identifier for file naming. Defaults to "".

    Returns:
        list[rdkit.Chem.Mol]: Generated ligand molecules as RDKit Mol objects
    """
    # Use auto-detected device if not specified
    if device is None:
        device = get_device()

    # Get molecule data
    prior = {k: v.to(device) if torch.is_tensor(v) else v for k, v in prior.items()}
    # Build starting times for the integrator
    lig_times_cont = torch.zeros(prior["coords"].size(0), device=device)
    lig_times_disc = torch.zeros(prior["coords"].size(0), device=device)
    prior_times = [lig_times_cont, lig_times_disc]

    # Run generation N times
    output = model._generate(
        prior,
        times=prior_times,
        steps=args.integration_steps,
        strategy=args.ode_sampling_strategy,
        solver=args.solver,
        corr_iters=args.corrector_iters,
        save_traj=save_traj,
        iter=iter,
    )

    # Generate RDKit molecules
    gen_ligs = model._generate_mols(output)
    return gen_ligs


def generate_ligands_per_target(
    args,
    model,
    prior,
    posterior,
    pocket_noise="fix",
    device=None,
    save_traj=False,
    iter="",
    guidance_params: dict | None = None,
):
    """
    Generate ligands for a specific protein target using flow-based molecular generation.

    This function performs molecular generation by flowing from a prior distribution to the
    target data distribution, conditioning on protein pocket information. It supports both
    rigid and flexible pocket conformations depending on the model architecture.
    If inpainting is enabled, the function will also handle masked regions in the ligand
    during generation, where the prior is modified accordingly (masked prior).

    Args:
        args: Configuration object containing generation parameters including:
            - arch: Model architecture ("pocket_flex" for flexible pocket models)
            - integration_steps: Number of ODE integration steps for generation
            - ode_sampling_strategy: Strategy for ODE sampling ("euler", "heun", etc.)
            - solver: ODE solver type for numerical integration
            - corrector_iters: Number of corrector iterations for predictor-corrector methods
            - save_dir: Directory path for saving generated structures
        model: Trained flow-based molecular generation model with methods:
            - builder.extract_ligand_from_complex(): Extract ligand data from complex
            - builder.extract_pocket_from_complex(): Extract pocket data from complex
            - _generate(): Core generation method using flow matching
            - _generate_mols(): Convert generated coordinates to RDKit molecules
            - retrieve_pdbs(): Save generated structures as PDB files (for flexible models)
        prior: Prior distribution samples (noise) for ligand generation
        posterior: Target data containing ground truth ligand-pocket complexes
        pocket_noise: Type of pocket noise ("apo", "holo", "random")
        device (str, optional): PyTorch device for computation. Defaults to auto-detected device.
        save_traj (bool, optional): Whether to save generation trajectory. Defaults to False.
        iter (str, optional): Iteration identifier for file naming. Defaults to "".

    Returns:
        For rigid pocket models:
            list[rdkit.Chem.Mol]: Generated ligand molecules as RDKit Mol objects

        For flexible pocket models ("pocket_flex"):
            tuple: (generated_ligands, generated_pdbs) where:
                - generated_ligands: list[rdkit.Chem.Mol] - Generated ligand molecules
                - generated_pdbs: list[str] - Paths to generated PDB files with pocket conformations
    """
    # Use auto-detected device if not specified
    if device is None:
        device = get_device()

    assert (
        len(set([cmpl.metadata["system_id"] for cmpl in posterior["complex"]])) == 1
    ), "Sampling N ligands per target, but found mixed targets"

    # Get ligand and pocket data
    lig_prior = model.builder.extract_ligand_from_complex(prior)
    lig_prior["interactions"] = prior["interactions"]
    lig_prior["fragment_mask"] = prior["fragment_mask"]
    lig_prior["fragment_mode"] = prior["fragment_mode"]
    lig_prior = dict_to_device(lig_prior, device)
    if args.arch == "pocket_flex" and pocket_noise == "apo":
        pocket_prior = model.builder.extract_pocket_from_complex(prior)
    else:
        pocket_prior = model.builder.extract_pocket_from_complex(posterior)
    pocket_prior["interactions"] = posterior["interactions"]
    pocket_prior["complex"] = posterior["complex"]
    pocket_prior = dict_to_device(pocket_prior, device)

    # Build starting times for the integrator
    lig_times_cont = torch.zeros(prior["coords"].size(0), device=device)
    lig_times_disc = torch.zeros(prior["coords"].size(0), device=device)
    pocket_times = torch.zeros(pocket_prior["coords"].size(0), device=device)
    prior_times = [lig_times_cont, lig_times_disc, pocket_times]

    # Run generation N times
    if guidance_params is None:
        guidance_params = {}
        guidance_params["apply_guidance"] = False
        guidance_params["window_start"] = 0.0
        guidance_params["window_end"] = 0.4
        guidance_params["value_key"] = "affinity"
        guidance_params["subvalue_key"] = "pic50"
        guidance_params["mu"] = 8.0
        guidance_params["sigma"] = 2.0
        guidance_params["maximize"] = True
        guidance_params["coord_noise_level"] = 0.2

    output = model._generate(
        lig_prior,
        pocket_prior,
        times=prior_times,
        steps=args.integration_steps,
        strategy=args.ode_sampling_strategy,
        solver=args.solver,
        corr_iters=args.corrector_iters,
        save_traj=save_traj,
        iter=iter,
        apply_guidance=guidance_params["apply_guidance"],
        guidance_window_start=guidance_params["window_start"],
        guidance_window_end=guidance_params["window_end"],
        value_key=guidance_params["value_key"],
        mu=guidance_params["mu"],
        sigma=guidance_params["sigma"],
        maximize=guidance_params["maximize"],
        coord_noise_level=guidance_params["coord_noise_level"],
    )

    # Generate RDKit molecules
    gen_ligs = model._generate_mols(output)

    # Attach affinity predictions as properties if present
    if "affinity" in output:
        affinity = output["affinity"]
        if affinity is not None:
            # Ensure affinity is a dict-like object with keys: pic50, pki, pkd, pec50
            for mol, idx in zip(gen_ligs, range(len(gen_ligs))):
                if mol is not None:
                    metadata = posterior["complex"][0].metadata
                    system_id = metadata["system_id"]
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
                            val = (
                                value[idx]
                                if isinstance(value, (list, tuple))
                                else value
                            )
                        mol.SetProp(key, str(val))

    if args.arch == "pocket_flex":
        pocket_prior = {
            k: v.cpu().detach() if torch.is_tensor(v) else v
            for k, v in pocket_prior.items()
        }
        gen_pdbs = model.retrieve_pdbs(
            pocket_prior,
            coords=output["pocket_coords"],
            save_dir=Path(args.save_dir) / "gen_pdbs",
            iter=iter,
        )
        return gen_ligs, gen_pdbs
    return gen_ligs


def generate_n_ligands(args, hparams, model, batch, batch_idx=0):
    prior, data, interpolated, _ = batch
    device = get_device()

    assert (
        len(set([cmpl.metadata["system_id"] for cmpl in data["complex"]])) == 1
    ), "Sampling N ligands per target, but found mixed targets"

    # get ligand and pocket data
    lig_prior = model.builder.extract_ligand_from_complex(prior)
    lig_prior["interactions"] = prior["interactions"]
    lig_prior["fragment_mask"] = prior["fragment_mask"]
    lig_prior = dict_to_device(lig_prior, device)
    if args.arch == "pocket_flex" and hparams["pocket_noise"] == "apo":
        pocket_prior = model.builder.extract_pocket_from_complex(prior)
    else:
        pocket_prior = model.builder.extract_pocket_from_complex(data)
    pocket_prior["interactions"] = data["interactions"]
    pocket_prior["complex"] = data["complex"]
    pocket_prior = dict_to_device(pocket_prior, device)

    # specify times
    lig_times = torch.zeros(prior["coords"].size(0), device=device)
    pocket_times = torch.zeros(pocket_prior["coords"].size(0), device=device)
    interaction_times = torch.zeros(prior["coords"].size(0), device=device)
    prior_times = [lig_times, pocket_times, interaction_times]

    k = 0
    num_ligands = 0
    all_gen_ligs = []
    while (
        num_ligands < args.sample_n_molecules_per_target and k <= args.max_sample_iter
    ):
        print(
            f"Sampling iteration {k + 1} for target {data['complex'][0].metadata['system_id']}...",
            end="\r",
        )
        # run generation N times
        output = model._generate(
            lig_prior,
            pocket_prior,
            times=prior_times,
            steps=args.integration_steps,
            strategy=args.ode_sampling_strategy,
            solver=args.solver,
            corr_iters=args.corrector_iters,
            save_traj=False,
            iter=iter,
        )
        # generate molecules
        gen_ligs = model._generate_mols(output)
        if args.filter_valid_unique:
            gen_ligs = smolRD.sanitize_list(
                gen_ligs,
                filter_uniqueness=True,
            )
        all_gen_ligs.extend(gen_ligs)
        num_ligands += len(gen_ligs)
        k += 1

    if num_ligands == 0:
        print(
            f"Reached {args.max_sample_iter} sampling iterations, but could not find any ligands. Skipping."
        )
    elif num_ligands < args.sample_n_molecules_per_target:
        print(
            f"FYI: Reached {args.max_sample_iter} sampling iterations, but could only find {num_ligands} ligands."
        )
    elif num_ligands > args.sample_n_molecules_per_target:
        all_gen_ligs = all_gen_ligs[: args.sample_n_molecules_per_target]

    # CREATE ground truth PDB files
    ref_ligs = model._generate_ligs(
        data, lig_mask=data["lig_mask"].bool(), scale=model.coord_scale
    )[0]
    ref_ligs_with_hs = model.retrieve_ligs_with_hs(data, save_idx=0)
    ref_pdbs = model.retrieve_pdbs(
        data, save_dir=Path(args.save_dir) / "ref_pdbs", save_idx=0
    )
    ref_pdbs_with_hs = model.retrieve_pdbs_with_hs(
        data, save_dir=Path(args.save_dir) / "ref_pdbs", save_idx=0
    )

    return {
        "gen_ligs": all_gen_ligs,
        "ref_ligs": ref_ligs,
        "ref_ligs_with_hs": ref_ligs_with_hs,
        "ref_pdbs": ref_pdbs,
        "ref_pdbs_with_hs": ref_pdbs_with_hs,
    }


def generate_ligands_per_target_selective(
    args,
    model,
    prior,
    posterior_target,
    posterior_untarget,
    pocket_noise="fix",
    device=None,
    save_traj=False,
    iter="",
    guidance_params: dict | None = None,
):
    """
    Generate ligands for a specific protein target using flow-based molecular generation.

    This function performs molecular generation by flowing from a prior distribution to the
    target data distribution, conditioning on protein pocket information. It supports both
    rigid and flexible pocket conformations depending on the model architecture.
    If inpainting is enabled, the function will also handle masked regions in the ligand
    during generation, where the prior is modified accordingly (masked prior).

    Args:
        args: Configuration object containing generation parameters including:
            - arch: Model architecture ("pocket_flex" for flexible pocket models)
            - integration_steps: Number of ODE integration steps for generation
            - ode_sampling_strategy: Strategy for ODE sampling ("euler", "heun", etc.)
            - solver: ODE solver type for numerical integration
            - corrector_iters: Number of corrector iterations for predictor-corrector methods
            - save_dir: Directory path for saving generated structures
        model: Trained flow-based molecular generation model with methods:
            - builder.extract_ligand_from_complex(): Extract ligand data from complex
            - builder.extract_pocket_from_complex(): Extract pocket data from complex
            - _generate(): Core generation method using flow matching
            - _generate_mols(): Convert generated coordinates to RDKit molecules
            - retrieve_pdbs(): Save generated structures as PDB files (for flexible models)
        prior: Prior distribution samples (noise) for ligand generation
        posterior: Target data containing ground truth ligand-pocket complexes
        pocket_noise: Type of pocket noise ("apo", "holo", "random")
        device (str, optional): PyTorch device for computation. Defaults to auto-detected device.
        save_traj (bool, optional): Whether to save generation trajectory. Defaults to False.
        iter (str, optional): Iteration identifier for file naming. Defaults to "".

    Returns:
        For rigid pocket models:
            list[rdkit.Chem.Mol]: Generated ligand molecules as RDKit Mol objects

        For flexible pocket models ("pocket_flex"):
            tuple: (generated_ligands, generated_pdbs) where:
                - generated_ligands: list[rdkit.Chem.Mol] - Generated ligand molecules
                - generated_pdbs: list[str] - Paths to generated PDB files with pocket conformations
    """
    # Use auto-detected device if not specified
    if device is None:
        device = get_device()

    assert (
        len(set([cmpl.metadata["system_id"] for cmpl in posterior_target["complex"]]))
        == 1
    ), "Sampling N ligands per target, but found mixed targets"

    # Ligand data
    lig_prior = model.builder.extract_ligand_from_complex(prior)
    lig_prior["interactions"] = prior["interactions"]
    lig_prior["fragment_mask"] = prior["fragment_mask"]
    lig_prior["fragment_mode"] = prior["fragment_mode"]
    lig_prior = dict_to_device(lig_prior, device)

    # Pocket data target
    if args.arch == "pocket_flex" and pocket_noise == "apo":
        pocket_data_target = model.builder.extract_pocket_from_complex(prior)
    else:
        pocket_data_target = model.builder.extract_pocket_from_complex(posterior_target)
    pocket_data_target["interactions"] = posterior_target["interactions"]
    pocket_data_target["complex"] = posterior_target["complex"]
    pocket_data_target = dict_to_device(pocket_data_target, device)

    # Pocket data off-target
    if args.arch == "pocket_flex" and pocket_noise == "apo":
        pocket_data_untarget = model.builder.extract_pocket_from_complex(prior)
    else:
        pocket_data_untarget = model.builder.extract_pocket_from_complex(
            posterior_untarget
        )
    pocket_data_untarget["interactions"] = posterior_untarget["interactions"]
    pocket_data_untarget["complex"] = posterior_untarget["complex"]
    pocket_data_untarget = dict_to_device(pocket_data_untarget, device)

    # Build starting times for the integrator
    lig_times_cont = torch.zeros(prior["coords"].size(0), device=device)
    lig_times_disc = torch.zeros(prior["coords"].size(0), device=device)
    pocket_times = torch.zeros(pocket_data_target["coords"].size(0), device=device)
    prior_times = [lig_times_cont, lig_times_disc, pocket_times]

    # Run generation N times
    if guidance_params is None:
        guidance_params = {}
        guidance_params["apply_guidance"] = False
        guidance_params["window_start"] = 0.0
        guidance_params["window_end"] = 0.4
        guidance_params["value_key"] = "affinity"
        guidance_params["subvalue_key"] = "pic50"
        guidance_params["mu"] = 8.0
        guidance_params["sigma"] = 2.0
        guidance_params["maximize"] = True
        guidance_params["coord_noise_level"] = 0.2

    output = model._generate_selective(
        prior=lig_prior,
        pocket_data_target=pocket_data_target,
        pocket_data_untarget=pocket_data_untarget,
        times=prior_times,
        steps=args.integration_steps,
        strategy=args.ode_sampling_strategy,
        solver=args.solver,
        corr_iters=args.corrector_iters,
        save_traj=save_traj,
        iter=iter,
        apply_guidance=guidance_params["apply_guidance"],
        guidance_window_start=guidance_params["window_start"],
        guidance_window_end=guidance_params["window_end"],
        value_key=guidance_params["value_key"],
        mu=guidance_params["mu"],
        sigma=guidance_params["sigma"],
        maximize=guidance_params["maximize"],
        coord_noise_level=guidance_params["coord_noise_level"],
    )

    # Generate RDKit molecules
    gen_ligs = model._generate_mols(output)

    # Attach affinity predictions as properties if present
    if "affinity" in output:
        affinity = output["affinity"]
        # Ensure affinity is a dict-like object with keys: pic50, pki, pkd, pec50
        for mol, idx in zip(gen_ligs, range(len(gen_ligs))):
            if mol is not None:
                metadata = posterior_target["complex"][0].metadata
                system_id = metadata["system_id"]
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
                for key in ["pic50", "pki", "pkd", "pec50"] + [
                    "pic50_untarget",
                    "pki_untarget",
                    "pkd_untarget",
                    "pec50_untarget",
                ]:
                    value = affinity[key]
                    # If value is a tensor, get the scalar for this molecule
                    if hasattr(value, "detach"):
                        val = value[idx].item() if value.ndim > 0 else value.item()
                    else:
                        val = value[idx] if isinstance(value, (list, tuple)) else value
                    mol.SetProp(key, str(val))
    return gen_ligs
