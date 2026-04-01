from ase import Atoms
from ase.io import Trajectory
from ase.optimize import BFGS, LBFGS
from fairchem.core import FAIRChemCalculator
from rdkit import Chem


def relax_molecules(
    ligands: list,
    ckpt_path: str = "./esen_md_direct_all.pt",
    optimizer="LBFGS",
    steps: int = 300,
    step_size: float = 0.01,
    device: str = "cpu",
    create_traj: bool = False,
    verbose: bool = False,
):
    """
    Relax a list of molecules using ASE and FAIRChem.
    This function converts RDKit molecules to ASE Atoms objects,
    sets up a calculator, and optimizes the geometry using BFGS or LBFGS.
    The optimized structures are then converted back to RDKit molecules.
    Returns:
    --------
    list
        List of optimized RDKit molecules
    """
    # Example usage
    optim_ligands = []
    for ligand in ligands:
        optim_lig = relax_molecule(
            ligand,
            ckpt_path=ckpt_path,
            optimizer=optimizer,
            steps=steps,
            step_size=step_size,
            device=device,
            create_traj=create_traj,
            verbose=verbose,
        )
        optim_ligands.append(optim_lig)
    return optim_ligands


def relax_molecule(
    ligand: Chem.Mol,
    ckpt_path: str = "./omol25/esen_md_direct_all.pt",
    optimizer="BFGS",
    steps: int = 300,
    step_size: float = 0.01,
    device: str = "cpu",
    create_traj: bool = False,
    verbose: bool = False,
):
    """
    Relax a molecule using ASE and FAIRChem.
    This function converts an RDKit molecule to an ASE Atoms object,
    sets up a calculator, and optimizes the geometry using BFGS or LBFGS.
    The optimized structure is then converted back to an RDKit molecule.
    """

    # Convert the RDKit molecule to an ASE Atoms object
    atoms = rdkit_conf_to_ase(ligand)

    # Set up the FAIRChem calculator
    calc = FAIRChemCalculator(checkpoint_path=ckpt_path, device=device)
    atoms.calc = calc

    # Create a trajectory file to store intermediate steps
    traj = Trajectory("optimization.traj", "w", atoms) if create_traj else None

    # Set up optimizer with trajectory file
    if optimizer == "LBFGS":
        optimizer = LBFGS(atoms, trajectory=traj)
    elif optimizer == "BFGS":
        optimizer = BFGS(atoms, trajectory=traj)
    else:
        raise ValueError("Optimizer must be 'BFGS' or 'LBFGS'.")

    # Run optimization with a callback function
    if verbose:
        # To print energies and forces during optimization
        def print_status():
            energy = atoms.get_potential_energy()
            fmax = max([norm for norm in [force.max() for force in atoms.get_forces()]])
            print(f"Energy: {energy:.6f} eV, Maximum force: {fmax:.6f} eV/Å")

        optimizer.attach(print_status, interval=1)
    optimizer.run(fmax=step_size, steps=steps)

    # Get the optimized RDKit molecule
    optimized_mol = ase_to_rdkit(atoms)
    return optimized_mol


def rdkit_conf_to_ase(mol, conf_id=-1):
    """
    Convert an RDKit molecule with a conformer to an ASE Atoms object.
    Stores a COPY of the RDKit molecule in the atoms.info dictionary.

    Parameters:
    -----------
    mol : rdkit.Chem.rdchem.Mol
        The RDKit molecule
    conf_id : int, default=-1
        The conformer ID to use. By default, uses the last conformer.

    Returns:
    --------
    ase.Atoms
        The ASE Atoms object with a copy of the RDKit molecule stored in atoms.info
    """
    # Create a deep copy of the molecule to avoid modifying the original
    mol_copy = Chem.Mol(mol)

    # Get the conformer
    conf = mol_copy.GetConformer(conf_id)

    # Get atomic positions from the conformer
    positions = []
    for atom_idx in range(mol_copy.GetNumAtoms()):
        pos = conf.GetAtomPosition(atom_idx)
        positions.append([pos.x, pos.y, pos.z])

    # Get atomic symbols
    symbols = [atom.GetSymbol() for atom in mol_copy.GetAtoms()]

    # Calculate the total formal charge from RDKit molecule
    charge = sum(atom.GetFormalCharge() for atom in mol_copy.GetAtoms())

    # Create an ASE Atoms object
    atoms = Atoms(symbols=symbols, positions=positions)

    # Store the charge and a copy of the RDKit molecule in atoms.info
    atoms.info["charge"] = charge
    atoms.info["rdkit_mol"] = mol_copy

    return atoms


def ase_to_rdkit(atoms):
    """
    Retrieve the RDKit molecule from an ASE Atoms object and update its coordinates.
    Returns a copy of the stored molecule with updated coordinates.

    Parameters:
    -----------
    atoms : ase.Atoms
        The ASE Atoms object with an RDKit molecule stored in atoms.info

    Returns:
    --------
    rdkit.Chem.rdchem.Mol
        A copy of the RDKit molecule with coordinates updated from ASE
    """
    if "rdkit_mol" not in atoms.info:
        raise ValueError(
            "No RDKit molecule found in atoms.info. This ASE Atoms object was not created with rdkit_conf_to_ase()."
        )

    # Make a copy of the stored RDKit molecule
    mol = Chem.Mol(atoms.info["rdkit_mol"])

    # Create a new conformer with updated coordinates
    conf = Chem.Conformer(len(atoms))
    for i in range(len(atoms)):
        x, y, z = atoms.positions[i]
        conf.SetAtomPosition(i, (float(x), float(y), float(z)))

    # Replace the first conformer (if any) with the new one
    if mol.GetNumConformers() > 0:
        mol.RemoveAllConformers()
    mol.AddConformer(conf)

    return mol
