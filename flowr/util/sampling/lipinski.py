from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, Lipinski


def log_partition_coefficient(rdmol):
    """
    Returns the octanol-water partition coefficient given a molecule SMILES
    string
    """
    return Crippen.MolLogP(rdmol)


def lipinski_trial(rdmol):
    """
    Returns which of Lipinski's rules a molecule has failed, or an empty list

    Lipinski's rules are:
    Hydrogen bond donors <= 5
    Hydrogen bond acceptors <= 10
    Molecular weight < 500 daltons
    logP < 5
    """
    passed = []
    failed = []

    num_hdonors = Lipinski.NumHDonors(rdmol)
    num_hacceptors = Lipinski.NumHAcceptors(rdmol)
    mol_weight = Descriptors.MolWt(rdmol)
    mol_logp = Crippen.MolLogP(rdmol)

    failed = []

    if num_hdonors > 5:
        failed.append("Over 5 H-bond donors, found %s" % num_hdonors)
    else:
        passed.append("Found %s H-bond donors" % num_hdonors)

    if num_hacceptors > 10:
        failed.append("Over 10 H-bond acceptors, found %s" % num_hacceptors)
    else:
        passed.append("Found %s H-bond acceptors" % num_hacceptors)

    if mol_weight >= 500:
        failed.append("Molecular weight over 500, calculated %s" % mol_weight)
    else:
        passed.append("Molecular weight: %s" % mol_weight)

    if mol_logp >= 5:
        failed.append("Log partition coefficient over 5, calculated %s" % mol_logp)
    else:
        passed.append("Log partition coefficient: %s" % mol_logp)

    return passed, failed


def lipinski_pass(rdmol):
    """
    Wraps around lipinski trial, but returns a simple pass/fail True/False
    """
    passed, failed = lipinski_trial(rdmol)
    if failed:
        return False
    else:
        return True
