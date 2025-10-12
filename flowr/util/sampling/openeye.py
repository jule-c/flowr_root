import os

import openeye
from openeye import oeshape
from openeye.oechem import *
from openeye.oedepict import *
from openeye.oeiupac import *
from openeye.oeomega import *
from openeye.oeshape import *

license_filename = "/sharedhome/cremej01/workspace/e3moldiffusion/oe_license.txt"

if os.path.isfile(license_filename):
    license_file = open(license_filename, "r")
    openeye.OEAddLicenseData(license_file.read())
    license_file.close()
    assert openeye.oechem.OEChemIsLicensed()
else:
    raise Exception(
        "Error: Your OpenEye license is not readable; please check your filename and that you have mounted your Google Drive"
    )

import rdkit
from rdkit import Chem, Geometry
from rdkit.Geometry import Point3D

from experiments.sampling.oe_utils import *


def oemol_from_rdmol(rdkitmol):
    rdmol = deepcopy(rdkitmol)
    """
    Creates an openeye molecule object that is identical to the input rdkit molecule
    """
    # RDK automatically includes explicit hydrogens in its SMILES patterns
    # print("Starting molecule: ", Chem.MolToSmiles(Chem.RemoveHs(rdmol)))

    # openeye stores bond orders as integers regardless of aromaticity
    # in order to properly extract these, we need to have the "Kekulized" version of the rdkit mol
    kekul_mol = Chem.Mol(rdmol)
    Chem.Kekulize(kekul_mol, True)

    oemol = oechem.OEMol()
    map_atoms = dict()  # {rd_idx: oe_atom}

    # setting chirality in openey requires using neighbor atoms
    # therefore we can't do it until after the atoms and bonds are all added
    chiral_atoms = dict()  # {rd_idx: openeye chirality}
    for rda in rdmol.GetAtoms():
        rd_idx = rda.GetIdx()

        # create a new atom
        oe_a = oemol.NewAtom(rda.GetAtomicNum())
        map_atoms[rd_idx] = oe_a
        oe_a.SetFormalCharge(rda.GetFormalCharge())
        oe_a.SetAromatic(rda.GetIsAromatic())

        # If chiral, store the chirality to be set later
        tag = rda.GetChiralTag()
        if tag == Chem.CHI_TETRAHEDRAL_CCW:
            chiral_atoms[rd_idx] = oechem.OECIPAtomStereo_R
        if tag == Chem.CHI_TETRAHEDRAL_CW:
            chiral_atoms[rd_idx] = oechem.OECIPAtomStereo_S

    # Similar to chirality, stereochemistry of bonds in OE is set relative to their neighbors
    stereo_bonds = list()
    # stereo_bonds stores tuples in the form (oe_bond, rd_idx1, rd_idx2, OE stereo specification)
    # where rd_idx1 and 2 are the atoms on the outside of the bond
    # i.e. Cl and F in the example above
    aro_bond = 0
    for rdb in rdmol.GetBonds():
        a1 = rdb.GetBeginAtomIdx()
        a2 = rdb.GetEndAtomIdx()

        # create a new bond
        newbond = oemol.NewBond(map_atoms[a1], map_atoms[a2])

        order = rdb.GetBondTypeAsDouble()
        if order == 1.5:
            # get the bond order for this bond in the kekulized molecule
            order = kekul_mol.GetBondWithIdx(rdb.GetIdx()).GetBondTypeAsDouble()
            newbond.SetAromatic(True)
        else:
            newbond.SetAromatic(False)
        newbond.SetOrder(int(order))

        # determine if stereochemistry is needed
        tag = rdb.GetStereo()
        if tag == Chem.BondStereo.STEREOCIS or tag == Chem.BondStereo.STEREOZ:
            stereo_atoms = rdb.GetStereoAtoms()
            stereo_bonds.append(
                (newbond, stereo_atoms[0], stereo_atoms[1], oechem.OEBondStereo_Cis)
            )

            bond2 = rdmol.GetBondBetweenAtoms(stereo_atoms[0], a1)
            bond4 = rdmol.GetBondBetweenAtoms(stereo_atoms[1], a2)
            print(tag, bond2.GetBondDir(), bond4.GetBondDir())
        if tag == Chem.BondStereo.STEREOTRANS or tag == Chem.BondStereo.STEREOE:
            stereo_atoms = rdb.GetStereoAtoms()
            stereo_bonds.append(
                (newbond, stereo_atoms[0], stereo_atoms[1], oechem.OEBondStereo_Trans)
            )
            bond2 = rdmol.GetBondBetweenAtoms(stereo_atoms[0], a1)
            bond4 = rdmol.GetBondBetweenAtoms(stereo_atoms[1], a2)
            print(tag, bond2.GetBondDir(), bond4.GetBondDir())

    # Now that all of the atoms are connected we can set stereochemistry
    # starting with atom chirality
    for rd_idx, chirality in chiral_atoms.items():
        # chirality is set relative to neighbors, so we will get neighboring atoms
        # assign Right handed direction, check the cip stereochemistry
        # if the cip stereochemistry isn't correct then we'll set left and double check

        oea = map_atoms[rd_idx]
        neighs = [n for n in oea.GetAtoms()]
        # incase you look at the documentation oe has two options for handedness for example:
        # oechem.OEAtomStereo_Left == oechem.OEAtomStereo_LeftHanded
        oea.SetStereo(neighs, oechem.OEAtomStereo_Tetra, oechem.OEAtomStereo_Right)
        cip = oechem.OEPerceiveCIPStereo(oemol, oea)
        if cip != chirality:
            oea.SetStereo(neighs, oechem.OEAtomStereo_Tetra, oechem.OEAtomStereo_Left)
            new_cip = oechem.OEPerceiveCIPStereo(oemol, oea)
            if new_cip != chirality:
                # Note, I haven't seen this happen yet, but it shouldn't be a problem since there
                # is only 2 directions for handedness and we're only running this for chiral atoms
                # print("PANIC!")
                pass

    # Set stereochemistry using the reference atoms extracted above
    for oeb, idx1, idx2, oestereo in stereo_bonds:
        oeb.SetStereo(
            [map_atoms[idx1], map_atoms[idx2]], oechem.OEBondStereo_CisTrans, oestereo
        )

    # If the rdmol has a conformer, add its coordinates to the oemol
    # Note, this currently only adds the first conformer, it will need to be adjusted if the
    # you wanted to convert multiple sets of coordinates
    if rdmol.GetConformers():
        # print("found an rdmol conformer")
        conf = rdmol.GetConformer()
        for rd_idx, oeatom in map_atoms.items():
            coords = conf.GetAtomPosition(rd_idx)
            oemol.SetCoords(oeatom, oechem.OEFloatArray(coords))

    # If RDMol has a title save it
    if rdmol.HasProp("_Name"):
        oemol.SetTitle(rdmol.GetProp("_Name"))

    # Clean Up phase
    # The only feature of a molecule that wasn't perceived above seemed to be ring connectivity, better to run it
    # here then for someone to inquire about ring sizes and get 0 when it shouldn't be
    oechem.OEFindRingAtomsAndBonds(oemol)

    # print('Final Molecule: ', oechem.OEMolToSmiles(oemol))
    return oemol


def rdmol_from_oemol(oemol):
    """
    Creates an openeye molecule object that is identical to the input rdkit molecule
    """
    # print("Starting molecule: ", oechem.OEMolToSmiles(oemol))

    # start function
    rdmol = Chem.RWMol()

    # RDKit keeps bond order as a type instead using these values, I don't really understand 7,
    # I took them from Shuzhe's example linked above
    _bondtypes = {
        1: Chem.BondType.SINGLE,
        1.5: Chem.BondType.AROMATIC,
        2: Chem.BondType.DOUBLE,
        3: Chem.BondType.TRIPLE,
        4: Chem.BondType.QUADRUPLE,
        5: Chem.BondType.QUINTUPLE,
        6: Chem.BondType.HEXTUPLE,
        7: Chem.BondType.ONEANDAHALF,
    }

    # atom map lets you find atoms again
    map_atoms = dict()  # {oe_idx: rd_idx}
    for oea in oemol.GetAtoms():
        oe_idx = oea.GetIdx()
        rda = Chem.Atom(oea.GetAtomicNum())
        rda.SetFormalCharge(oea.GetFormalCharge())
        rda.SetIsAromatic(oea.IsAromatic())

        # unlike OE, RDK lets you set chirality directly
        cip = oechem.OEPerceiveCIPStereo(oemol, oea)
        if cip == oechem.OECIPAtomStereo_S:
            rda.SetChiralTag(Chem.CHI_TETRAHEDRAL_CW)
        if cip == oechem.OECIPAtomStereo_R:
            rda.SetChiralTag(Chem.CHI_TETRAHEDRAL_CCW)

        map_atoms[oe_idx] = rdmol.AddAtom(rda)

    # As discussed above, setting bond stereochemistry requires neighboring bonds
    # so we will store that information by atom index in this list
    stereo_bonds = list()
    # stereo_bonds will have tuples with the form (rda1, rda2, rda3, rda4, is_cis)
    # where rda[n] is an atom index for a double bond of form 1-2=3-4
    # and is_cis is a Boolean is True then onds 1-2 and 3-4 are cis to each other

    aro_bond = 0
    for oeb in oemol.GetBonds():
        # get neighboring rd atoms
        rd_a1 = map_atoms[oeb.GetBgnIdx()]
        rd_a2 = map_atoms[oeb.GetEndIdx()]

        # AddBond returns the total number of bonds, so addbond and then get it
        rdmol.AddBond(rd_a1, rd_a2)
        rdbond = rdmol.GetBondBetweenAtoms(rd_a1, rd_a2)

        # Assign bond type, which is based on order unless it is aromatic
        order = oeb.GetOrder()
        rdbond.SetBondType(_bondtypes[order])
        if oeb.IsAromatic():
            rdbond.SetBondType(_bondtypes[1.5])
            rdbond.SetIsAromatic(True)
        # print(order)

        # if oeb.IsAromatic():
        #    rdbond.SetBondType(_bondtypes[1.5])
        #    rdbond.SetIsAromatic(True)
        # else:
        #    rdbond.SetBondType(_bondtypes[order])
        #    rdbond.SetIsAromatic(False)

        # If the bond has specified stereo add the required information to stereo_bonds
        if oeb.HasStereoSpecified(oechem.OEBondStereo_CisTrans):
            # OpenEye determined stereo based on neighboring atoms so get two outside atoms
            n1 = [n for n in oeb.GetBgn().GetAtoms() if n != oeb.GetEnd()][0]
            n2 = [n for n in oeb.GetEnd().GetAtoms() if n != oeb.GetBgn()][0]

            rd_n1 = map_atoms[n1.GetIdx()]
            rd_n2 = map_atoms[n2.GetIdx()]

            stereo = oeb.GetStereo([n1, n2], oechem.OEBondStereo_CisTrans)
            if stereo == oechem.OEBondStereo_Cis:
                # print('cis')
                stereo_bonds.append((rd_n1, rd_a1, rd_a2, rd_n2, True))
            elif stereo == oechem.OEBondStereo_Trans:
                # print('trans')
                stereo_bonds.append((rd_n1, rd_a1, rd_a2, rd_n2, False))

    # add bond stereochemistry:
    for rda1, rda2, rda3, rda4, is_cis in stereo_bonds:
        # get neighbor bonds
        bond1 = rdmol.GetBondBetweenAtoms(rda1, rda2)
        bond2 = rdmol.GetBondBetweenAtoms(rda3, rda4)

        # Since this is relative, the first bond always goes up
        # as explained above these names come from SMILES slashes so UP/UP is Trans and Up/Down is cis
        bond1.SetBondDir(Chem.BondDir.ENDUPRIGHT)
        if is_cis:
            bond2.SetBondDir(Chem.BondDir.ENDDOWNRIGHT)
        else:
            bond2.SetBondDir(Chem.BondDir.ENDUPRIGHT)

    # if oemol has coordinates (The dimension is non-zero)
    # add those coordinates to the rdmol
    if oechem.OEGetDimensionFromCoords(oemol) > 0:
        # print("oemol has 3D coords")
        conformer = Chem.Conformer()
        oecoords = oemol.GetCoords()
        for oe_idx, rd_idx in map_atoms.items():
            (x, y, z) = oecoords[oe_idx]
            conformer.SetAtomPosition(rd_idx, Geometry.Point3D(x, y, z))
        rdmol.AddConformer(conformer)

    # Save the molecule title
    rdmol.SetProp("_Name", oemol.GetTitle())

    # Cleanup the rdmol
    # Note I copied UpdatePropertyCache and GetSSSR from Shuzhe's code to convert oemol to rdmol here:
    rdmol.UpdatePropertyCache(strict=False)
    Chem.GetSSSR(rdmol)
    # I added AssignStereochemistry which takes the directions of the bond set
    # and assigns the stereochemistry tags on the double bonds
    try:
        rdkit.Chem.rdmolops.AssignStereochemistryFrom3D(rdmol)
    except:
        Chem.AssignStereochemistry(rdmol, force=False)

    # print("Final Molecule: ", Chem.MolToSmiles(Chem.RemoveHs(rdmol), isomericSmiles=True))
    return rdmol.GetMol()


def ROCS_shape_overlap(query_list, reference, cast_RDKit=True):
    refmol = oemol_from_rdmol(deepcopy(reference))
    # prep = oeshape.OEOverlapPrep() # colors
    # prep.Prep(refmol) # colors
    overlay = oeshape.OEMultiRefOverlay()
    overlay.SetupRef(refmol)

    scores = []
    for query_mol in query_list:
        fitmol = oemol_from_rdmol(deepcopy(query_mol))
        # prep.Prep(fitmol)

        # Sort all scores according to highest tanimoto
        scoreiter = oeshape.OEBestOverlayScoreIter()
        oeshape.OESortOverlayScores(
            scoreiter, overlay.Overlay(fitmol), oeshape.OEHighestTanimoto()
        )

        for (
            score
        ) in scoreiter:  # only 1 conformer, so only 1 score (iterates through 1 time)
            outmol = oechem.OEGraphMol(
                fitmol.GetConf(oechem.OEHasConfIdx(score.GetFitConfIdx()))
            )
            score.Transform(outmol)

            # oechem.OESetSDData(outmol, "RefConfIdx", "%-d" % score.GetRefConfIdx())
            # oechem.OESetSDData(outmol, "tanimoto shape", "%-.3f" % score.GetTanimoto())
            # oechem.OESetSDData(outmol, "tanimoto color", "%-.3f" % score.GetColorTanimoto())
            # oechem.OESetSDData(outmol, "tanimoto combo", "%-.3f" % score.GetTanimotoCombo())
            # oechem.OEWriteMolecule(outfs, refmol.GetConf(oechem.OEHasConfIdx(score.GetRefConfIdx())))
            # oechem.OEWriteMolecule(outfs, outmol)

        try:
            outmol_rdkit = rdmol_from_oemol(outmol)

            if cast_RDKit:
                query_mol_aligned = deepcopy(query_mol)
                reindexing_map = get_reindexing_map_for_matching(
                    query_mol_aligned,
                    list(range(0, query_mol_aligned.GetNumAtoms())),
                    outmol_rdkit,
                )
                for key in reindexing_map.keys():
                    x, y, z = outmol_rdkit.GetConformer().GetPositions()[
                        reindexing_map[key]
                    ]
                    query_mol_aligned.GetConformer().SetAtomPosition(
                        key, Point3D(x, y, z)
                    )
            else:
                query_mol_aligned = outmol_rdkit

            # this has aromaticity reading issues for aromatic heterorings
            # assert rdkit.Chem.MolToSmiles(outmol_rdkit, isomericSmiles = False) == rdkit.Chem.MolToSmiles(query_mol, isomericSmiles = False)
            scores.append(
                (
                    score.GetTanimotoCombo(),
                    score.GetTanimoto(),
                    get_ROCS_mols(reference, query_mol_aligned).item(),
                    query_mol_aligned,
                )
            )

        except:
            scores.append((score.GetTanimoto(), None, None))

    scores = {
        "tanimoto_color_shape_overlap": [s[0] for s in scores],
        "tanimoto_shape_overlap": [s[1] for s in scores],
        "ROCS": [s[2] for s in scores if s[2] is not None],
        "mol_aligned": [s[3] for s in scores if s[3] is not None],
    }
    return scores


def ROCS_color_overlap(query_list, reference, cast_RDKit=True):
    refmol = oemol_from_rdmol(deepcopy(reference))
    prep = oeshape.OEOverlapPrep()  # colors
    prep.Prep(refmol)  # colors
    overlay = oeshape.OEMultiRefOverlay()
    overlay.SetupRef(refmol)

    scores = []
    for query_mol in tqdm(query_list):
        fitmol = oemol_from_rdmol(deepcopy(query_mol))
        prep.Prep(fitmol)

        # Sort all scores according to highest tanimoto
        scoreiter = oeshape.OEBestOverlayScoreIter()
        oeshape.OESortOverlayScores(
            scoreiter, overlay.Overlay(fitmol), oeshape.OEHighestTanimoto()
        )

        for (
            score
        ) in scoreiter:  # only 1 conformer, so only 1 score (iterates through 1 time)
            outmol = oechem.OEGraphMol(
                fitmol.GetConf(oechem.OEHasConfIdx(score.GetFitConfIdx()))
            )
            score.Transform(outmol)

            # oechem.OESetSDData(outmol, "RefConfIdx", "%-d" % score.GetRefConfIdx())
            # oechem.OESetSDData(outmol, "tanimoto shape", "%-.3f" % score.GetTanimoto())
            # oechem.OESetSDData(outmol, "tanimoto color", "%-.3f" % score.GetColorTanimoto())
            # oechem.OESetSDData(outmol, "tanimoto combo", "%-.3f" % score.GetTanimotoCombo())
            # oechem.OEWriteMolecule(outfs, refmol.GetConf(oechem.OEHasConfIdx(score.GetRefConfIdx())))
            # oechem.OEWriteMolecule(outfs, outmol)

        try:
            outmol_rdkit = rdmol_from_oemol(outmol)

            if cast_RDKit:
                query_mol_aligned = deepcopy(query_mol)
                reindexing_map = get_reindexing_map_for_matching(
                    query_mol_aligned,
                    list(range(0, query_mol_aligned.GetNumAtoms())),
                    outmol_rdkit,
                )
                for key in reindexing_map.keys():
                    x, y, z = outmol_rdkit.GetConformer().GetPositions()[
                        reindexing_map[key]
                    ]
                    query_mol_aligned.GetConformer().SetAtomPosition(
                        key, Point3D(x, y, z)
                    )
            else:
                query_mol_aligned = outmol_rdkit

            # this has aromaticity reading issues for aromatic heterorings
            # assert rdkit.Chem.MolToSmiles(outmol_rdkit, isomericSmiles = False) == rdkit.Chem.MolToSmiles(query_mol, isomericSmiles = False)

            scores.append(
                (
                    score.GetTanimotoCombo(),
                    get_ROCS_mols(reference, query_mol_aligned).item(),
                    query_mol_aligned,
                )
            )

        except:
            scores.append((score.GetTanimotoCombo(), None, None))

    scores = {
        "tanimoto_color_shape_overlap": [s[0] for s in scores],
        "ROCS": [s[1] for s in scores if s is not None],
        "mol_aligned": [s[2] for s in scores if s is not None],
    }
    return scores
    return scores
