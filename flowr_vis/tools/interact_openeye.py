"""
Protein-Ligand Interaction Visualization using OpenEye

This script generates 2D depictions of protein-ligand interactions using OpenEye's
OEGrapheme toolkit. It can accept either a complex file or separate protein and
ligand files.

Usage:
    # Using a complex file:
    python evaluate_interact_openeye.py -c complex.pdb -o output.svg

    # Using separate protein and ligand files:
    python evaluate_interact_openeye.py -p protein.pdb -l ligand.sdf -o output.png

    # With custom image size and interactive legend (SVG only):
    python evaluate_interact_openeye.py -c complex.pdb -o output.svg --width 1200 --height 800 --interactive-legend
"""

import argparse
import os
import sys
from pathlib import Path

from openeye import oechem, oedepict, oegrapheme

# Set OpenEye license path - modify this to your license location
OE_LICENSE_PATH = os.environ.get("OE_LICENSE", "./oe_license.txt")
os.environ["OE_LICENSE"] = OE_LICENSE_PATH
os.environ["SINGULARITYENV_OE_LICENSE"] = OE_LICENSE_PATH


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate 2D protein-ligand interaction diagrams using OpenEye.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # From a complex file:
  %(prog)s -c complex.pdb -o interactions.svg

  # From separate protein and ligand files:
  %(prog)s -p protein.pdb -l ligand.sdf -o interactions.png

  # With custom dimensions and interactive legend:
  %(prog)s -c complex.pdb -o interactions.svg -W 1200 -H 800 --interactive-legend
        """,
    )

    # Input options (mutually exclusive groups)
    input_group = parser.add_argument_group("Input Options")
    input_group.add_argument(
        "-c",
        "--complex",
        type=str,
        help="Input filename of the protein-ligand complex (PDB format)",
    )
    input_group.add_argument(
        "-p",
        "--protein",
        type=str,
        help="Input filename of the protein (PDB format)",
    )
    input_group.add_argument(
        "-l",
        "--ligand",
        type=str,
        help="Input filename of the ligand (SDF, MOL2, PDB format)",
    )

    # Output options
    output_group = parser.add_argument_group("Output Options")
    output_group.add_argument(
        "-o",
        "--output",
        type=str,
        required=True,
        help="Output filename for the generated image (svg, png, pdf supported)",
    )

    # Image options
    image_group = parser.add_argument_group("Image Options")
    image_group.add_argument(
        "-W",
        "--width",
        type=float,
        default=2400.0,
        help="Image width in pixels (default: 2400)",
    )
    image_group.add_argument(
        "-H",
        "--height",
        type=float,
        default=1800.0,
        help="Image height in pixels (default: 1800)",
    )

    # Display options
    display_group = parser.add_argument_group("Display Options (SVG only)")
    display_group.add_argument(
        "--interactive-legend",
        action="store_true",
        default=False,
        help="Show legend on mouse hover (SVG feature)",
    )
    display_group.add_argument(
        "--magnify-residue",
        type=float,
        default=1.0,
        help="Scaling factor to magnify residue glyph on hover (1.0-3.0, SVG feature)",
    )

    # Complex splitting options
    split_group = parser.add_argument_group("Complex Splitting Options")
    split_group.add_argument(
        "--ligand-name",
        type=str,
        default=None,
        help="Ligand residue name to extract from complex (e.g., 'LIG', 'UNK')",
    )
    split_group.add_argument(
        "--ligand-idx",
        type=int,
        default=0,
        help="Index of ligand to use from multi-molecule SDF file (0-based, default: 0)",
    )

    args = parser.parse_args()

    # Validate input arguments
    has_complex = args.complex is not None
    has_protein = args.protein is not None
    has_ligand = args.ligand is not None

    if has_complex and (has_protein or has_ligand):
        parser.error(
            "Cannot specify both --complex and --protein/--ligand. "
            "Use either --complex OR both --protein and --ligand."
        )

    if not has_complex and not (has_protein and has_ligand):
        parser.error("Must specify either --complex OR both --protein and --ligand.")

    # Validate output file extension
    ext = Path(args.output).suffix.lower().lstrip(".")
    if ext not in ["svg", "png", "pdf", "ps", "eps"]:
        parser.error(f"Unsupported output format: {ext}. Use svg, png, or pdf.")

    # Validate magnify-residue range
    if not 1.0 <= args.magnify_residue <= 3.0:
        parser.error("--magnify-residue must be between 1.0 and 3.0")

    return args


def main():
    """Main entry point."""
    args = parse_args()

    # Get output extension
    ext = Path(args.output).suffix.lower().lstrip(".")

    # Check if output file can be created
    if not oedepict.OEIsRegisteredImageFile(ext):
        print(f"Error: Unknown image type '{ext}'!", file=sys.stderr)
        return 1

    # Initialize protein and ligand molecules
    protein = oechem.OEGraphMol()
    ligand = oechem.OEGraphMol()

    if not get_protein_and_ligand(protein, ligand, args):
        print("Error: Cannot initialize protein and/or ligand!", file=sys.stderr)
        return 1

    # Create image
    width, height = args.width, args.height
    image = oedepict.OEImage(width, height)

    # Configure display options based on format
    interactive_legend = False
    magnify_residue = 4.0

    if ext == "svg":
        interactive_legend = args.interactive_legend
        magnify_residue = args.magnify_residue

    # Calculate content dimensions
    cwidth, cheight = width, height
    if not interactive_legend:
        cwidth = cwidth * 0.8

    # Setup active site display options
    opts = oegrapheme.OE2DActiveSiteDisplayOptions(cwidth, cheight)
    opts.SetRenderInteractiveLegend(interactive_legend)
    opts.SetSVGMagnifyResidueInHover(magnify_residue)

    # Render the complex
    if interactive_legend:
        depict_complex(image, protein, ligand, opts)
    else:
        main_frame = oedepict.OEImageFrame(
            image, width * 0.80, height, oedepict.OE2DPoint(width * 0.2, 0.0)
        )
        legend_frame = oedepict.OEImageFrame(
            image, width * 0.20, height, oedepict.OE2DPoint(width * 0.0, 0.0)
        )
        depict_complex(main_frame, protein, ligand, opts, legend_frame)

    # Add interactive elements for SVG
    if ext == "svg" and (interactive_legend or magnify_residue > 1.0):
        iconscale = 0.5
        oedepict.OEAddInteractiveIcon(
            image, oedepict.OEIconLocation_TopRight, iconscale
        )

    # Add border and write output
    oedepict.OEDrawCurvedBorder(image, oedepict.OELightGreyPen, 10.0)
    oedepict.OEWriteImage(args.output, image)

    print(f"Successfully wrote interaction diagram to: {args.output}")
    return 0


def depict_complex(image, protein, ligand, opts, legend_frame=None):
    """
    Depict the protein-ligand complex with interactions.

    Args:
        image: OEImageBase to render to
        protein: OEMolBase containing the protein
        ligand: OEMolBase containing the ligand
        opts: OE2DActiveSiteDisplayOptions for rendering
        legend_frame: Optional OEImageBase for the legend
    """
    # Perceive interactions
    asite = oechem.OEInteractionHintContainer(protein, ligand)
    if not asite.IsValid():
        oechem.OEThrow.Fatal("Cannot initialize active site!")
    asite.SetTitle("")

    oechem.OEPerceiveInteractionHints(asite)

    # Render depiction
    oegrapheme.OEPrepareActiveSiteDepiction(asite)
    adisp = oegrapheme.OE2DActiveSiteDisplay(asite, opts)
    oegrapheme.OERenderActiveSite(image, adisp)

    if legend_frame is not None:
        lopts = oegrapheme.OE2DActiveSiteLegendDisplayOptions(18, 1)
        oegrapheme.OEDrawActiveSiteLegend(legend_frame, adisp, lopts)


def split_complex(protein, ligand, complexmol, ligand_name=None):
    """
    Split a complex molecule into protein and ligand components.

    Args:
        protein: OEGraphMol to store the protein
        ligand: OEGraphMol to store the ligand
        complexmol: OEMolBase containing the complex
        ligand_name: Optional ligand residue name to filter by

    Returns:
        bool: True if splitting was successful
    """
    sopts = oechem.OESplitMolComplexOptions()

    # Configure ligand filter if name provided
    if ligand_name:
        lig_filter = oechem.OEMolComplexFilterFactory(
            oechem.OEMolComplexFilterCategory_Ligand
        )
        sopts.SetLigandFilter(
            oechem.OEAndRoleSet(
                lig_filter,
                oechem.OEHasResidueProperty(oechem.OEResiduePropertyName(ligand_name)),
            )
        )

    water = oechem.OEGraphMol()
    other = oechem.OEGraphMol()

    # Merge water with protein
    pfilter = sopts.GetProteinFilter()
    wfilter = sopts.GetWaterFilter()
    sopts.SetProteinFilter(oechem.OEOrRoleSet(pfilter, wfilter))
    filter_nothing = oechem.OEMolComplexFilterCategory_Nothing
    sopts.SetWaterFilter(oechem.OEMolComplexFilterFactory(filter_nothing))

    oechem.OESplitMolComplex(ligand, protein, water, other, complexmol, sopts)

    return ligand.NumAtoms() != 0 and protein.NumAtoms() != 0


def get_protein_and_ligand(protein, ligand, args):
    """
    Load protein and ligand from input files.

    Args:
        protein: OEGraphMol to store the protein
        ligand: OEGraphMol to store the ligand
        args: Parsed command line arguments

    Returns:
        bool: True if loading was successful
    """
    if args.complex:
        # Load and split complex file
        ifs = oechem.oemolistream()
        if not ifs.open(args.complex):
            print(f"Error: Cannot open complex file: {args.complex}", file=sys.stderr)
            return False

        complexmol = oechem.OEGraphMol()
        if not oechem.OEReadMolecule(ifs, complexmol):
            print(f"Error: Unable to read complex from {args.complex}", file=sys.stderr)
            return False

        # Ensure residue information is present
        if not oechem.OEHasResidues(complexmol):
            oechem.OEPerceiveResidues(complexmol, oechem.OEPreserveResInfo_All)

        if not split_complex(protein, ligand, complexmol, args.ligand_name):
            print(
                "Error: Cannot separate complex into protein and ligand!",
                file=sys.stderr,
            )
            return False
    else:
        # Load protein from file
        ifs = oechem.oemolistream()
        if not ifs.open(args.protein):
            print(f"Error: Cannot open protein file: {args.protein}", file=sys.stderr)
            return False

        if not oechem.OEReadMolecule(ifs, protein):
            print(f"Error: Unable to read protein from {args.protein}", file=sys.stderr)
            return False

        # Load ligand from file, selecting by index
        ifs = oechem.oemolistream()
        if not ifs.open(args.ligand):
            print(f"Error: Cannot open ligand file: {args.ligand}", file=sys.stderr)
            return False

        # Read molecules until we reach the desired index
        ligand_idx = args.ligand_idx
        current_idx = 0
        temp_mol = oechem.OEGraphMol()

        while oechem.OEReadMolecule(ifs, temp_mol):
            if current_idx == ligand_idx:
                oechem.OECopyMol(ligand, temp_mol)
                break
            current_idx += 1
            temp_mol.Clear()

        if ligand.NumAtoms() == 0:
            print(
                f"Error: Could not find ligand at index {ligand_idx} in {args.ligand}. "
                f"File contains {current_idx} molecule(s).",
                file=sys.stderr,
            )
            return False

    return ligand.NumAtoms() != 0 and protein.NumAtoms() != 0


def write_complex_pdb(
    pdb_file: str,
    sdf_file: str,
    out_path: str,
    ligand_name: str = "LIG",
    method: str = "pymol",
) -> bool:
    """
    Combine protein PDB and ligand SDF into a single complex PDB file.

    Args:
        pdb_file: Path to the protein PDB file
        sdf_file: Path to the ligand SDF file
        out_path: Output path for the combined complex
        ligand_name: Residue name for the ligand (default: 'LIG')
        method: Method to use ('pymol' or 'obabel')

    Returns:
        bool: True if successful
    """
    if method == "obabel":
        lig_pdb = sdf_file.replace(".sdf", ".pdb")
        ret1 = os.system(f"obabel {sdf_file} -O {lig_pdb}")
        ret2 = os.system(f"obabel {pdb_file} {lig_pdb} -O {out_path}")
        return ret1 == 0 and ret2 == 0
    elif method == "pymol":
        try:
            from pymol import cmd

            cmd.reinitialize()
            cmd.load(pdb_file, "protein")
            cmd.load(sdf_file, "ligand")
            cmd.alter("ligand", f"resn='{ligand_name}'")
            cmd.save(out_path, "protein or ligand")
            cmd.reinitialize()
            return True
        except ImportError:
            print(
                "Error: PyMOL not available. Install pymol or use --method obabel",
                file=sys.stderr,
            )
            return False
    else:
        print(
            f"Error: Unknown method '{method}'. Use 'pymol' or 'obabel'.",
            file=sys.stderr,
        )
        return False


if __name__ == "__main__":
    sys.exit(main())
