"""The shipped defaults for the decode repair and the sampler guard, pinned.

A mismatch between two layers is the dangerous failure here: the CLI could default the
repair on while `scriptutil` defaults it off, or -- worst -- bond deletion could be
re-enabled somewhere without anyone naming it, which silently converts valence failures
into disconnected molecules. So this pins the values AND pins that all eight entrypoints
declare them identically.
"""

import ast
import pathlib
import sys
import unittest

REPO = pathlib.Path(__file__).resolve().parents[3]

ENTRYPOINTS = [
    "flowr/gen/generate_from_pdb.py",
    "flowr/gen/generate_from_lmdb.py",
    "flowr/gen/generate_from_lmdb_mol.py",
    "flowr/gen/generate_from_smol.py",
    "flowr/gen/generate_from_sdf_mol.py",
    "flowr/gen/generate_from_pdb_selective.py",
    "flowr/predict/predict_from_pdb.py",
    "flowr/predict/predict_from_lmdb.py",
]

FLAGS = (
    "--ligand_valence_repair",
    "--no_ligand_valence_repair",
    "--ligand_valence_repair_allow_bond_deletion",
    "--ligand_valence_repair_max_edits",
    "--ligand_valence_repair_top_k",
    "--ligand_valence_repair_max_states",
    "--cat_noise_euler_guard",
    "--no_cat_noise_euler_guard",
)


def _add_argument_calls(path):
    """{flag: (action, default)} for every `parser.add_argument` in one entrypoint."""
    tree = ast.parse((REPO / path).read_text())
    found = {}
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
            continue
        if node.func.attr != "add_argument" or not node.args:
            continue
        first = node.args[0]
        if not (isinstance(first, ast.Constant) and isinstance(first.value, str)):
            continue
        kw = {k.arg: k.value for k in node.keywords}
        action = kw.get("action")
        default = kw.get("default")
        found[first.value] = (
            action.value if isinstance(action, ast.Constant) else None,
            default.value if isinstance(default, ast.Constant) else "<absent>",
        )
    return found


class ShippedDefaultsTests(unittest.TestCase):
    def test_generate_from_pdb_defaults(self):
        """The end-to-end truth: what a bare command line actually produces."""
        from flowr.gen.generate_from_pdb import get_args

        argv = [
            "x", "--pdb_file", "a", "--ligand_file", "b",
            "--pocket_type", "holo", "--arch", "pocket",
        ]
        old = sys.argv
        try:
            sys.argv = list(argv)
            args = get_args()
            # The repair is ON: it is gated on a build that already returned None, so it can
            # only ADD molecules -- measured on 1500 generations, the repaired population was
            # the baseline population plus four, with nothing altered and nothing lost.
            self.assertTrue(args.ligand_valence_repair)
            # Bond deletion stays FORBIDDEN: "no bond" is a bond class like any other, so
            # deleting is often the cheapest escape from an over-valence -- and it can split
            # the molecule, trading a valence failure for a disconnected one.
            self.assertFalse(args.ligand_valence_repair_allow_bond_deletion)
            # The guard is ON as a correctness fix, not for a measured gain: over 4000
            # generations it moved neither build yield (35 -> 36 failures) nor PoseBusters
            # validity (0.9963 -> 0.9963). The Euler step genuinely is not a valid
            # probability step in the window it silences, and at low integration counts
            # that window is a large fraction of the trajectory.
            self.assertTrue(args.cat_noise_euler_guard)
            self.assertEqual(args.ligand_valence_repair_max_edits, 2)
            self.assertEqual(args.ligand_valence_repair_top_k, 4)
            self.assertEqual(args.ligand_valence_repair_max_states, 200)

            sys.argv = list(argv) + ["--no_ligand_valence_repair"]
            self.assertFalse(get_args().ligand_valence_repair)
            sys.argv = list(argv) + ["--no_cat_noise_euler_guard"]
            self.assertFalse(get_args().cat_noise_euler_guard)
        finally:
            sys.argv = old

    def test_every_entrypoint_declares_the_flags_identically(self):
        reference = None
        for path in ENTRYPOINTS:
            with self.subTest(entrypoint=path):
                calls = _add_argument_calls(path)
                declared = {f: calls.get(f) for f in FLAGS}
                for flag in FLAGS:
                    self.assertIsNotNone(declared[flag], f"{path} is missing {flag}")
                if reference is None:
                    reference = declared
                else:
                    self.assertEqual(declared, reference, f"{path} diverged")

    def test_the_declared_values_are_the_intended_ones(self):
        calls = _add_argument_calls(ENTRYPOINTS[0])
        self.assertEqual(calls["--ligand_valence_repair"], ("store_true", True))
        self.assertEqual(calls["--no_ligand_valence_repair"], ("store_false", "<absent>"))
        # store_true with no default == False. Anything else here re-enables deletion.
        self.assertEqual(
            calls["--ligand_valence_repair_allow_bond_deletion"], ("store_true", "<absent>")
        )
        self.assertEqual(calls["--cat_noise_euler_guard"], ("store_true", True))
        self.assertEqual(calls["--no_cat_noise_euler_guard"], ("store_false", "<absent>"))


class EveryConstructionSiteTests(unittest.TestCase):
    """The knobs must reach EVERY model, not just the generation entrypoints.

    The gap this pins: `load_model` (inference) and `build_model` (training) are separate
    constructors. If only the first carries the settings, the in-training VALIDATION panel
    samples and builds molecules differently from the shipped entrypoints -- so its validity
    is not the validity anyone actually gets, and a training curve and a test-split number
    disagree for a reason that has nothing to do with the model.
    """

    SCRIPTUTIL = REPO / "flowr/scriptutil.py"
    TRAINERS = ["flowr/train.py", "flowr/train_mol.py", "flowr/finetune.py"]

    def test_every_integrator_construction_passes_the_guard(self):
        tree = ast.parse(self.SCRIPTUTIL.read_text())
        sites = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "Integrator"
        ]
        self.assertGreaterEqual(len(sites), 5, "expected 2 inference + 3 training sites")
        for i, node in enumerate(sites):
            with self.subTest(site=i, line=node.lineno):
                names = {k.arg for k in node.keywords}
                self.assertIn(
                    "cat_noise_euler_guard", names,
                    f"Integrator at scriptutil.py:{node.lineno} does not pass the guard",
                )

    def test_every_model_hparams_dict_carries_the_repair_settings(self):
        """Both inference loaders and all three training builders."""
        text = self.SCRIPTUTIL.read_text()
        self.assertGreaterEqual(
            text.count('"ligand_valence_repair"'), 5,
            "expected the repair key in 2 inference loaders + 3 training builders",
        )
        self.assertGreaterEqual(
            text.count('"ligand_valence_repair_allow_bond_deletion"'), 5
        )

    def test_the_training_entrypoints_expose_the_flags(self):
        for path in self.TRAINERS:
            with self.subTest(entrypoint=path):
                calls = _add_argument_calls(path)
                self.assertEqual(calls.get("--ligand_valence_repair"), ("store_true", True))
                self.assertEqual(
                    calls.get("--no_ligand_valence_repair"), ("store_false", "<absent>")
                )
                self.assertEqual(calls.get("--cat_noise_euler_guard"), ("store_true", True))
                self.assertEqual(
                    calls.get("--no_cat_noise_euler_guard"), ("store_false", "<absent>")
                )
                # deletion stays forbidden on the training side too
                self.assertEqual(
                    calls.get("--ligand_valence_repair_allow_bond_deletion"),
                    ("store_true", "<absent>"),
                )


if __name__ == "__main__":
    unittest.main()
