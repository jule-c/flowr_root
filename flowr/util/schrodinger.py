import os
import re
import yaml
import shutil
import tempfile
import subprocess as sp
from pathlib import Path


class SchrodingerJob:
    """Runner for Schrodinger applications and programs"""

    def __init__(self, config_file=None):
        """
        config: yaml_file
        """

        self.schrodinger_path = os.environ["SCHRODINGER"]
        self.license_re = re.compile(r"Users\s+of\s+(.*):.*of\s+(\d+)\s+licenses\s+issued;.*of\s+(\d+)\s+licenses\s+in\s+use")

        if config_file is None:
            self.config = {}
        else:
            self.config = yaml.safe_load(open(config_file, "rb"))

        # Set env variables for schrodinger
        if self.has_license():
            os.environ["SCHROD_LICENSE_FILE"] = self.config["SCHROD_LICENSE_FILE"]

    def has_license(self) -> bool:
        """``True`` if the system has a Schrodinger license, ``False`` otherwise."""
        return "SCHROD_LICENSE_FILE" in self.config

    def prepwizard(self, complex_path, output_path, log_path, max_attempts=2):
        current_dir = str(Path(os.getcwd()).resolve())
        complex_path = str(Path(complex_path).resolve())
        log_path = str(Path(log_path).resolve())

        output_path = Path(output_path)
        output_file = str(output_path.name)

        # Need to change to output file directory for schrodinger to work
        os.chdir(str(output_path.parent.resolve()))

        if output_path.exists():
            print(f"Found output existing output file at {str(output_path.resolve())}")
            output_path.unlink()

        for _ in range(max_attempts):
            try:
                cmd = str((Path(self.schrodinger_path) / "utilities/prepwizard").resolve())
                cmds = [cmd, "-fillsidechains", "-WAIT", complex_path, output_file]
                out = sp.run(cmds, capture_output=True)
            except Exception as e:
                print("Error when running prepwizard subprocess...")
                print(e)

            returncode = out.returncode
            if returncode == 0:
                break

        os.chdir(current_dir)

        # If everything seems ok return None
        if returncode == 0 and output_path.exists():
            return None

        # Otherwise return the subprocess output from the final attempt
        return out

    def mae2pdb(self, complex_path, output_path, log_path, max_attemps=5):
        current_dir = os.getcwd()
        complex_path = os.path.abspath(complex_path)

        with tempfile.TemporaryDirectory() as tmp_dir:
            returncode = -1
            cmd = os.path.join(self.schrodinger_path, "utilities/structconvert")

            for attempt in range(max_attemps):
                try:
                    os.chdir(tmp_dir)
                    shutil.copy(complex_path, "_complex.mae")
                    out = sp.run([cmd, "_complex.mae", "_complex.pdb"], capture_output=True)
                    os.chdir(current_dir)

                    returncode = out.returncode
                    if returncode == 0:
                        break

                except BaseException:
                    os.chdir(current_dir)

            self._save_log("mae2pdb", log_path, os.path.join(tmp_dir, "_complex.log"))

            if returncode != 0:
                return False

            if not os.path.exists(os.path.join(tmp_dir, "_complex.pdb")):
                return False

            shutil.copy(os.path.join(tmp_dir, "_complex.pdb"), output_path)
            return returncode == 0

    def _save_log(self, log_name, log_path, tmp_log_path):
        log_fobj = open(log_path, "a")

        log_fobj.write(f"---{log_name.upper()} LOG---\n")
        if not os.path.exists(tmp_log_path):
            log_fobj.write("No log found\n")

        else:
            log_lines = open(tmp_log_path).readlines()
            for line in log_lines:
                log_fobj.write(line)

        log_fobj.write(f"---END {log_name.upper()} LOG---\n")
        log_fobj.close()

    # def make_grid(self, complex_path, output_path, log_path, max_attemps=5):
    #     current_dir = os.getcwd()
    #     complex_path = os.path.abspath(complex_path)

    #     with tempfile.TemporaryDirectory() as tmp_dir:
    #         returncode = -1
    #         cmd = os.path.join(self.schrodinger_path, "utilities/generate_glide_grids")
    #         for attempt in range(max_attemps):
    #             try:
    #                 os.chdir(tmp_dir)
    #                 shutil.copy(complex_path, "_complex.mae")
    #                 out = sp.run(
    #                     [
    #                         cmd,
    #                         "-WAIT",
    #                         "-rec",
    #                         "_complex.mae",
    #                         "-lig_asl",
    #                         'res.ptype "INH "',
    #                     ],
    #                     capture_output=True,
    #                 )
    #                 os.chdir(current_dir)
    #                 returncode = out.returncode
    #                 if returncode == 0:
    #                     break
    #             except BaseException:
    #                 os.chdir(current_dir)

    #         self.save_log(
    #             "grid", log_path, os.path.join(tmp_dir, "generate_glide_grids_run.log")
    #         )

    #         if returncode != 0:
    #             return False

    #         if not os.path.exists(
    #             os.path.join(tmp_dir, "generate_glide_grids_run.log")
    #         ):
    #             return False

    #         for line in open(
    #             os.path.join(tmp_dir, "generate_glide_grids_run.log")
    #         ).readlines()[::-1]:
    #             if (
    #                 "Output grids file: generate-grids-gridgen.zip" in line
    #                 and os.path.exists(
    #                     os.path.join(tmp_dir, "generate-grids-gridgen.zip")
    #                 )
    #             ):
    #                 shutil.copy(
    #                     os.path.join(tmp_dir, "generate-grids-gridgen.zip"), output_path
    #                 )
    #                 return returncode == 0
    #         return False

    # def docking_score(self, grid_path, ligand_path, output_path, log_path, max_attemps=5):
    #     current_dir = os.getcwd()
    #     grid_path = os.path.abspath(grid_path)
    #     ligand_path = os.path.abspath(ligand_path)

    #     glide_in = [
    #         "GRIDFILE grid.zip",
    #         "DOCKING_METHOD inplace",
    #         "PRECISION SP",
    #         "COMPRESS_POSES FALSE",
    #         "POSE_OUTTYPE ligandlib_sd",
    #         "LIGANDFILE ligand.sdf",
    #         "POSTDOCKSTRAIN TRUE",
    #         "CALC_INPUT_RMS FALSE",
    #         "WRITE_CSV TRUE",
    #     ]

    #     tokens, attempts = 0, 0
    #     while (tokens == 0) and (attempts < max_attemps):
    #         total, used = self.get_tokens()["GLIDE_SP_DOCKING"]
    #         tokens = total - used
    #         time.sleep(5)
    #         attempts += 1

    #     if tokens == 0:
    #         raise TimeoutError("No available tokens to run glide")

    #     with tempfile.TemporaryDirectory() as tmp_dir:
    #         returncode = -1
    #         cmd = os.path.join(self.schrodinger_path, "glide")
    #         for attempt in range(max_attemps):
    #             try:
    #                 os.chdir(tmp_dir)
    #                 shutil.copy(grid_path, "grid.zip")
    #                 shutil.copy(ligand_path, "ligand.sdf")
    #                 with open("glide_in", "w") as fobj:
    #                     fobj.write("\n".join(glide_in))
    #                 out = sp.run(
    #                     [cmd, "-WAIT", "-HOST", "localhost", "glide_in"],
    #                     capture_output=True,
    #                 )
    #                 os.chdir(current_dir)
    #                 returncode = out.returncode
    #                 if returncode == 0:
    #                     break
    #             except BaseException:
    #                 os.chdir(current_dir)
    #         self.save_log(
    #             "docking_score", log_path, os.path.join(tmp_dir, "glide_in.log")
    #         )

    #         if returncode != 0:
    #             return False

    #         if not os.path.exists(os.path.join(tmp_dir, "glide_in.log")):
    #             return False
    #         for line in open(os.path.join(tmp_dir, "glide_in.log")).readlines()[::-1]:
    #             if "Pose file glide_in_lib.sdf was written" in line and os.path.exists(
    #                 os.path.join(tmp_dir, "glide_in.csv")
    #             ):
    #                 shutil.copy(os.path.join(tmp_dir, "glide_in.csv"), output_path)
    #                 return returncode == 0
    #         return False

    # def dock(self, grid_path, ligand_path, output_path, log_path, max_attemps=5):
    #     current_dir = os.getcwd()
    #     grid_path = os.path.abspath(grid_path)
    #     ligand_path = os.path.abspath(ligand_path)

    #     glide_in = [
    #         "GRIDFILE grid.zip",
    #         "PRECISION SP",
    #         "COMPRESS_POSES FALSE",
    #         "POSE_OUTTYPE ligandlib_sd",
    #         "LIGANDFILE ligand.sdf",
    #         "POSTDOCKSTRAIN TRUE",
    #         "CALC_INPUT_RMS FALSE",
    #         "WRITE_CSV TRUE",
    #     ]

    #     tokens, attempts = 0, 0
    #     while (tokens == 0) and (attempts < max_attemps):
    #         total, used = self.get_tokens()["GLIDE_SP_DOCKING"]
    #         tokens = total - used
    #         time.sleep(5)
    #         attempts += 1

    #     if tokens == 0:
    #         raise TimeoutError("No available tokens to run glide")

    #     with tempfile.TemporaryDirectory() as tmp_dir:
    #         returncode = -1
    #         cmd = os.path.join(self.schrodinger_path, "glide")
    #         for attempt in range(max_attemps):
    #             try:
    #                 os.chdir(tmp_dir)
    #                 shutil.copy(grid_path, "grid.zip")
    #                 shutil.copy(ligand_path, "ligand.sdf")
    #                 with open("glide_in", "w") as fobj:
    #                     fobj.write("\n".join(glide_in))
    #                 out = sp.run(
    #                     [cmd, "-WAIT", "-HOST", "localhost", "glide_in"],
    #                     capture_output=True,
    #                 )
    #                 os.chdir(current_dir)
    #                 returncode = out.returncode
    #                 if returncode == 0:
    #                     break
    #             except BaseException:
    #                 os.chdir(current_dir)
    #         self.save_log(
    #             "docking_score", log_path, os.path.join(tmp_dir, "glide_in.log")
    #         )

    #         if returncode != 0:
    #             return False

    #         if not os.path.exists(os.path.join(tmp_dir, "glide_in.log")):
    #             return False
    #         for line in open(os.path.join(tmp_dir, "glide_in.log")).readlines()[::-1]:
    #             if "Pose file glide_in_lib.sdf was written" in line and os.path.exists(
    #                 os.path.join(tmp_dir, "glide_in.csv")
    #             ):
    #                 shutil.copy(os.path.join(tmp_dir, "glide_in.csv"), os.path.join(output_path, "score.csv"))
    #                 shutil.copy(os.path.join(tmp_dir, "glide_in_lib.sdf"),  os.path.join(output_path, "ligand.sdf"))
                    
    #                 return returncode == 0
    #         return False

    # def get_tokens(self, max_attemps=5):
    #     returncode = -1
    #     licadmin_cmd = os.path.join(self.schrodinger_path, "licadmin")
    #     for attempt in range(max_attemps):
    #         try:
    #             out = sp.run([licadmin_cmd, "STAT"], capture_output=True)
    #             returncode = out.returncode
    #             if returncode == 0:
    #                 break
    #         except BaseException:
    #             pass

    #     if returncode != 0:
    #         raise RuntimeError("Max attemps reached for licadmin")

    #     tokens = {}
    #     for line in out.stdout.decode().splitlines():
    #         if match := re.match(self.license_re, line):
    #             name, total, used = match.groups()
    #             tokens[name] = (int(total), int(used))
    #     return tokens
