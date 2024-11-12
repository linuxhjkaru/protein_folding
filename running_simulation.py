### This script is used to run a sensitivity search for a protein folding simulation.
### Used for benchmarking.
 
import glob
import os
import sys
import time
from typing import Dict, List
from tqdm import tqdm
from pathlib import Path
import pandas as pd
from box import Box  # install using pip install box
import copy
import plotly.express as px
import openmm.app as app
import openmm.unit as unit
from folding.base.simulation import OpenMMSimulation
from folding.utils.ops import select_random_pdb_id, load_and_sample_random_pdb_ids, check_if_directory_exists
from folding.utils.uids import filter_log_file
from folding.validators.hyperparameters import HyperParameters
from folding.utils.opemm_simulation_config import SimulationConfig
from folding.validators.protein import Protein
from folding.utils.reporters import ExitFileReporter, LastTwoCheckpointsReporter
import pickle
import shutil
import numpy as np


WORKING_DIR = os.getcwd()
ROOT_DIR = Path(__file__).resolve().parents[2]
PDB_IDS, input_source = load_and_sample_random_pdb_ids(
    root_dir=f"{WORKING_DIR}/", filename="pdb_ids.pkl", input_source="rcsb"
)  # TODO: Currently this is a small list of PDBs without MISSING flags.
 
ROOT_PATH = Path(__file__).parent
SEED = 725
SIMULATION_STEPS = {"md_0_1": 499001}
CHECKPOINT_INTERVAL = 5010

def attach_files(
    files_to_attach: List, state: str, 
):
    md_output = {}
    for filename in files_to_attach:
        try:
            with open(filename, "rb") as f:
                filename = filename.split("/")[
                    -1
                ]  # remove the directory from the filename
                md_output[filename] = f.read()
        except Exception as e:
            print(f"Failed to read file {filename!r} with error: {e}")

    return md_output

def attach_files_to_synapse(
    data_directory: str,
    state: str,
):
    """load the output files as bytes and add to md_output

    Args:
        data_directory (str): directory where the miner is holding the necessary data for the validator.
        state (str): the current state of the simulation

    state is either:
     1. nvt
     2. npt
     3. md_0_1
     4. finished

    Returns:
        md_output attached
    """

    md_output = {}  # ensure that the initial state is empty

    try:
        state_files = os.path.join(data_directory, f"{state}")
        # This should be "state.cpt" and "state_old.cpt"
        all_state_files = glob.glob(f"{state_files}*")  # Grab all the state_files
        print(state_files, all_state_files)

        if len(all_state_files) == 0:
            raise FileNotFoundError(
                f"No files found for {state}"
            )  # if this happens, goes to except block

        md_output = attach_files(files_to_attach=all_state_files, state=state)

    except Exception as e:
        print(
            f"Failed to attach files for pdb with error: {e}"
        )
        md_output = {}
    finally:
        md_output_sizes = list(map(len, md_output.values()))
        print(f"MD OUTPUT SIZES: {md_output_sizes}")
        return md_output

def configure_commands(
    state: str,
    seed: int,
    system_config: SimulationConfig,
    pdb_obj: app.PDBFile,
    output_dir: str,
    CHECKPOINT_INTERVAL: int = 10000,
    STATE_DATA_REPORTER_INTERVAL: int = 10,
    EXIT_REPORTER_INTERVAL: int = 10,
) -> Dict[str, List[str]]:
    simulation, _, _ = OpenMMSimulation().create_simulation(
        pdb=pdb_obj,
        system_config=system_config.get_config(),
        seed=seed,
        state=state,
    )
    simulation.reporters.append(
        LastTwoCheckpointsReporter(
            file_prefix=f"{output_dir}/{state}",
            reportInterval=CHECKPOINT_INTERVAL,
        )
    )
    simulation.reporters.append(
        app.StateDataReporter(
            file=f"{output_dir}/{state}.log",
            reportInterval=STATE_DATA_REPORTER_INTERVAL,
            step=True,
            potentialEnergy=True,
        )
    )
    simulation.reporters.append(
        ExitFileReporter(
            filename=f"{output_dir}/{state}",
            reportInterval=EXIT_REPORTER_INTERVAL,
            file_prefix=state,
        )
    )

    return simulation

def extact_energies(state: str, data_directory: str):
    check_log_file = pd.read_csv(os.path.join(data_directory, f"{state}.log"))

    return check_log_file["Potential Energy (kJ/mole)"].values


def cpt_file_mapper(output_dir: str, state: str):
    return f"{output_dir}/em.cpt"
 
def create_new_challenge(pdb_id: str,system_kwargs: dict) -> Dict:
    """Create a new challenge by sampling a random pdb_id and running a hyperparameter search
    using the try_prepare_challenge function.
    Args:
        exclude (List): list of pdb_ids to exclude from the search
    Returns:
        Dict: event dictionary containing the results of the hyperparameter search
    """
 
    forward_start_time = time.time()
 
    # Perform a hyperparameter search until we find a valid configuration for the pdb
    # print(f"Attempting to prepare challenge for pdb {pdb_id}")
    protein, event = try_prepare_challenge(pdb_id=pdb_id, input_source=input_source, system_kwargs=system_kwargs)
 
    if event.get("validator_search_status"):
        print(f"✅✅ Successfully created challenge for pdb_id {pdb_id} ✅✅")
    else:
        # forward time if validator step fails
        event["hp_search_time"] = time.time() - forward_start_time
        print(
            f"❌❌ All hyperparameter combinations failed for pdb_id {pdb_id}.. Skipping! ❌❌"
        )
 
    return protein, event
 
 
def try_prepare_challenge(pdb_id: str, input_source: str, system_kwargs: dict) -> Dict:
    """Attempts to setup a simulation environment for the specific pdb & config
    Uses a stochastic sampler to find hyperparameters that are compatible with the protein
    """
 
    # exclude_in_hp_search = parse_config(config)
    hp_sampler = HyperParameters()
 
    print(f"Searching parameter space for pdb {pdb_id}")
 
    hp_sampler_time = time.time()
    event = {}

    hps = {
        "ff": system_kwargs["ff"],
        "water": system_kwargs["water"],
        "box": system_kwargs["box"],
    }
    config = Box({"input_source": input_source, "force_use_pdb": False})
    print("system_kwargs = ", system_kwargs)
    new_system_kwargs = {
      "temperature": system_kwargs['temperature'],
      "friction": system_kwargs['friction'],
    }
    protein = Protein(pdb_id=pdb_id, config=config, system_kwargs=new_system_kwargs, **hps)
    # protein.md_inputs = protein.read_and_return_files(filenames=protein.input_files)

    try:
        print(
            f"Launching {protein.pdb_id} Protein Job with the following configuration\nff : {protein.ff}\nbox : {protein.box}\nwater : {protein.water}"
        )

        ## Setup the protein directory and sample a random pdb_id if not provided
        protein.gather_pdb_id()
        protein.setup_pdb_directory()

        
        # self.generate_input_files()
        print(f"Changing path to {protein.pdb_directory}")
        os.chdir(protein.pdb_directory)

        # Here we are going to change the path to a validator folder, and move ALL the files except the pdb file
        check_if_directory_exists(output_directory=protein.validator_directory)
        config_folder_path = f"{WORKING_DIR}/miner-config/{pdb_id}"
        pkl_file_path = os.path.join(config_folder_path, f"config_{pdb_id}.pkl")
        cpt_file_path = os.path.join(config_folder_path, "em.cpt")
        pdb_file_path = os.path.join(config_folder_path, f"{pdb_id}.pdb")
        shutil.copy(pkl_file_path, f"{protein.validator_directory}/")
        shutil.copy(cpt_file_path, f"{protein.validator_directory}/")
        shutil.copy(pdb_file_path, f"{WORKING_DIR}/data/{pdb_id}/")
        # Move all files
        cmd = f'find . -maxdepth 1 -type f ! -name "*.pdb" -exec mv {{}} {protein.validator_directory}/ \;'
        print(f"Moving all files except pdb to {protein.validator_directory}")
        os.system(cmd)

        # Create a validator directory to store the files
        check_if_directory_exists(output_directory=protein.validator_directory)

        # Read the files that should exist now based on generate_input_files.
        protein.md_inputs = protein.read_and_return_files(filenames=protein.input_files)

        protein.save_files(
            files=protein.md_inputs,
            output_directory=protein.validator_directory,
            write_mode="w",
        )

    except Exception as e:
        print(f"Error occurred for pdb_id aaaaa {pdb_id}: {e}")
        event["validator_search_status"] = False

    

    finally:
        event["pdb_id"] = pdb_id
        event["hp_sample_time"] = time.time() - hp_sampler_time
        event["pdb_complexity"] = [dict(protein.pdb_complexity)]
        event["init_energy"] = protein.init_energy

        if "validator_search_status" not in event:
            print("✅✅ Simulation ran successfully! ✅✅")
            event["validator_search_status"] = True  # simulation passed!
            # break out of the loop if the simulation was successful
 
    return protein, event

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Please input pdb_id Usage: python client.py pdb_id")
        sys.exit(1)

    pdb_id = sys.argv[1]

    # Check existence of pdb, config and current checkpoint file
    config_folder_path = f"{WORKING_DIR}/miner-config/{pdb_id}"
    if not os.path.exists(config_folder_path):
        print(f"Error: Folder '{config_folder_path}' does not exist.")
        sys.exit()

    pkl_file_path = os.path.join(config_folder_path, f"config_{pdb_id}.pkl")
    cpt_file_path = os.path.join(config_folder_path, "em.cpt")

    if not os.path.exists(pkl_file_path):
        print(f"Error: 'config_{pdb_id}.pkl' does not exist in '{config_folder_path}'.")
        sys.exit()
    if not os.path.exists(cpt_file_path):
        print(f"Error: 'em.cpt' does not exist in '{config_folder_path}'.")
        sys.exit()
    
    # Read config and create protein class
    with open(pkl_file_path, 'rb') as f:
        data = pickle.load(f)

    system_kwargs = data.to_dict()
    try:
        protein, event = create_new_challenge(pdb_id=pdb_id, system_kwargs=system_kwargs)
    except Exception as e:
        print(f"Error occurred for pdb_id {pdb_id}: {e}")
        sys.exit()
    
    pdb_obj: app.PDBFile = protein.load_pdb_file(protein.pdb_location)
    output_dir = protein.validator_directory

    try:
        os.rmdir(f"{WORKING_DIR}/miner-data/{pdb_id}")
        print(f"Successfully removed empty folder: {WORKING_DIR}/minerdata/{pdb_id}")
    except OSError as e:
        print(f"Already delete or Error: {WORKING_DIR}/minerdata/{pdb_id} : {e.strerror}")

    system_config = copy.deepcopy(protein.system_config)
    system_config.seed = system_kwargs["seed"]
    temperature = system_config.temperature

    for state, steps_to_run in SIMULATION_STEPS.items():
        # Creates the simulation object needed for the stage.

        simulation = configure_commands(
            state=state,
            seed=protein.system_config.seed,
            system_config=system_config,
            pdb_obj=pdb_obj,
            output_dir=protein.validator_directory,
        )

        print(
            f"Running {state} for {steps_to_run} steps for pdb {pdb_id}"
        )

        simulation.loadCheckpoint(cpt_file_mapper(output_dir, state))

        start_time = time.time()
        # --------------------------------------------
        # SIMULATION PART
        # --------------------------------------------
        simulation.step(steps_to_run)
        simulation_time = time.time() - start_time
        event[f"{state}_time"] = simulation_time
        print(f"Simulation {state} took: {simulation_time} seconds")

        # --------------------------------------------
        # SIMULATION RUNNING PART
        # --------------------------------------------


        # --------------------------------------------
        # VALIDATION PART
        # --------------------------------------------
        print("----------------------------------")
        print("----------------------------------")
        print("PROCESSING VALIDATION ..... ")
        simulation.loadCheckpoint(f"{output_dir}/md_0_1.cpt")
        print(f"Current step md: {simulation.currentStep}, energy: {simulation.context.getState(getEnergy=True).getPotentialEnergy() / unit.kilojoules_per_mole}")
        md_output = attach_files_to_synapse(f"/root/protein_folding/data/{pdb_id}/validator", state)
        can_process = protein.process_md_output(
            md_output=md_output,
            seed=SEED,
            state=state
        )
        if can_process:
            reported_energy = protein.get_energy()
            print(f"reported_energy: {reported_energy}")
            is_valid, checked_energy, miner_energy = protein.is_run_valid()
            energy = np.median(checked_energy[-10:]) if is_valid else 0
            print(f"Is_valid: {is_valid}, energy: {np.median(checked_energy[-10:])}")

        # --------------------------------------------
        # VALIDATION PART
        # --------------------------------------------

        energy_array = extact_energies(
            state=state, data_directory=protein.validator_directory
        )
        event[f"{state}_energies_temp_{temperature}"] = energy_array.tolist()

        fig = px.scatter(
            energy_array,
            title=f"Energy array for {pdb_id} for state {state} for temperature {temperature}",
            labels={"index": "Step", "value": "energy"},
            height=600,
            width=1400,
        )
        fig.write_image(os.path.join(output_dir, f"{state}_energy.png"))
