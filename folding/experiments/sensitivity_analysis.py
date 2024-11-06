### This script is used to run a sensitivity search for a protein folding simulation.
### Used for benchmarking.

import os
import time
import wandb
from typing import Dict, List
from tqdm import tqdm
from pathlib import Path
import pandas as pd
from box import Box  # install using pip install box
import copy
import glob
import base64
import plotly.express as px
import openmm as mm
import openmm.app as app
import openmm.unit as unit
import numpy as np
from folding.base.simulation import OpenMMSimulation
from folding.utils.ops import select_random_pdb_id, load_and_sample_random_pdb_ids, get_tracebacks
from folding.validators.hyperparameters import HyperParameters
from folding.utils.opemm_simulation_config import SimulationConfig
from folding.validators.protein import Protein
from folding.utils.reporters import ExitFileReporter, LastTwoCheckpointsReporter

ROOT_DIR = Path(__file__).resolve().parents[2]
PDB_ID, input_source = load_and_sample_random_pdb_ids(
    root_dir=ROOT_DIR, filename="pdb_ids.pkl", input_source="rcsb"
)  # TODO: Currently this is a small list of PDBs without MISSING flags.
ROOT_PATH = Path(__file__).parent
SEED = 42
SIMULATION_STEPS = {"nvt": 5000, "npt": 5000, "md_0_1": 12000}


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
                md_output[filename] = base64.b64encode(f.read())
        except Exception as e:
            print(f"Failed to read file {filename!r} with error: {e}")
            get_tracebacks()

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
    CHECKPOINT_INTERVAL: int = 5000,
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


def create_new_challenge(pdb_id: str, system_kwargs: dict) -> Dict:
    """Create a new challenge by sampling a random pdb_id and running a hyperparameter search
    using the try_prepare_challenge function.
    Args:
        exclude (List): list of pdb_ids to exclude from the search
    Returns:
        Dict: event dictionary containing the results of the hyperparameter search
    """

    forward_start_time = time.time()

    # Perform a hyperparameter search until we find a valid configuration for the pdb
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

    for tries in tqdm(
        range(hp_sampler.TOTAL_COMBINATIONS), total=hp_sampler.TOTAL_COMBINATIONS
    ):
        hp_sampler_time = time.time()

        event = {"hp_tries": tries}
        sampled_combination: Dict = hp_sampler.sample_hyperparameters()

        hps = {
            "ff": sampled_combination["FF"],
            "water": sampled_combination["WATER"],
            "box": sampled_combination["BOX"],
        }

        config = Box({"input_source": input_source, "force_use_pdb": False})
        protein = Protein(pdb_id=pdb_id, config=config, system_kwargs=system_kwargs, **hps)

        try:
            protein.setup_simulation()

        except Exception as e:
            print(f"Error occurred for pdb_id {pdb_id}: {e}")
            event["validator_search_status"] = False

        finally:
            event["pdb_id"] = pdb_id
            event.update(hps)  # add the dictionary of hyperparameters to the event
            event["hp_sample_time"] = time.time() - hp_sampler_time
            event["pdb_complexity"] = [dict(protein.pdb_complexity)]
            event["init_energy"] = protein.init_energy
            # event["epsilon"] = protein.epsilon

            if "validator_search_status" not in event:
                print("✅✅ Simulation ran successfully! ✅✅")
                event["validator_search_status"] = True  # simulation passed!
                # break out of the loop if the simulation was successful
                break
            if tries == 3:
                print(f"Max tries reached for pdb_id {pdb_id} :x::x:")
                break

    return protein, event


def extact_energies(state: str, data_directory: str):
    check_log_file = pd.read_csv(os.path.join(data_directory, f"{state}.log"))

    return check_log_file["Potential Energy (kJ/mole)"].values


def cpt_file_mapper(output_dir: str, state: str):
    if state == "nvt":
        return f"{output_dir}/em.cpt"

    if "npt" in state:
        state = "nvt" + state.split("npt")[-1]

    if "md" in state:
        state = "npt" + state.split("md_0_1")[-1]

    return f"{output_dir}/{state}.cpt"


if __name__ == "__main__":
    num_experiments = 1
    temperatures = [300]
    # temperatures = [50, 100, 200, 300, 400, 500]
    friction = 1
    pdbs_to_exclude = []

    while num_experiments > 0:
        # You can specific your pdb_id for testing more
        # pdb_id = "a123"
        pdb_id = PDB_ID
        system_kwargs = {"temperature": temperatures[0], "friction": friction}

        try:
            protein, event = create_new_challenge(pdb_id=pdb_id, system_kwargs=system_kwargs)
        except Exception as e:
            print(f"Error occurred for pdb_id {pdb_id}: {e}")
            pdbs_to_exclude.append(pdb_id)
            continue

        pdb_obj: app.PDBFile = protein.load_pdb_file(protein.pdb_location)
        output_dir = protein.validator_directory

        for temperature in temperatures:
            system_config = copy.deepcopy(protein.system_config)
            system_config.temperature = float(temperature)

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
                simulation.step(steps_to_run)
                simulation_time = time.time() - start_time
                event[f"{state}_time"] = simulation_time
                print(f"Simulation {state} took: {simulation_time} seconds")
                if state == "md_0_1":
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

        num_experiments = num_experiments - 1
        pdbs_to_exclude.append(pdb_id)
