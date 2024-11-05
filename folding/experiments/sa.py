### This script is used to run a sensitivity search for a protein folding simulation.
### Used for benchmarking.
import math
import os
import time
from typing import Dict, List
from tqdm import tqdm
from pathlib import Path
import base64
import pandas as pd
from box import Box
import copy
import glob
import plotly.express as px
import openmm as mm
import openmm.app as app
import openmm.unit as unit
import subprocess
from folding.base.simulation import OpenMMSimulation
from folding.utils.uids import custom_loop_annealing
from folding.utils.ops import select_random_pdb_id, load_and_sample_random_pdb_ids, check_if_directory_exists
from folding.validators.hyperparameters import HyperParameters
from folding.utils.opemm_simulation_config import SimulationConfig
from folding.validators.protein import Protein
from folding.utils.custom_reporters import ExitFileReporter, LastTwoCheckpointsReporter, CustomStateReporter
import numpy as np
import requests
import json
import pickle
import shutil

ROOT_DIR = Path(__file__).resolve().parents[2]
PDB_IDS, input_source = load_and_sample_random_pdb_ids(
    root_dir=ROOT_DIR, filename="pdb_ids.pkl", input_source="rcsb"
)  # TODO: Currently this is a small list of PDBs without MISSING flags.

ROOT_PATH = Path(__file__).parent
SEED = 42
SIMULATION_STEPS = {"nvt": 50000, "npt": 75000, "md_0_1": 500000}
CHECKPOINT_INTERVAL = 3010

def configure_commands(
    state: str,
    seed: int,
    system_config: SimulationConfig,
    pdb_obj: app.PDBFile,
    output_dir: str,
    run_before_annealing: int,
    annealing_loop: int,
    middle_cpt: int,
    CHECKPOINT_INTERVAL: int = 10000,
    STATE_DATA_REPORTER_INTERVAL: int = 10,
    EXIT_REPORTER_INTERVAL: int = 10,
) -> Dict[str, List[str]]:
    annealing_loop_cpt = annealing_loop + middle_cpt
    simulation, integrator, _ = OpenMMSimulation().create_simulation(
        pdb=pdb_obj,
        system_config=system_config.get_config(),
        seed=seed,
        state=state,
    )
    simulation.reporters.append(
        LastTwoCheckpointsReporter(
            file_prefix=f"{output_dir}/{state}",
            reportInterval=CHECKPOINT_INTERVAL,
            amount_first_loop=run_before_annealing,
            annealing_loop_cpt=annealing_loop_cpt,
        )
    )
    simulation.reporters.append(
        CustomStateReporter(
            file=f"{output_dir}/{state}_origin.log",
            reportInterval=STATE_DATA_REPORTER_INTERVAL,
            skip_start=run_before_annealing*CHECKPOINT_INTERVAL,
            skip_end=(annealing_loop_cpt + run_before_annealing)*CHECKPOINT_INTERVAL,
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

    return simulation, integrator


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

    print("Searching parameter space for pdb {pdb_id}")

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
            print(f"Error occurred for pdb_id aaaaa {pdb_id}: {e}")
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


def sample_pdb(exclude: List = [], pdb_id: str = None):
    return pdb_id or select_random_pdb_id(PDB_IDS, exclude=exclude)


def extact_energies(state: str, data_directory: str):
    check_log_file = pd.read_csv(os.path.join(data_directory, f"{state}.log"))

    return check_log_file["Potential Energy (kJ/mole)"].values

def cpt_file_mapper(output_dir: str):
    return f"{output_dir}/em.cpt"



if __name__ == "__main__":
    num_experiments = 1
    temperatures = [255.25]
    friction = 0.94
    pdbs_to_exclude = []
    system_kwargs = {"temperature": temperatures[0], "friction": friction}
    while num_experiments > 0:
        pdb_id = "6clh"

        try:
            protein, event = create_new_challenge(pdb_id=pdb_id, system_kwargs=system_kwargs)
        except Exception as e:
            print(f"Error occurred for pdb_id {pdb_id}: {e}")
            pdbs_to_exclude.append(pdb_id)
            continue

        pdb_obj: app.PDBFile = protein.load_pdb_file(protein.pdb_location)
        pdb_len=1074036
        jump_temp = 6
        run_before_annealing = 5
        run_after_annealing = 20
        middle_cpt=20

        annealing_loop, jump_temp, run_before_annealing, run_after_annealing = custom_loop_annealing(pdb_len)
        output_dir = protein.validator_directory
        with open(f"{output_dir}/md_0_1.log", 'w') as file:
            # Write the header line
            pass
        for temperature in temperatures:
            system_config = copy.deepcopy(protein.system_config)
            print(system_config)
            system_config.temperature = float(temperature)
            system_config.friction = float(friction)

            for state, steps_to_run in SIMULATION_STEPS.items():
                # Creates the simulation object needed for the stage.
                temp_state = state + f"_temp_{temperature}"
                
                simulation, integrator = configure_commands(
                    state=state,
                    seed=protein.system_config.seed,
                    system_config=system_config,
                    pdb_obj=pdb_obj,
                    output_dir=output_dir,
                    run_before_annealing=run_before_annealing,
                    annealing_loop=annealing_loop*cycles,
                    middle_cpt=middle_cpt,
                )
                print(
                    f"Running {tmp_state} for {steps_to_run} steps for pdb {pdb_id}"
                )

                simulation.loadCheckpoint(cpt_file_mapper(output_dir))
                middle = (annealing_loop - 1) // 2

                # simulation.minimizeEnergy()
                for i in range(annealing_loop):
                    if i < middle:
                        temperature_tmp = temperature_tmp + jump_temp
                        integrator.setTemperature(temperature_tmp * unit.kelvin)
                        simulation.step(CHECKPOINT_INTERVAL)
                    elif i == middle:
                        simulation.step(CHECKPOINT_INTERVAL * middle_cpt)
                    else:
                        temperature_tmp = temperature_tmp - jump_temp
                        integrator.setTemperature(temperature_tmp * unit.kelvin)
                        simulation.step(CHECKPOINT_INTERVAL)
                print(f"Current step: {simulation.currentStep}, energy: {simulation.context.getState(getEnergy=True).getPotentialEnergy() / unit.kilojoules_per_mole}")
                start_time = time.time()
                simulation.step(steps_to_run)
                simulation_time = time.time() - start_time
                event[f"{state}_time"] = simulation_time

                energy_array = extact_energies(
                    state=f"{tmp_state}_origin", data_directory=protein.validator_directory
                )
                event[f"{tmp_state}_energies_temp_{temperature}"] = energy_array.tolist()

                fig = px.scatter(
                    energy_array,
                    title=f"Energy array for {pdb_id} for tmp_state {tmp_state} for temperature {temperature}",
                    labels={"index": "Step", "value": "energy"},
                    height=600,
                    width=1400,
                )
                
                fig.write_image(os.path.join(output_dir, f"{tmp_state}_energy.png"))

        num_experiments = num_experiments - 1
        pdbs_to_exclude.append(pdb_id)
