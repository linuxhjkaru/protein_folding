from abc import ABC, abstractmethod
import time
from typing import Tuple
import functools
import openmm as mm
from openmm import app
from openmm import unit
from folding.utils.opemm_simulation_config import SimulationConfig
from folding.utils.uids import check_available_gpus


class GenericSimulation(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def create_simulation(self):
        pass

    @staticmethod
    def timeit(method):
        @functools.wraps(method)
        def timed(*args, **kwargs):
            start_time = time.time()
            result = method(*args, **kwargs)
            end_time = time.time()
            print(f"Method {method.__name__} took {end_time - start_time:.4f} seconds")
            return result

        return timed


class OpenMMSimulation(GenericSimulation):
    @GenericSimulation.timeit
    def create_simulation(
        self, pdb: app.PDBFile, system_config: dict, state: str, seed: int = None, test: bool = False, cur_forcefield = None
    ) -> Tuple[app.Simulation, SimulationConfig]:
        """Recreates a simulation object based on the provided parameters.

        This method takes in a seed, state, and checkpoint file path to recreate a simulation object.
        Args:
            seed (str): The seed for the random number generator.
            state (str): The state of the simulation.
            cpt_file (str): The path to the checkpoint file.

        Returns:
            app.Simulation: The recreated simulation object.
            system_config: The potentially altered system configuration in SimulationConfig format.
        """
        start_time = time.time()
        if cur_forcefield is not None:
            forcefield = cur_forcefield
        else:
            forcefield = app.ForceField(system_config["ff"], system_config["water"])
        print(f"Creating ff took {time.time() - start_time:.4f} seconds")

        modeller = app.Modeller(pdb.topology, pdb.positions)

        start_time = time.time()
        modeller.deleteWater()
        print(
            f"Deleting water took {time.time() - start_time:.4f} seconds"
        )
        print('New system has', modeller.topology.getNumChains(), 'chains')
        print('New system has', modeller.topology.getNumResidues(), 'residues')
        print('New system has', modeller.topology.getNumAtoms(), 'atoms')
        # modeller.addExtraParticles(forcefield)

        start_time = time.time()
        # if test:
        #     modeller.addHydrogens(forcefield)
        #     print(
        #         f"Adding hydrogens took {time.time() - start_time:.4f} seconds"
        #     )
        modeller.addHydrogens(forcefield)
        print(
            f"Adding hydrogens took {time.time() - start_time:.4f} seconds"
        )
        start_time = time.time()
        # modeller.addSolvent(
        #     forcefield,
        #     padding=1.0 * unit.nanometer
        # )
        print(
            f"Adding solvent took {time.time() - start_time:.4f} seconds"
        )
        # Create the system
        start_time = time.time()
        # The assumption here is that the system_config cutoff MUST be given in nanometers
        threshold = (
            pdb.topology.getUnitCellDimensions().min().value_in_unit(mm.unit.nanometers)
        ) / 2
        if system_config["cutoff"] > threshold:
            nonbondedCutoff = threshold * mm.unit.nanometers
            # set the attribute in the config for the pipeline.
            system_config["cutoff"] = threshold
            print(
                f"Nonbonded cutoff is greater than half the minimum box dimension. Setting nonbonded cutoff to {threshold} nm"
            )
        else:
            nonbondedCutoff = system_config["cutoff"] * mm.unit.nanometers
        system = forcefield.createSystem(
            modeller.topology,
            nonbondedMethod=mm.app.NoCutoff,
            nonbondedCutoff=nonbondedCutoff,
            constraints=system_config["constraints"],
        )
        print(
            f"Creating system took {time.time() - start_time:.4f} seconds"
        )
        # Integrator settings
        integrator = mm.LangevinIntegrator(
            system_config["temperature"],
            system_config["friction"] / unit.picosecond,
            system_config["time_step_size"] * unit.picoseconds,
        )
        seed = seed if system_config["seed"] is None else system_config["seed"]
        integrator.setRandomNumberSeed(seed)

        # Periodic boundary conditions
        # pdb.topology.setPeriodicBoxVectors(system.getDefaultPeriodicBoxVectors())

        # if state != "nvt":
        #     system.addForce(
        #         mm.MonteCarloBarostat(
        #             system_config["pressure"] * unit.bar,
        #             system_config["temperature"] * unit.kelvin,
        #         )
        #     )

        platform = mm.Platform.getPlatformByName("CUDA")

        # Reference for DisablePmeStream: https://github.com/openmm/openmm/issues/3589
        properties = {
            "DeterministicForces": "true",
            "Precision": "double",
            'CudaPrecision': 'single',
            "DisablePmeStream": "true",
        }

        ## For 4x GPU
        avail_gpu = check_available_gpus()
        properties["DeviceIndex"] = f"{avail_gpu}"
        print(f"USE CURRENT AVAILABLE GPU: {avail_gpu}")
        start_time = time.time()
        simulation = mm.app.Simulation(
            modeller.topology, system, integrator, platform, properties
        )
        print(
            f"Creating simulation took {time.time() - start_time:.4f} seconds"
        )

        # Set initial positions
        start_time = time.time()
        simulation.context.setPositions(modeller.positions)

        print(
            f"Setting positions took {time.time() - start_time:.4f} seconds"
        )

        # Converting the system config into a Dict[str,str] and ensure all values in system_config are of the correct type
        for k, v in system_config.items():
            if not isinstance(v, (str, int, float, dict)):
                system_config[k] = str(v)

        return simulation, integrator, SimulationConfig(**system_config)
