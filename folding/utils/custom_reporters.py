import os
import openmm.app as app
import pandas as pd 
import numpy as np
from folding.utils.uids import filter_log_file

class LastTwoCheckpointsReporter(app.CheckpointReporter):
    def __init__(self, file_prefix, reportInterval, amount_first_loop=5, annealing_loop_cpt=83, num_elements=300):
        super().__init__(file_prefix + "_1.cpt", reportInterval)
        self.file_prefix = file_prefix
        self.reportInterval = reportInterval
        self.num_elements = num_elements
        self.min_energy = float('inf')
        self.steps_since_last_checkpoint = 0
        self.last_saved_state = None
        self.count_cpt = 0
        self.amount_first_loop = amount_first_loop
        self.finish_annealing_loop_at_cpt = amount_first_loop + annealing_loop_cpt
        self.after_new_min=False

    def report(self, simulation, state):
        # Create a new checkpoint
        ## Get new energies of next 5k step
        file_path = f"{self.file_prefix}_origin.log"
        log_file = pd.read_csv(file_path)
        max_step = self.steps_since_last_checkpoint + self.reportInterval
        miner_energies = log_file[
            (log_file['#"Step"'] > self.steps_since_last_checkpoint)
            & (log_file['#"Step"'] <= max_step)
        ]["Potential Energy (kJ/mole)"].values
        if self.after_new_min:
            # TODO: add to log file next 2k step to
            filter_log_file(
                file_path,
                f"{self.file_prefix}.log",
                self.steps_since_last_checkpoint,
                self.steps_since_last_checkpoint + 2000,
                mode='a'
            )
            self.after_new_min = False
        if len(miner_energies) >= self.num_elements:
            selected_energies = miner_energies[:self.num_elements]
            if len(selected_energies) >= 10:
                last_10 = selected_energies[-10:]
                new_min_energy = np.median(last_10)
                if new_min_energy < self.min_energy and (self.count_cpt < self.amount_first_loop or self.count_cpt > self.finish_annealing_loop_at_cpt):
                    WINDOW = 50  # Number of steps to calculate the gradient over
                    GRADIENT_THRESHOLD = 10  # kJ/mol/nm
                    ## CHECK GRADIENT
                    mean_gradient = np.diff(selected_energies[:WINDOW]).mean().item()
                    if mean_gradient <= GRADIENT_THRESHOLD:
                        self.min_energy = new_min_energy
                        if self.last_saved_state:
                            current_checkpoint = f"{self.file_prefix}.cpt"
                            # if os.path.exists(current_checkpoint):
                            #     os.rename(current_checkpoint, f"{self.file_prefix}_old.cpt")
                            with open(f"{self.file_prefix}.cpt", 'wb') as f:
                                f.write(self.last_saved_state)
                            filter_log_file(
                                file_path,
                                f"{self.file_prefix}.log",
                                self.steps_since_last_checkpoint + 10,
                                self.steps_since_last_checkpoint + self.reportInterval
                            )
                            self.after_new_min = True
        self.last_saved_state = simulation.context.createCheckpoint()
        self.steps_since_last_checkpoint = self.steps_since_last_checkpoint + self.reportInterval
        self.count_cpt = self.count_cpt + 1

    def describeNextReport(self, simulation):
        steps = self.reportInterval - simulation.currentStep % self.reportInterval
        return (steps, False, False, False, False, False)


class ExitFileReporter(object):
    def __init__(self, filename, reportInterval, file_prefix):
        self.filename = filename
        self.reportInterval = reportInterval
        self.file_prefix = file_prefix

    def describeNextReport(self, simulation):
        steps_left = simulation.currentStep % self.reportInterval
        return (steps_left, False, False, False, False)

    def report(self, simulation, state):
        if os.path.exists(self.filename):
            with open(f"{self.file_prefix}.cpt", "wb") as f:
                f.write(simulation.context.createCheckpoint())
            raise Exception("Simulation stopped")

    def finalize(self):
        pass

class CustomStateReporter(app.StateDataReporter):
    def __init__(self, file, reportInterval, skip_start: int, skip_end: int,  **kwargs):
        super(CustomStateReporter, self).__init__(file, reportInterval, **kwargs)
        self.skip_start = skip_start
        self.skip_end = skip_end

    def report(self, simulation, state):
        # Call the parent report to handle standard logging
        current_step = simulation.currentStep
        if self.skip_start is not None and self.skip_end is not None:
            if self.skip_start <= current_step <= self.skip_end:
                # Skip logging for this step
                return
        super(CustomStateReporter, self).report(simulation, state)