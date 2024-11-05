import os
import random
import string

from typing import List
from queue import Queue

import numpy as np
import pandas as pd
from dataclasses import dataclass, asdict

DB_DIR = os.path.join(os.path.dirname(__file__), "db")


class PandasJobStore:
    """Basic csv-based job store using pandas."""

    columns = {
        # "pdb": "str",
        "active": "bool",
        "hotkeys": "object",
        "created_at": "datetime64[s]",
        "updated_at": "datetime64[s]",
        "best_loss": "float64",
        "best_loss_at": "datetime64[s]",
        "best_hotkey": "str",
        "commit_hash": "str",
        "gro_hash": "str",
        "update_interval": "timedelta64[s]",
        "updated_count": "int",
        "max_time_no_improvement": "timedelta64[s]",
        "event": "object",
    }

    def __init__(self, db_path=DB_DIR, table_name="protein_jobs", force_create=False):
        self.db_path = db_path
        self.table_name = table_name
        self.file_path = os.path.join(self.db_path, f"{self.table_name}.csv")

        self._db = self.load_table(force_create=force_create)

    def __repr__(self):
        """Just shows the underlying DataFrame."""
        return f"{self.__class__.__name__}\n{self._db.__repr__()}"

    def write(self):
        """The write method writes the current state of the database to a csv file."""
        self._db.to_csv(self.file_path, index=True, index_label="pdb")

    def load_table(self, force_create=False) -> pd.DataFrame:
        """Creates a table in the database to store jobs."""
        # Use context manager to handle the database connection
        if not os.path.exists(self.file_path) or force_create:
            os.makedirs(self.db_path, exist_ok=True)

            self._db = pd.DataFrame(
                columns=self.columns.keys(), index=pd.Index([], name="pdb")
            )
            self.write()

        df = pd.read_csv(self.file_path).astype(self.columns).set_index("pdb")
        df["hotkeys"] = df["hotkeys"].apply(eval)
        return df

    def get_queue(self, ready=True) -> Queue:
        """Checks DB for all jobs with active status and returns them as a DataFrame.

        Args:
            ready (bool, optional): Return rows where rows of the db have not been updated longer than the update_interval. Defaults to True.

        Returns:
            Queue: queue with jobs
        """
        active = self._db["active"] == True

        if ready:
            pending = (pd.Timestamp.now() - self._db["updated_at"]) >= self._db[
                "update_interval"
            ]

            jobs = self._db.loc[active & pending]
        else:
            jobs = self._db.loc[active]

        queue = Queue()
        for pdb, row in jobs.iterrows():
            queue.put(Job(pdb=pdb, **row.to_dict()))

        return queue

    def insert(
        self,
        pdb: str,
        ff: str,
        box: str,
        water: str,
        hotkeys: List[str],
        epsilon: float,
        system_kwargs: dict,
        **kwargs,
    ):
        """Adds a new job to the database."""

        if pdb in self._db.index.tolist():
            raise ValueError(f"pdb {pdb!r} is already in the store")

        job = Job(
            pdb=pdb,
            ff=ff,
            box=box,
            water=water,
            hotkeys=hotkeys,
            created_at=pd.Timestamp.now().floor("s"),
            updated_at=pd.Timestamp.now().floor("s"),
            epsilon=epsilon,
            system_kwargs=system_kwargs,
            **kwargs,
        ).to_frame()

        if len(self._db) == 0:
            self._db = job  # .astype(self.columns)
        else:
            self._db = pd.concat([self._db, job], ignore_index=False, axis=0)

        self._db.index.name = "pdb"
        self._db = self._db.astype(self.columns)

        self.write()

    def update(self, job):
        """Updates the status of a job in the database."""

        job_to_update = job.to_frame()

        self._db.update(job_to_update)
        self.write()


@dataclass
class Job:
    # TODO: inherit from pydantic BaseModel which should take care of dtypes and mutation

    pdb: str
    ff: str
    box: str
    water: str
    hotkeys: list
    created_at: pd.Timestamp
    updated_at: pd.Timestamp
    active: bool = True
    best_loss: float = np.inf
    best_loss_at: pd.Timestamp = pd.NaT
    best_hotkey: str = None
    commit_hash: str = None
    gro_hash: str = None
    update_interval: pd.Timedelta = pd.Timedelta('10 minutes')
    updated_count: int = 0
    min_updates: int = 5
    max_time_no_improvement: pd.Timedelta = pd.Timedelta('25 minutes')
    epsilon: float = 0.05  # percentage.
    event: dict = None
    system_kwargs: dict = None

    def to_dict(self):
        return asdict(self)

    def to_series(self):
        data = asdict(self)
        name = data.pop("pdb")
        return pd.Series(data, name=name)

    def to_frame(self):
        return pd.DataFrame([self.to_series()])

    def update(
        self,
        loss: float,
        hotkey: str,
        hotkeys: List[str] = None,
    ):
        """Updates the status of a job in the database. If the loss improves, the best loss, hotkey and hashes are updated."""

        if hotkeys is not None:
            assert len(hotkeys) > 0, "hotkeys must be a non-empty list"
            self.hotkeys = hotkeys

        if hotkey not in self.hotkeys:
            raise ValueError(f"Hotkey {hotkey!r} is not a valid choice")

        percent_improvement = (
            (loss - self.best_loss) / self.best_loss
            if not np.isinf(self.best_loss) and not self.best_loss == 0
            else np.nan
        )
        self.updated_at = pd.Timestamp.now().floor("s")
        self.updated_count += 1

        never_updated_better_loss = (
            np.isnan(percent_improvement) and loss < self.best_loss
        )  # only happens if best_loss is 0 or inf
        better_loss = (
            percent_improvement >= self.epsilon
        )  # only happens if best_loss is not 0 or inf

        if never_updated_better_loss or better_loss:
            self.best_loss = loss
            self.best_loss_at = pd.Timestamp.now().floor("s")
            self.best_hotkey = hotkey
        elif (  # if loss has been improved but not recently enough, trigger early stopping
            pd.Timestamp.now().floor("s") - self.best_loss_at
            > self.max_time_no_improvement
            and self.updated_count >= self.min_updates
        ):
            self.active = False
        elif (  # if loss has never been improved and the job has been running a long time, trigger early stopping
            isinstance(
                self.best_loss_at, pd._libs.tslibs.nattype.NaTType
            )  # a best loss has not been assigned
            and pd.Timestamp.now().floor("s") - self.created_at
            > self.max_time_no_improvement  # the time since the last best loss is greater than the max allowed time
        ):
            self.active = False

    def check_for_available_hotkeys(self, hotkeys: List[str]) -> bool:
        """Checks the job's hotkeys to only include those that are still valid. This permanently removes hotkeys from a job. If no hotkeys are left, the job is set to inactive.

        Returns:
            bool : True if there are remaining hotkeys in the job, and False if there are none.
        """

        # Get the list of hotkeys that are still valid and set the attribute.
        self.hotkeys = list(set(self.hotkeys) & set(hotkeys))

        # If no hotkeys are left, set the job to inactive and return False
        if not self.hotkeys:
            self.active = False
            return False

        return True


class MockJob(Job):
    def __init__(self, n_hotkeys=5, update_seconds=5, stop_after_seconds=10):
        self.pdb = self._make_pdb()
        self.ff = "charmm27"
        self.box = "cube"
        self.water = "tip3p"
        self.hotkeys = self._make_hotkeys(n_hotkeys)
        self.created_at = (
            pd.Timestamp.now().floor("s")
            - pd.Timedelta(seconds=random.randint(0, 3600 * 24))
        ).floor("s")
        self.best_loss = random.random()
        self.best_hotkey = random.choice(self.hotkeys)
        self.commit_hash = self._make_commit_hash()
        self.gro_hash = self._make_commit_hash()
        self.update_interval = pd.Timedelta(seconds=update_seconds)
        self.min_updates = 1
        self.max_time_no_improvement = pd.Timedelta(seconds=stop_after_seconds)

    @staticmethod
    def _make_pdb():
        return "".join(random.choices(string.digits + string.ascii_lowercase, k=4))

    @staticmethod
    def _make_hotkeys(n=10, length=8):
        return [MockJob._make_hotkey(length) for _ in range(n)]

    @staticmethod
    def _make_hotkey(length=8):
        return "".join(random.choices(string.digits + string.ascii_letters, k=length))

    @staticmethod
    def _make_commit_hash(k=40):
        return "".join(random.choices(string.digits + string.ascii_letters, k=k))