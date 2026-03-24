"""NNLOJET execution on the local machine

implementation of the backend for ExecutionPolicy.LOCAL
"""

import os
import subprocess
import time
from pathlib import Path

from ...db._loglevel import LogLevel
from .._executor import Executor


class BatchLocalExec(Executor):
    """Execute multiple NNLOJET seeds concurrently on the local machine.

    Spawns all seeds as independent subprocesses in parallel and polls for
    completion, analogous to HTCondorExec and SlurmExec.  The entire batch
    is managed by a *single* Luigi task, so only one worker thread is held
    while the subprocesses run.  This avoids the previous design, where each
    seed was a separate SingleLocalExec Luigi task that blocked its own worker
    thread for the full job duration — which prevented true parallelism when
    the number of concurrent seeds approached the Luigi worker count, and left
    cores idle even when jobs_max_concurrent slots were available.

    Configuration via ``policy_settings``:

    local_ncores : int, optional
        Number of OMP threads passed to each NNLOJET process (default: 1).
    local_poll_time : float, optional
        Polling interval in seconds while waiting for subprocesses to
        finish (default: 30.0).
    """

    @property
    def resources(self):
        return {"jobs_concurrent": self.njobs}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.njobs: int = len(self.exe_data["jobs"])
        self.ncores: int = self.exe_data["policy_settings"].get("local_ncores", 1)
        self.poll_time: float = self.exe_data["policy_settings"].get("local_poll_time", 30.0)

    def exe(self):
        job_env = os.environ.copy()
        job_env["OMP_NUM_THREADS"] = str(self.ncores)
        job_env["OMP_STACKSIZE"] = "1024M"

        procs: dict[int, subprocess.Popen] = {}
        out_handles: dict[int, object] = {}
        err_handles: dict[int, object] = {}

        # Launch all seeds in parallel
        for job_id, job_data in self.exe_data["jobs"].items():
            seed: int = job_data["seed"]
            file_out: Path = self.exe_data.path / f"job.s{seed}.out"
            file_err: Path = self.exe_data.path / f"job.s{seed}.err"
            out_handles[job_id] = open(file_out, "w")  # noqa: SIM115
            err_handles[job_id] = open(file_err, "w")  # noqa: SIM115
            procs[job_id] = subprocess.Popen(
                [
                    self.exe_data["exe"],
                    "-run",
                    self.exe_data["input_files"][0],
                    "-iseed",
                    str(seed),
                ],
                env=job_env,
                cwd=self.exe_data.path,
                stdout=out_handles[job_id],
                stderr=err_handles[job_id],
                text=True,
            )
            self._logger(
                f"BatchLocalExec::exe: launched seed {seed} (pid={procs[job_id].pid})",
                LogLevel.DEBUG,
            )

        # Poll until all subprocesses have finished
        while any(proc.poll() is None for proc in procs.values()):
            time.sleep(self.poll_time)

        # Close file handles and report any failures
        for job_id, proc in procs.items():
            out_handles[job_id].close()
            err_handles[job_id].close()
            if proc.returncode != 0:
                seed = self.exe_data["jobs"][job_id]["seed"]
                self._logger(
                    f"BatchLocalExec::exe: seed {seed} failed"
                    f" (pid={proc.pid}, returncode={proc.returncode})",
                    LogLevel.ERROR,
                )
