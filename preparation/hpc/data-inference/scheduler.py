import subprocess
import re
from time import sleep
import logging
from datetime import datetime
import pandas as pd
import os

MAX_JOB_ID = 221_919
STEP_SIZE = 3000
BATCH_SIZE = 25
SHELL_JOB_FOLDER = "sjobs/"

SHELL_TEMPLATE = """#!/bin/sh
#SBATCH --partition=gpu_4_a100
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --job-name=finf-{job_id}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=sjobs/{job_id}.out
#SBATCH --mail-user=fb6372@partner.kit.edu

module load devel/cuda/11.8

source $HOME/miniconda3/etc/profile.d/conda.sh

python3 -u full_inference.py \\
 --llama_variant 13B \\
 --model_id full-finetune-100 \\
 --start {job_id} \\
 --n {step_size} \\
 --input_file data/works.csv \\
 --batch_size {batch_size} \\
 --max_new_tokens 512
"""

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filename="scheduler.log",
)


def valid(lst):
    return [x for x in lst if x]


class Slurm:
    def amount_submitted(self):
        return len(self.submitted_jobs())

    def submitted_jobs(self):
        out = Slurm.execute('squeue -o "%j" --noheader')
        return valid(out.split("\n"))

    def submit(self, job_id):
        os.makedirs(SHELL_JOB_FOLDER, exist_ok=True)
        file_name = f"finf-{job_id}.sh"
        full_path = os.path.join(SHELL_JOB_FOLDER, file_name)
        Slurm.create_shell_script(
            full_path, job_id=job_id, batch_size=BATCH_SIZE, step_size=STEP_SIZE
        )
        logging.info(f"sbatch {full_path}")
        Slurm.execute(f"sbatch {full_path}")

    @staticmethod
    def create_shell_script(path, **kwargs):
        logging.info(f"Creating shell script: {path}")
        with open(path, "w") as f:
            f.write(SHELL_TEMPLATE.format(**kwargs))

    @staticmethod
    def execute(cmd):
        logging.debug(f"Executing: {cmd}")

        return subprocess.run(
            cmd, shell=True, capture_output=True, text=True
        ).stdout.strip()


class JobHandler:
    NA = "-"

    def __init__(self, job_file):
        self.job_file = job_file

    def get_next_jobs(self, n=1):
        job_list = self.get_job_list()
        if job_list.empty:  # start with 0
            return list(range(0, STEP_SIZE * n, STEP_SIZE))

        last_job = int(job_list.iloc[-1]["job_id"])
        return list(
            range(last_job + STEP_SIZE, last_job + STEP_SIZE * (n + 1), STEP_SIZE)
        )

    def save_started_jobs(self, job_ids):
        job = pd.DataFrame(
            {
                "job_id": sorted(job_ids),
                "start_time": [JobHandler.now()] * len(job_ids),
                "end_time": [self.NA] * len(job_ids),
            }
        )
        job.to_csv(self.job_file, mode="a", header=False, index=False)

    def get_open_jobs(self):
        jl = self.get_job_list()
        return list(jl[jl.end_time == self.NA].job_id)

    def save_finished_jobs(self, job_ids):
        job_list = self.get_job_list()
        finished = job_list[job_list.job_id.isin(job_ids)]
        finished["end_time"] = JobHandler.now()
        job_list.update(finished)
        job_list.to_csv(self.job_file, index=False)

    def get_job_list(self):
        return pd.read_csv(self.job_file, dtype={"job_id": int})

    @staticmethod
    def now():
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


class Scheduler:
    JOB_ID_REGEX = r"finf-(\d+)"

    def __init__(self, jh: JobHandler, max_jobs=10):
        self.jh = jh
        self.max_jobs = max_jobs
        self.slurm = Slurm()

    def save_finished_jobs(self):
        finished = set(self.jh.get_open_jobs()) - set(
            self.submitted_job_ids()
        )  # still open but not submitted anymore = finished
        logging.info(f"Finished jobs: {sorted(finished)}")
        self.jh.save_finished_jobs(finished)

    def submitted_job_ids(self):
        return map(
            lambda job_name: int(self.get_job_id(job_name)), self.slurm.submitted_jobs()
        )

    def submit_job(self, job_id):
        if self.slurm.amount_submitted() >= self.max_jobs:
            raise RuntimeError("Too many jobs submitted")

        if job_id > MAX_JOB_ID:
            logging.info("Processed all jobs")
            exit(0)

        self.slurm.submit(job_id)
        logging.debug(f"Submitted job {job_id}")

    def submit_jobs(self):
        amount_to_submit = self.max_jobs - self.slurm.amount_submitted()

        if amount_to_submit <= 0:
            logging.info("No jobs to submit")
            return

        logging.info(f"Submitting {amount_to_submit} jobs")
        next_jobs = self.jh.get_next_jobs(amount_to_submit)
        for job_id in next_jobs:
            self.submit_job(job_id)

        currently_submitted = self.submitted_job_ids()
        actually_submitted = set(next_jobs) & set(currently_submitted)  # intersection
        logging.info(f"Actually submitted jobs: {sorted(actually_submitted)}")
        self.jh.save_started_jobs(actually_submitted)

    @staticmethod
    def get_job_id(jobname):
        return re.search(Scheduler.JOB_ID_REGEX, jobname).group(1)


jh = JobHandler("jobs.csv")
sc = Scheduler(jh, max_jobs=2)

SLEEP_AMOUNT = 60 * 60  # 1 hour

if __name__ == "__main__":
    scheduler = Scheduler(jh, max_jobs=10)

    while True:
        logging.info("Checking for finished jobs")
        scheduler.save_finished_jobs()

        logging.info("Submitting new jobs")
        scheduler.submit_jobs()
        logging.info(f"Sleeping for {SLEEP_AMOUNT // 60} minutes")
        sleep(SLEEP_AMOUNT)
