import pandas as pd
import numpy as np
from concurrent.futures import ProcessPoolExecutor
import datetime


def apply_in_parallel(df, func, n_jobs):
    tasks = np.array_split(df, n_jobs, axis=0)  # split df along row axis
    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        result = executor.map(func, tasks)

    return pd.concat(result)


class Timer:
    def __init__(self, name: str):
        self.name = name
        self.start = None

    def __enter__(self):
        self.start = datetime.datetime.now()
        return self

    def __exit__(self, *args):
        print(
            f"{self.name} time elapsed: {(datetime.datetime.now() - self.start).seconds} seconds..."
        )
