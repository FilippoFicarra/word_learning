import logging
from multiprocessing import Pool, cpu_count
from typing import Callable, Dict, Generator, List, Union

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from tqdm import tqdm

from src.interfaces import DataHandlingArguments


class AoAExtractor:
    def __init__(
        self,
        thresholds: List[float],
        result_df: pd.DataFrame,
        fit_func: Callable,
        func: Callable | Dict[bool, Callable],
        extract_func: Callable,
        n_cpus: int = cpu_count(),
    ):
        self.thresholds = thresholds
        self.result_df = result_df
        self.fit_func = fit_func
        self.func = func
        self.extract_func = extract_func

        self.num_processes = n_cpus

        self.max_steps = {
            "gpt2": {
                "childes": {
                    42: 2800,
                    123: 2800,
                    28053: 2600,
                },
                "babylm": {
                    42: 4800,
                    123: 7200,
                    28053: 6000,
                },
                "unified": {
                    42: 40000,
                    123: 40000,
                    28053: 40000,
                },
            },
            "children": 30,
        }

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        console_handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

    def moving_average(self, y: np.ndarray, window: int) -> Generator:
        y = np.pad(y, (window // 2, window // 2 + 1), mode="edge")
        for i in range(window, len(y)):
            yield np.mean(y[i - window : i])

    def aoa_summary_wrapper(self, args):
        # use the logger in child processes
        # ---------------------------- #
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        console_handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        # ---------------------------- #
        return self.extract_aoa(*args)

    def extract_aoa(
        self,
        x: Union[np.ndarray, List[float]],
        y: Union[np.ndarray, List[float]],
        increasing: bool,
        func: Callable,
        info: Dict[str, int],
        data_handling_args: DataHandlingArguments,
    ) -> pd.DataFrame:
        aoa_dicts = []

        if not isinstance(x, np.ndarray):
            x = np.array(x)
        if not isinstance(y, np.ndarray):
            y = np.array(y)

        if data_handling_args.log_x:
            x = np.log10(x + 1e-15)
        if data_handling_args.log_y:
            y = np.log10(y + 1e-15)
        if data_handling_args.scale_x:
            x = (x - x.min()) / (x.max() - x.min())
        if data_handling_args.scale_y:
            y = (y - y.min()) / (y.max() - y.min())

        try:
            popt = None
            x_new = x
            y_new = y

            if data_handling_args.smooth:
                # y_new = savgol_filter(y, 21, 3)
                y_new = list(self.moving_average(y, 7))

            if data_handling_args.fit:
                x_initial, popt = self.fit_func(x, y, increasing)
                x_new = x[x_initial:]
                y_new = func(x_new, *popt)

                # compute rsquared betwen the fitted curve and the data
                r_squared = r2_score(y[x_initial:], y_new)

                if r_squared < 0.15:
                    raise ValueError(f"R squared is too low: {r_squared}")

                if abs(y_new[0] - y_new[-1]) < 1e-5 * (max(y_new) - min(y_new)):
                    self.logger.error(f"No convergence while fitting, {info}, {popt}")

            for threshold in self.thresholds:
                aoa_x, aoa_y = self.extract_func(x_new, y_new, threshold, func, increasing, popt)
                if aoa_x is not None and aoa_y is not None:
                    aoa_dicts.append(info | {"threshold": threshold, "aoa_y": aoa_y, "aoa_x": aoa_x})
        except Exception as e:
            self.logger.error(f"Error while extracting AoA: {e}, for info: {info}")
            # self.logger.error(traceback.format_exc())

        return pd.DataFrame(aoa_dicts)

    def get_aoa(
        self,
        data_handling_args: DataHandlingArguments,
    ) -> pd.DataFrame:
        self.logger.info("Extracting AoA")

        summary_df = self.result_df
        with Pool(processes=self.num_processes) as pool:
            tasks = []
            for model in summary_df["model"].unique():
                for dataset in summary_df["dataset"].unique():
                    for task in summary_df["task"].unique():
                        for seed in summary_df["seed"].unique():
                            summary_df_word = summary_df[
                                (summary_df["model"] == model)
                                & (summary_df["dataset"] == dataset)
                                & (summary_df["task"] == task)
                                & (summary_df["seed"] == seed)
                            ]

                            increasing = None

                            if "extrinsic" in task:
                                if "positive" in task or "negative" in task or "combined" in task:
                                    increasing = False
                                else:
                                    increasing = None
                            if "intrinsic" in task:
                                if "positive" in task:
                                    increasing = False
                                if "negative" in task:
                                    increasing = True

                            func = None
                            if increasing is not None and self.func is not None:
                                func = self.func[increasing]

                            for word in summary_df_word["word"].unique():
                                if word == "i":
                                    continue
                                word_df = summary_df_word[summary_df_word["word"] == word]
                                mean_val = word_df["mean_value"].values
                                steps = [int(step) for step in word_df["step"].values]

                                indices = np.argsort(steps)
                                word_step = [
                                    steps[i] for i in indices if steps[i] <= self.max_steps[model][dataset][int(seed)]
                                ]
                                word_val = [mean_val[i] for i in indices][: len(word_step)]

                                # min max scale the steps, surprisal values
                                word_step = np.array(word_step)
                                word_val = np.array(word_val)

                                infos = {
                                    "model": model,
                                    "dataset": dataset,
                                    "task": task,
                                    "seed": seed,
                                    "word": word,
                                }
                                tasks.append((word_step, word_val, increasing, func, infos, data_handling_args))

            self.logger.info(f"Number of tasks: {len(tasks)}")
            aoa_df = pd.DataFrame()

            with tqdm(total=len(tasks), desc="Processing tasks") as pbar:
                for result_df in pool.map(self.aoa_summary_wrapper, tasks):
                    aoa_df = pd.concat([aoa_df, result_df], ignore_index=True)
                    pbar.update(1)
            return aoa_df
