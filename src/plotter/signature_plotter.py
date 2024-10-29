import logging
import logging.handlers
import os
import warnings
from typing import Dict, Generator, List, Union

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import gridspec
from matplotlib import pyplot as plt
from tqdm import tqdm

from src.interfaces import DataHandlingArguments, PlotArguments

warnings.filterwarnings("ignore")


class SignaturePlotter:
    def __init__(
        self,
        result_dir: str,
        palette: str = "tab10",
    ):
        self.result_dir = result_dir

        # palette
        self.cmap = sns.color_palette(palette, as_cmap=True)

        original_palette = sns.color_palette(palette)
        palette_list = list(original_palette)
        color_to_remove = palette_list[3]
        new_palette = [color for color in palette_list if color != color_to_remove]
        new_palette = sns.color_palette(new_palette)
        self.palette = new_palette
        self.palette_name = palette

        sns.set_theme(style="darkgrid")

        self.lexical_class_colors = {
            "nouns": "blue",
            "predicates": "red",
            "function_words": "black",
        }
        self.hue_order = ["nouns", "predicates", "function_words"]

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

        self.lm_signatures = pd.read_parquet(os.path.join(self.result_dir, "lm_signatures.parquet"))
        self.lm_aoa = pd.read_parquet(os.path.join(self.result_dir, "signatures_aoa.parquet"))
        self.aoa_and_predictors_df = pd.read_csv(os.path.join(self.result_dir, "aoa_and_predictors.csv"))

        # ---------------------------- #
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        console_handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        # ---------------------------- #

    def moving_average(self, y: np.ndarray, window: int) -> Generator:
        y = np.pad(y, (window // 2, window // 2 + 1), mode="edge")
        for i in range(window, len(y)):
            yield np.mean(y[i - window : i])

    def plot(
        self,
        x: Union[np.ndarray, List[float]],
        y: Union[np.ndarray, List[float]],
        std: Union[np.ndarray, List[float]],
        word: str,
        aoas: List[Dict[str, int]],
        output_dir: str,
        thresholds: List[float],
        plt_args: PlotArguments,
        data_handling_args: DataHandlingArguments,
    ) -> None:
        fig = plt.figure(figsize=plt_args.fig_size, dpi=plt_args.dpi)
        gs = gridspec.GridSpec(1, 2)

        ax1 = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs[1])

        if not isinstance(x, np.ndarray):
            x = np.array(x)
        if not isinstance(y, np.ndarray):
            y = np.array(y)
        if not isinstance(std, np.ndarray):
            std = np.array(std)

        if data_handling_args.log_x:
            x = np.log10(x + 1e-15)
        if data_handling_args.log_y:
            y = np.log10(y + 1e-15)
            std = np.log10(std + 1e-15)
        if data_handling_args.scale_x:
            x = (x - x.min()) / (x.max() - x.min())
        if data_handling_args.scale_y:
            den = y.max() - y.min()
            y = (y - y.min()) / den
            std = (std) / den

        data_raw = pd.DataFrame({"Step": x, "Surprisal": y, "std": std})
        sns.lineplot(x="Step", y="Surprisal", data=data_raw, color="black", zorder=1, ax=ax1)

        ax1.fill_between(x, y - 2 * std, y + 2 * std, color="gray", alpha=0.5, zorder=2)

        y_moving_avg = list(self.moving_average(y, 7))
        sns.lineplot(x=x, y=y_moving_avg, color="red", zorder=3, ax=ax1)

        colors = plt.cm.tab10.colors[1:-2]
        for i, aoa in enumerate(aoas):
            if aoa["aoa_x"] < 2 * x[-1] or aoa["aoa_y"] < 2 * y[-1]:
                ax1.scatter(aoa["aoa_x"], aoa["aoa_y"], color=colors[i], s=15, zorder=4, label=thresholds[i])

        handles, labels = ax1.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))

        ax1.legend(by_label.values(), by_label.keys())
        ax1.set_title(f"{word} - Avg Surprisal - ")
        ax1.set_xticks(np.linspace(min(x), max(x), 4))

        sns.lineplot(x="Step", y="95 % CI", data=pd.DataFrame({"Step": x, "95 % CI": 2 * std}), ax=ax2)
        ax2.set_title(f"{word} - 95% CI")
        ax2.set_xticks(np.linspace(min(x), max(x), 4))

        plt.tight_layout()
        plt.savefig(f"{output_dir}/{word}.png")
        plt.close()

    def plot_curves(
        self,
        plt_args: PlotArguments,
        data_handling_args: DataHandlingArguments,
    ) -> None:
        """
        Arguments:
            plt_args: PlotArguments
                Arguments for plotting.
            data_handling_args: DataHandlingArguments
                Arguments for data handling.

        """

        self.logger.info("Plotting surprisal learning curves")

        summary_df = self.lm_signatures

        total = (
            len(summary_df["model"].unique())
            * len(summary_df["dataset"].unique())
            * len(summary_df["task"].unique())
            * len(summary_df["word"].unique())
        )
        with tqdm(total=total, desc="Processing tasks") as pbar:
            for model in summary_df["model"].unique():
                for dataset in summary_df["dataset"].unique():
                    for task in summary_df["task"].unique():
                        for seed in ["42"]:
                            path_to_save = os.path.join(
                                self.result_dir, "assets", "curves", model, dataset, task, str(seed)
                            )

                            if not os.path.exists(path_to_save):
                                os.makedirs(path_to_save)

                            summary_df_word = summary_df[
                                (summary_df["model"] == model)
                                & (summary_df["dataset"] == dataset)
                                & (summary_df["task"] == task)
                                & (summary_df["seed"] == seed)
                            ]

                            for word in summary_df_word["word"].unique():
                                word_df = summary_df_word[summary_df_word["word"] == word]
                                surprisals = word_df["mean_value"].values
                                surprisals_std = word_df["std_value"].values
                                steps = word_df["step"].values

                                order = np.argsort(steps)
                                word_step = steps[order]

                                word_step = np.array(
                                    [s for s in word_step if s <= self.max_steps[model][dataset][int(seed)]]
                                )
                                word_surprisal = np.array(surprisals[order][: len(word_step)])
                                surprisals_std = np.array(surprisals_std[order][: len(word_step)])

                                aoa_df = self.lm_aoa[
                                    (self.lm_aoa["model"] == model)
                                    & (self.lm_aoa["dataset"] == dataset)
                                    & (self.lm_aoa["task"] == task)
                                    & (self.lm_aoa["seed"] == seed)
                                    & (self.lm_aoa["word"] == word)
                                ]

                                thresholds = aoa_df["threshold"].values

                                aoa_x = aoa_df["aoa_x"].values
                                aoa_y = aoa_df["aoa_y"].values

                                aoas = [{"aoa_x": x, "aoa_y": y} for x, y in zip(aoa_x, aoa_y)]

                                self.plot(
                                    word_step,
                                    word_surprisal,
                                    surprisals_std,
                                    word,
                                    aoas,
                                    path_to_save,
                                    thresholds,
                                    plt_args,
                                    data_handling_args,
                                )

                                pbar.update(1)
