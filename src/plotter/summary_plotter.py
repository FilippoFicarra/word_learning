import logging
import os
from typing import List

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from tqdm import tqdm

from src.interfaces import PlotArguments


class SummaryPlotter:
    def __init__(
        self,
        summary_df: pd.DataFrame,
        save_dir: str,
    ):
        self.summary_df = summary_df
        self.save_dir = save_dir
        self.hue_order = ["nouns", "predicates", "function_words"]
        self.task_map = {
            "corpus_positive": r"$\widehat{\sigma}_{+}$",
            "corpus_negative": r"$\widehat{\sigma}_{-}$",
            "corpus_combined": r"$\widehat{\sigma}_{\pm}$",
            "intrinsic_positive": r"$\widehat{\sigma}_{I+}$",
            "intrinsic_negative": r"$\widehat{\sigma}_{I-}$",
            "intrinsic_combined": r"$\widehat{\sigma}_{I\pm}$",
            "extrinsic_positive": r"$\widehat{\sigma}_{\mathrm{E}+}$",
            "extrinsic_negative": r"$\widehat{\sigma}_{\mathrm{E}-}$",
            "extrinsic_combined": r"$\widehat{\sigma}_{\mathrm{E} \pm}$",
            "children": "Children",
        }

        self.tasks = list(self.task_map.keys())
        self.col_order = [self.task_map[t] for t in self.tasks]
        self.row_order = ["log_frequency", "mlu", "n_chars", "concreteness"]

        palette = "tab10"
        original_palette = sns.color_palette(palette)
        palette_list = list(original_palette)
        color_to_remove = palette_list[3]
        new_palette = [color for color in palette_list if color != color_to_remove]
        new_palette = sns.color_palette(new_palette)
        self.palette = new_palette

        self.palette_name = "tab10"

        self.line_color = "r"

        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        sns.set_theme(style="darkgrid")

        # ---------------------------- #
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        console_handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        # ---------------------------- #

    def plot_specific_words(self, plt_args: PlotArguments, word_df: pd.DataFrame, words: List[str]):
        self.logger.info("Plotting specific words")
        plt.rcParams["font.family"] = "serif"

        df = word_df.copy()

        df = df[df["task"].isin(self.tasks)]
        df["task"] = df["task"].apply(lambda x: self.task_map[x])

        tab10_colors = plt.get_cmap(self.palette_name).colors
        palette = {word: tab10_colors[i // 2] for i, word in enumerate(words)}

        g = sns.relplot(
            data=df,
            x="step",
            y="value",
            hue="word",
            col="task",
            kind="line",
            style="style",
            palette=palette,
            facet_kws=dict(sharey=False, sharex=False),
            linewidth=2.5,
            col_order=self.col_order,
            col_wrap=5,
            aspect=plt_args.fig_size[0] / plt_args.fig_size[1],
        )

        g.figure.subplots_adjust(hspace=0.4)

        # mke xlabel and ylabel bigger
        for i, ax in enumerate(g.axes.flatten()):
            ax.set_xlabel("Log Step (Normalized)", fontsize=25)
            # ax.set_ylabel(ax.get_ylabel().capitalize(), fontsize=35)
            ax.set_ylabel("")
            ax.tick_params(axis="both", which="major", labelsize=20)
            ax.yaxis.set_major_locator(plt.MaxNLocator(4))
            ax.xaxis.set_major_locator(plt.MaxNLocator(5))
            if "Children" not in ax.get_title():
                ax.set_title(ax.get_title().replace("task = ", ""), fontsize=45, fontweight="bold", pad=10)
            else:
                ax.set_title(ax.get_title().replace("task = ", ""), fontsize=35, fontweight="bold", pad=10)
                ax.set_xlabel("Months", fontsize=25)

        g._legend.remove()

        g.figure.legend(
            [
                plt.Line2D(
                    [0],
                    [0],
                    color=tab10_colors[i // 2],
                    lw=2,
                    linestyle="-" if i % 2 == 0 else "--",
                )
                for i, _ in enumerate(words)
            ],
            [w_c for _, w_c in enumerate(words)],
            fontsize=25,
            bbox_to_anchor=(0.85, -0.02),
            ncols=len(words),
        )

        if not os.path.exists(f"{self.save_dir}/specific"):
            os.makedirs(f"{self.save_dir}/specific")

        g.savefig(f"{self.save_dir}/specific/specific_words.pdf")

    def plot_aoa_vs_predictor(self, plt_args: PlotArguments, log_aoa=False) -> None:
        """
        Plot the AoA vs predictors.

        Args:
            utils_folder (str): Path to the utils folder.
            threshold (float): Threshold for the AoA.
        """

        plt.rcParams["font.family"] = "serif"

        y_map = {
            "aoa": "AoA",
            "log_aoa": "Log AoA",
        }

        task_dict = {
            "intrinsic": ["intrinsic_positive", "intrinsic_negative", "intrinsic_combined", "children"],
            "corpus": ["corpus_positive", "corpus_negative", "corpus_combined", "children"],
            "extrinsic": ["extrinsic_positive", "extrinsic_negative", "extrinsic_combined", "children"],
        }

        self.logger.info("Plotting AoA vs predictors")

        predictors = ["log_frequency", "mlu", "n_chars", "concreteness"]

        aoa_and_predictors_df = self.summary_df

        aoa_and_predictors_df = aoa_and_predictors_df[aoa_and_predictors_df["threshold"] == 0.07]

        aoa_and_predictors_df = aoa_and_predictors_df.drop(columns=["threshold"])
        if log_aoa:
            y = "log_aoa"
            aoa_and_predictors_df["log_aoa_x"] = np.log10(aoa_and_predictors_df["aoa_x"] + 1e-15)
        else:
            y = "aoa"

        aoa_and_predictors_df = aoa_and_predictors_df.rename(columns={f"{y}_x": y})
        aoa_and_predictors_df = aoa_and_predictors_df[aoa_and_predictors_df["lexical_class"] != "other"]
        if "log_frequency" not in aoa_and_predictors_df.columns:
            aoa_and_predictors_df["log_frequency"] = np.log10(aoa_and_predictors_df["frequency"])

        child_df = aoa_and_predictors_df[
            [
                f"child_{y}",
                "word",
                "lexical_class",
                "child_mlu",
                "concreteness",
                "n_chars",
                "child_log_frequency",
                "seed",
                "model",
                "task",
                "dataset",
            ]
        ]
        child_df = child_df.rename(columns={c: c.replace("child_", "") for c in child_df.columns})
        child_df["task"] = "children"
        child_df["aoa"] = child_df["aoa"].apply(lambda x: (x - 16) / (30 - 16))
        child_df = child_df.drop_duplicates()
        aoa_and_predictors_df = pd.concat([aoa_and_predictors_df, child_df])

        for name, ts in task_dict.items():
            df_ = aoa_and_predictors_df[aoa_and_predictors_df["task"].isin(ts)]

            print(f"task: {name}, {df_["task"].unique()}")
            self.col_order = ts

            with tqdm(
                total=len(aoa_and_predictors_df["dataset"].unique()) * len(aoa_and_predictors_df["model"].unique()),
                desc="Processing predictors",
            ) as pbar:
                # for predictor in predictors:
                melted_df = df_[["word", "lexical_class", y, "seed", "model", "task", "dataset"] + predictors]

                melted_df = pd.melt(
                    melted_df,
                    id_vars=["word", "lexical_class", y, "seed", "model", "task", "dataset"],
                    value_vars=predictors,
                    var_name="predictor",
                    value_name="value",
                )

                for model in melted_df["model"].unique():
                    for dataset in melted_df["dataset"].unique():
                        if not os.path.exists(os.path.join(self.save_dir, "predictors", model)):
                            os.makedirs(os.path.join(self.save_dir, "predictors", model))
                        predictors_df = melted_df[(melted_df["model"] == model) & (melted_df["dataset"] == dataset)]

                        # remove the lines that have the words that have number of different seed different from max_number_of_seeds
                        predictors_df = predictors_df.groupby(
                            ["word", "lexical_class", "task", "predictor", "model"]
                        ).filter(lambda x: len(x) == len(predictors_df["seed"].unique()))

                        predictors_df = predictors_df.drop_duplicates()
                        predictors_df = (
                            predictors_df.groupby(
                                [
                                    "word",
                                    "lexical_class",
                                    "task",
                                    "predictor",
                                    "model",
                                ]
                            )
                            .mean()
                            .reset_index()
                        )
                        predictors_df = predictors_df.drop(columns=["seed"])

                        df = pd.DataFrame()

                        for task in predictors_df["task"].unique():
                            task_df = predictors_df[predictors_df["task"] == task]
                            mean_y = task_df[predictors_df["task"] == task][y].mean()
                            std_y = task_df[predictors_df["task"] == task][y].std()
                            words_to_remove_for_task = task_df[
                                (task_df[y] > mean_y + 3 * std_y) | (task_df[y] < mean_y - 3 * std_y)
                            ]["word"].unique()
                            df = pd.concat([df, task_df[~task_df["word"].isin(words_to_remove_for_task)]])
                        predictors_df = df

                        if predictors_df.empty:
                            pbar.update(1)

                        g = sns.lmplot(
                            data=predictors_df,
                            x="value",
                            y=y,
                            col="task",
                            row="predictor",
                            palette=self.palette,
                            facet_kws={"sharey": False, "sharex": False},
                            hue_order=self.hue_order,
                            hue="lexical_class",
                            scatter_kws={"alpha": 0.5, "s": 30, "rasterized": True},
                            fit_reg=False,
                            aspect=plt_args.fig_size[0] / plt_args.fig_size[1],
                            row_order=self.row_order,
                            col_order=self.col_order,
                        )

                        g.figure.subplots_adjust(hspace=0.25, wspace=0.25)

                        for ax in g.axes.flatten():
                            x_ = []
                            y_ = []
                            for collection in ax.collections:
                                offsets = collection.get_offsets()
                                x_.extend(offsets[:, 0])
                                y_.extend(offsets[:, 1])

                            sns.regplot(
                                x=x_,
                                y=y_,
                                ax=ax,
                                scatter=False,
                                ci=95,
                                order=1,
                                line_kws={"color": "r", "lw": 5, "alpha": 0.7},
                            )

                            title = ax.get_title().split(" | ")
                            task = self.task_map[title[1].split(" = ")[1]]

                            title = (
                                "".join(ax.get_title().split(" | ")[1])
                                .replace("task = ", "")
                                .replace("_", " ")
                                .capitalize()
                            )
                            if title == "Children":
                                fontsize = 60
                            else:
                                fontsize = 65

                            ax.set_title(
                                task,
                                fontsize=fontsize,
                                pad=20,
                                fontweight="bold",
                            )

                            ax.set_ylabel(y_map[y], fontsize=50)
                            ax.set_xlabel("")
                            ax.tick_params(axis="both", which="major", labelsize=30)
                            ax.yaxis.set_major_locator(plt.MaxNLocator(5))
                            ax.xaxis.set_major_locator(plt.MaxNLocator(5))

                        for ax in g.axes[1:].flat:
                            ax.set_title("")

                        for ax in g.axes[:, 1:].flat:
                            ax.set_ylabel("")

                        g._legend.remove()

                        plt.legend(
                            loc="lower center",
                            bbox_to_anchor=(-1.3, -0.6),
                            labels=[h.replace("_", " ").capitalize() for h in self.hue_order],
                            fontsize=40,
                            ncols=len(self.hue_order),
                            labelspacing=0.5,
                            markerscale=2,
                        )

                        plt.subplots_adjust(bottom=0.1, top=0.94, left=0.06, right=0.97)

                        plt.savefig(
                            os.path.join(
                                self.save_dir,
                                "predictors",
                                model,
                                f"{dataset}_{name}.pdf",
                            ),
                            format="pdf",
                        )

                        plt.close()
                        pbar.update(1)
