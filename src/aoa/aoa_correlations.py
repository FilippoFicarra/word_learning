import argparse
import itertools
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from src.base_logger import logger

warnings.filterwarnings("ignore")


class CorrelationMatrix:
    def __init__(self, save_dir: str, lm_results: str):
        self.save_dir = save_dir
        self.task_map = {
            "corpus_positive": r"1$\widehat{\sigma}_{+}$ ",
            "corpus_negative": r"2$\widehat{\sigma}_{-}$ ",
            "corpus_combined": r"3$\widehat{\sigma}_{\pm}$ ",
            "intrinsic_positive": r"4$\widehat{\sigma}_{I+}$",
            "intrinsic_negative": r"5$\widehat{\sigma}_{I-}$",
            "intrinsic_combined": r"6$\widehat{\sigma}_{I\pm}$",
            "extrinsic_positive": r"7$\widehat{\sigma}_{\mathrm{E}+}$",
            "extrinsic_negative": r"8$\widehat{\sigma}_{\mathrm{E}-}$",
            "extrinsic_combined": r"9$\widehat{\sigma}_{\mathrm{E}\pm}$",
            "children": "CChildren",
        }
        self.tasks = list(self.task_map.keys())
        self.lm_results = lm_results

    def plot_heatmap(self, correlation_matrix, save_dir, var_name, map_, fig_size, title, args, children_present=False):
        if var_name == "task":
            correlation_matrix = correlation_matrix[self.tasks]

        correlation_matrix = correlation_matrix.reset_index()
        correlation_matrix = correlation_matrix.melt(id_vars="index", var_name=var_name, value_name="correlation")
        correlation_matrix["correlation"] = correlation_matrix["correlation"].apply(lambda x: round(x, 2))

        if map_:
            correlation_matrix["index"] = correlation_matrix["index"].map(self.task_map)
            correlation_matrix[var_name] = correlation_matrix[var_name].map(self.task_map)

        correlation_matrix.fillna(0, inplace=True)

        correlation_matrix = correlation_matrix.pivot(index="index", columns=var_name, values="correlation")
        correlation_matrix = correlation_matrix.apply(pd.to_numeric, errors="coerce")
        plt.rcParams["font.family"] = "serif"

        if children_present:
            _, ax = plt.subplots(figsize=(fig_size))

            correlation_matrix_no_c = correlation_matrix.iloc[:-1, :-1]

            mask = np.triu(np.ones_like(correlation_matrix_no_c, dtype=bool), k=1)
            g = sns.heatmap(
                correlation_matrix_no_c,
                ax=ax,
                vmin=-1,
                vmax=1,
                cmap="coolwarm",
                mask=mask,
                square=False,
                **args,
            )

            g.set_title(title, fontsize=16, weight="bold")

            g.set_xlabel("")
            g.set_ylabel("")
            g.set_yticklabels([x.get_text()[1:] for x in g.get_yticklabels()])
            g.set_xticklabels([x.get_text()[1:] for x in g.get_xticklabels()])
            g.tick_params(axis="both", which="major", labelsize=14)

            g.figure.tight_layout(rect=[0, 0, 1, 1])
            g.figure.savefig(save_dir, format="pdf")

            plt.close()

            col_len = 1.7
            _, ax1 = plt.subplots(figsize=(5, col_len))

            correlation_matrix_only_c = correlation_matrix.iloc[-1:, :-1]
            args["cbar_kws"] = {"orientation": "vertical", "pad": 0.3}
            g1 = sns.heatmap(
                correlation_matrix_only_c,
                ax=ax1,
                vmin=-1,
                vmax=1,
                cmap="coolwarm",
                square=False,
                **args,
            )

            g1.collections[0].colorbar.remove()

            g1.set_title(title, fontsize=16, weight="bold")

            g1.set_xlabel("")
            g1.set_ylabel("")
            g1.set_yticklabels([x.get_text()[1:] for x in g1.get_yticklabels()])
            g1.set_xticklabels([x.get_text()[1:] for x in g1.get_xticklabels()])
            g1.tick_params(axis="both", which="major", labelsize=14)

            g1.figure.tight_layout(rect=[0, 0, 1, 1])
            g1.figure.savefig(save_dir.rstrip(".pdf") + "_only_children.pdf", format="pdf")
            plt.close()

        else:
            _, ax = plt.subplots(figsize=(fig_size))

            mask = np.triu(np.ones_like(correlation_matrix, dtype=bool), k=1)

            g = sns.heatmap(
                correlation_matrix,
                ax=ax,
                vmin=-1,
                vmax=1,
                cmap="coolwarm",
                mask=mask,
                square=False,
                **args,
            )

            g.set_title(title[1:], fontsize=16, weight="bold")

            g.set_xlabel("")
            g.set_ylabel("")
            g.tick_params(axis="both", which="major", labelsize=14)
            g.set_yticklabels([x.get_text()[1:] for x in g.get_yticklabels()])
            g.set_xticklabels([x.get_text()[1:] for x in g.get_xticklabels()])

            plt.tight_layout(rect=[0, 0, 1, 1])
            plt.savefig(save_dir, format="pdf")
            plt.close()

    def correlate_among_tasks(self):
        lm_df = pd.read_csv(f"{self.lm_results}/aoa_and_predictors.csv")
        lm_df = lm_df[lm_df["threshold"] == 0.07]

        save_dir = self.save_dir

        if not os.path.exists(f"{save_dir}/correlation_among_tasks"):
            os.makedirs(f"{save_dir}/correlation_among_tasks")

        logger.info(f"Saving to {save_dir}/correlation_among_tasks")

        df = lm_df.copy()

        df = df[df["lexical_class"] != "other"]

        num_unique_seeds = len(df["seed"].unique())

        task_dfs = []

        for task in df["task"].unique():
            task_df = df[df["task"] == task]
            task_df = task_df.drop(columns=["threshold"])
            task_dfs.append(task_df)

        # children task
        children_df = lm_df.copy()
        children_df = children_df[
            ["word", "child_aoa", "child_mlu", "concreteness", "n_chars", "child_log_frequency", "lexical_class"]
        ]
        children_df["task"] = "children"

        children_df["child_aoa"] = children_df["child_aoa"].apply(lambda x: (x - 16) / (30 - 16))

        task_dfs.append(children_df)

        task_names = df["task"].unique()
        dataset_names = df["dataset"].unique()

        correlation_matrix = {
            d: pd.DataFrame(index=task_names, columns=task_names) for d in task_dfs[0]["dataset"].unique()
        }
        common_rows_matrix = {
            d: pd.DataFrame(index=task_names, columns=task_names) for d in task_dfs[0]["dataset"].unique()
        }

        for dataset in dataset_names:
            for task_df1, task_df2 in list(itertools.combinations(task_dfs, 2)):
                if task_df1["task"].unique()[0] == "children":
                    task_df1 = task_df1.dropna().drop_duplicates()
                    mean_1 = task_df1["child_aoa"].mean()
                    std_1 = task_df1["child_aoa"].std()
                    task_df1_th = task_df1[
                        (task_df1["child_aoa"] >= mean_1 - 3 * std_1) & (task_df1["child_aoa"] <= mean_1 + 3 * std_1)
                    ]
                    words_all_seed_task1 = task_df1_th["word"].unique()
                    task_df1_th["aoa_x"] = task_df1_th["child_aoa"]

                else:
                    task_df1_th = task_df1[(task_df1["dataset"] == dataset) & (task_df1["model"] == "gpt2")]
                    task_df1_th = (
                        task_df1_th[
                            [
                                "word",
                                "aoa_x",
                                "mlu",
                                "concreteness",
                                "n_chars",
                                "log_frequency",
                                "frequency",
                                "lexical_class",
                                "seed",
                                "task",
                            ]
                        ]
                        .dropna()
                        .drop_duplicates()
                    )
                    mean_1 = task_df1_th["aoa_x"].mean()
                    std_1 = task_df1_th["aoa_x"].std()
                    task_df1_th = task_df1_th[
                        (task_df1_th["aoa_x"] >= mean_1 - 3 * std_1) & (task_df1_th["aoa_x"] <= mean_1 + 3 * std_1)
                    ]
                    words_all_seed_task1 = (
                        task_df1_th.groupby("word")
                        .filter(lambda x: x["seed"].nunique() == num_unique_seeds)["word"]
                        .unique()
                    )
                    task_df1_th = (
                        task_df1_th.groupby(["word", "task", "lexical_class"])[
                            ["aoa_x", "mlu", "concreteness", "n_chars", "log_frequency", "frequency"]
                        ]
                        .mean()
                        .reset_index()
                    )

                if task_df2["task"].unique()[0] == "children":
                    task_df2 = task_df2.dropna().drop_duplicates()
                    mean_2 = task_df2["child_aoa"].mean()
                    std_2 = task_df2["child_aoa"].std()
                    task_df2_th = task_df2[
                        (task_df2["child_aoa"] >= mean_2 - 3 * std_2) & (task_df2["child_aoa"] <= mean_2 + 3 * std_2)
                    ]
                    words_all_seed_task2 = task_df2_th["word"].unique()
                    task_df2_th["aoa_x"] = task_df2_th["child_aoa"]

                else:
                    task_df2_th = task_df2[(task_df2["dataset"] == dataset) & (task_df2["model"] == "gpt2")]
                    task_df2_th = (
                        task_df2_th[
                            [
                                "word",
                                "aoa_x",
                                "mlu",
                                "concreteness",
                                "n_chars",
                                "log_frequency",
                                "frequency",
                                "lexical_class",
                                "seed",
                                "task",
                            ]
                        ]
                        .dropna()
                        .drop_duplicates()
                    )
                    mean_2 = task_df2_th["aoa_x"].mean()
                    std_2 = task_df2_th["aoa_x"].std()
                    task_df2_th = task_df2_th[
                        (task_df2_th["aoa_x"] >= mean_2 - 3 * std_2) & (task_df2_th["aoa_x"] <= mean_2 + 3 * std_2)
                    ]
                    words_all_seed_task2 = (
                        task_df2_th.groupby("word")
                        .filter(lambda x: x["seed"].nunique() == num_unique_seeds)["word"]
                        .unique()
                    )
                    task_df2_th = (
                        task_df2_th.groupby(["word", "task", "lexical_class"])[
                            ["aoa_x", "mlu", "concreteness", "n_chars", "log_frequency", "frequency"]
                        ]
                        .mean()
                        .reset_index()
                    )

                common_words = set(words_all_seed_task1).intersection(set(words_all_seed_task2))

                task_df1_th_common = task_df1_th[task_df1_th["word"].isin(common_words)]
                task_df2_th_common = task_df2_th[task_df2_th["word"].isin(common_words)]

                task_df1_th_common = task_df1_th_common.sort_values(by=["word"])
                task_df2_th_common = task_df2_th_common.sort_values(by=["word"])

                merged_df = pd.merge(task_df1_th_common, task_df2_th_common, on=["word", "lexical_class"], how="inner")

                correlation = round(stats.pearsonr(merged_df["aoa_x_x"], merged_df["aoa_x_y"])[0], 2)

                task_name1 = task_df1["task"].unique()[0]
                task_name2 = task_df2["task"].unique()[0]

                correlation_matrix[dataset].loc[task_name1, task_name2] = correlation
                correlation_matrix[dataset].loc[task_name2, task_name1] = correlation
                correlation_matrix[dataset].loc[task_name1, task_name1] = 1
                correlation_matrix[dataset].loc[task_name2, task_name2] = 1
                common_rows_matrix[dataset].loc[task_name1, task_name2] = len(common_words)

            # correlation_matrix[dataset].to_csv(f"{save_dir}/correlation_matrix_{dataset}.csv")
            # common_rows_matrix[dataset].to_csv(f"{save_dir}/common_rows_matrix_{dataset}.csv")

            self.plot_heatmap(
                correlation_matrix[dataset][self.tasks],
                f"{save_dir}/correlation_among_tasks/correlation_matrix_{dataset}.pdf",
                fig_size=(5, 5),
                args={
                    "annot": True,
                    "cbar_kws": {"orientation": "horizontal", "pad": 0.08},
                    "annot_kws": {"weight": "bold"},
                },
                title=f"{dataset.capitalize()}",
                map_=True,
                var_name="task",
                children_present=True,
            )

    def correlate_among_thresholds(self):
        lm_df = pd.read_csv(f"{self.lm_results}/aoa_and_predictors.csv")
        lm_df = lm_df[lm_df["threshold"] != 0.01]
        lm_df = lm_df[lm_df["lexical_class"] != "other"]

        # concatenate the two dataframes
        df = lm_df.copy()
        save_dir = self.save_dir

        if not os.path.exists(os.path.join(save_dir, "correlation_among_thresholds")):
            os.makedirs(os.path.join(save_dir, "correlation_among_thresholds"))

        logger.info(f"Saving to {save_dir}/correlation_among_thresholds")

        correlations_matrix = {}
        for dataset in df["dataset"].unique():
            if dataset not in correlations_matrix:
                correlations_matrix[dataset] = {}
            for task in df["task"].unique():
                if task not in correlations_matrix[dataset]:
                    correlations_matrix[dataset][task] = pd.DataFrame(
                        index=df["threshold"].unique(), columns=df["threshold"].unique()
                    )
                for threshold in df["threshold"].unique():
                    for threshold2 in df["threshold"].unique():
                        if threshold != threshold2:
                            df2 = df[(df["task"] == task) & (df["threshold"] == threshold) & (df["dataset"] == dataset)]
                            df3 = df[
                                (df["task"] == task) & (df["threshold"] == threshold2) & (df["dataset"] == dataset)
                            ]
                            df4 = pd.merge(df2, df3, on=["word", "seed", "lexical_class"], how="inner")
                            corr = df4["aoa_x_x"].corr(df4["aoa_x_y"])
                            correlations_matrix[dataset][task].loc[threshold, threshold2] = corr
                        else:
                            correlations_matrix[dataset][task].loc[threshold, threshold2] = 1.0

                args = {"annot": True, "annot_kws": {"size": 14}, "cbar": False}

                self.plot_heatmap(
                    correlations_matrix[dataset][task],
                    f"{save_dir}/correlation_among_thresholds/correlations_matrix_{dataset}_{task}.pdf",
                    fig_size=(3, 3),
                    args=args,
                    title=f"{self.task_map[task]}",
                    var_name="threshold",
                    map_=False,
                )


def parse_args():
    parser = argparse.ArgumentParser(description="Convert CSV to LaTeX table")

    parser.add_argument(
        "--lm_results",
        type=str,
        default="src/signatures/results",
        help="Folder with CSV files (default: src/signatures/results)",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="src/aoa/results/correlations",
        help="File to save the output (default: tasks/regressions/results/)",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    correlation_matrix = CorrelationMatrix(args.save_dir, lm_results=args.lm_results)
    correlation_matrix.correlate_among_tasks()
    correlation_matrix.correlate_among_thresholds()
