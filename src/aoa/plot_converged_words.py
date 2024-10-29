import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--aoa_and_predictors", type=str, default="src/signatures/results/aoa_and_predictors.csv")
    parser.add_argument("--output_dir", type=str, default="src/aoa/results/converged_words")
    return parser.parse_args()


def main():
    task_map = {
        "corpus_positive": r"$\widehat{\sigma}_{+}$",
        "corpus_negative": r"$\widehat{\sigma}_{-}$",
        "corpus_combined": r"$\widehat{\sigma}_{\pm}$",
        "intrinsic_positive": r"$\widehat{\sigma}_{I+}$",
        "intrinsic_negative": r"$\widehat{\sigma}_{I-}$",
        "intrinsic_combined": r"$\widehat{\sigma}_{I\pm}$",
        "extrinsic_positive": r"$\widehat{\sigma}_{\mathrm{E}+}$",
        "extrinsic_negative": r"$\widehat{\sigma}_{\mathrm{E}-}$",
        "extrinsic_combined": r"$\widehat{\sigma}_{\mathrm{E}\pm}$",
    }

    args = parse_args()
    aoa_and_predictors = args.aoa_and_predictors
    output_dir = args.output_dir

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    df = pd.read_csv(aoa_and_predictors)

    df = df[df["lexical_class"] != "other"]
    df_2 = pd.DataFrame(columns=["task", "num_words", "threshold", "dataset"])

    for dataset in df["dataset"].unique():
        for task in df["task"].unique():
            for threshold in df["threshold"].unique():
                df_dataset = df[(df["dataset"] == dataset) & (df["task"] == task) & (df["threshold"] == threshold)]

                df_dataset = df_dataset.groupby(["word", "lexical_class", "task", "model"]).filter(
                    lambda x: len(x) == len(df_dataset["seed"].unique())
                )

                num_words = len(df_dataset["word"].unique())
                percentage = round(100 - len(df_dataset["word"].unique()) / 2.62, 2)

                df_2 = pd.concat(
                    [
                        df_2,
                        pd.DataFrame(
                            {
                                "task": [task],
                                "num_words": [num_words],
                                "percentage": [percentage],
                                "threshold": [threshold],
                                "dataset": [dataset],
                            }
                        ),
                    ],
                    ignore_index=True,
                )

    df_2["num_words"] = df_2["num_words"].apply(lambda x: round(x, 2))
    df_analysis = df_2
    df_analysis = df_analysis[df_analysis["task"].isin(list(task_map.keys()))]
    df_analysis = df_analysis.sort_values(by="task")

    df_2.to_csv(f"{output_dir}/converged_words.csv", index=False)

    plt.rcParams["font.family"] = "serif"

    col_order = list(task_map.keys())

    for dataset in df_analysis["dataset"].unique():
        df_dataset = df_analysis[(df_analysis["dataset"] == dataset)]

        plt.figure(figsize=(8, 2))

        sns.set_theme(style="darkgrid")
        ax = sns.barplot(
            x="task",
            y="percentage",
            data=df_dataset,
            hue="threshold",
            palette="viridis",
            errorbar=None,
            order=col_order,
        )

        ax.set_ylabel("% of non conv.", fontsize=15)
        ax.set_xlabel("")

        ax.set_yticks(ax.get_yticks())
        ax.set_yticklabels([f"{round(int(y), 2)}" for y in ax.get_yticks()], fontsize=15)
        ax.set_yticklabels([f"{y}%" for y in ax.get_yticks()], fontsize=15)

        ax.set_xticks(ax.get_xticks())
        ax.set_xticklabels([task_map[t.get_text()] for t in ax.get_xticklabels()], fontsize=15)
        ax.set_ylim(0, 100)

        plt.legend(bbox_to_anchor=(0.5, 1.25), loc="upper center", ncols=5)

        ax.figure.savefig(f"{output_dir}/converged_words_{dataset}.pdf", bbox_inches="tight")

        plt.close()


if __name__ == "__main__":
    main()
