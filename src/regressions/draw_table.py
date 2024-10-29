import argparse
import os
import warnings

import pandas as pd

warnings.filterwarnings("ignore")


class LatexTable:
    def __init__(self, df: pd.DataFrame, save_dir: str):
        self.df = df
        self.task_map = {
            "corpus_positive": r"\corpuspositiveEstimator",
            "corpus_negative": r"\corpusnegativeEstimator",
            "corpus_combined": r"\corpuscombinedEstimator",
            "intrinsic_positive": r"\intpositiveEstimator",
            "intrinsic_negative": r"\intnegativeEstimator",
            "intrinsic_combined": r"\intcombinedEstimator",
            "extrinsic_positive": r"\extpositiveEstimator",
            "extrinsic_negative": r"\extnegativeEstimator",
            "extrinsic_combined": r"\extcombinedEstimator",
        }

        self.task_map_reverse = {v: k for k, v in self.task_map.items()}

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        self.save_dir = save_dir

    def to_latex(self) -> None:
        df = self.df.copy()
        if "max_vif" in df.columns:
            df = df.drop(columns=["max_vif"])

        if "model" in df.columns and len(df["model"].unique()) == 1:
            df = df.drop(columns=["model"])

        if "dataset" in df.columns and len(df["dataset"].unique()) == 1:
            df = df.drop(columns=["dataset"])

        if "threshold" in df.columns and len(df["threshold"].unique()) == 1:
            df = df.drop(columns=["threshold"])

        # enforce dataset order to be (childes, babylm and unified)
        df["dataset"] = pd.Categorical(df["dataset"], ["childes", "babylm", "unified"])
        df["task"] = pd.Categorical(df["task"], self.task_map.keys())

        df = df.sort_values(by=["dataset", "task"])

        df["task"] = df["task"].map(self.task_map)
        df = df.rename(columns={"task": r"\signature", "num_words": r"\#words"})
        df = df.round(3)
        table = df.to_latex(index=False, escape=False, column_format="l" + "c" * (len(df.columns) - 1))

        with open(f"{self.save_dir}/lm_regresion.tex", "w") as f:
            f.write(table)


def parse_args():
    parser = argparse.ArgumentParser(description="Convert CSV to LaTeX table")

    parser.add_argument(
        "--result_file",
        type=str,
        default="src/regressions/results/lm_regression.csv",
        help="Folder with CSV files (default: src/regressions/results/lm_regression.csv)",
    )

    parser.add_argument(
        "--save_dir",
        type=str,
        default="src/regressions/results/tables",
        help="File to save the output (default: src/regressions/results/tables)",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    result_df = pd.read_csv(args.result_file)
    latex_table = LatexTable(df=result_df, save_dir=args.save_dir)

    latex_table.to_latex()
