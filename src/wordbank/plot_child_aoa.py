import argparse
import os
import re
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.optimize import curve_fit
from tqdm import tqdm


def sigmoid(x, L, x0, k, b):
    y = L / (1 + np.exp(-k * (x - x0))) + b
    return y


def initialize_parameters(x, y):
    L = np.max(y)
    x0 = np.mean(x)
    k = 1.0 / (np.max(x) - np.min(x))
    b = np.min(y)
    return L, x0, k, b


def plot_aoa(
    x: List[float],
    y: List[float],
    curve_fit_data: List[float],
    item_definition: str,
    aoa: float,
    output_dir: str,
) -> None:
    """
    Plot the learning curve for a word.

    Args:
        x (List[float]): List of ages in months.
        y (List[float]): List of proportions of children who know the word at a certain age.
        item_definition (str): Definition of the word.
        aoa (float): Age of acquisition of the word.
        output_dir (str): Directory in which the learning curves are saved.

    Returns:
        None: The learning curve is saved in the output directory.

    """

    data = pd.DataFrame({"Age (months)": x, "Proportion (Children %)": y})
    data_fit = pd.DataFrame(
        {
            "Age (months)": np.linspace(min(x), max(x), 10000),
            "Proportion (Children %)": curve_fit_data,
        }
    )

    plt.figure(figsize=(3, 3))
    plt.grid(True)
    plt.tight_layout()
    plt.gca().set_position([0.2, 0.2, 0.7, 0.7])
    plt.gca().yaxis.set_label_position("right")

    sns.lineplot(x="Age (months)", y="Proportion (Children %)", data=data, color="black")
    sns.lineplot(x="Age (months)", y="Proportion (Children %)", data=data_fit, color="blue")
    plt.axhline(y=0.5, color="r", linestyle="--")
    plt.scatter(aoa, 0.5, color="r")

    plt.title(f"{item_definition}")
    plt.xticks(np.linspace(min(x) - 1, max(x), 4))
    plt.yticks([round(z, 2) for z in np.linspace(min(y), max(y), 5)])

    plt.savefig(f"{output_dir}/{item_definition}.png")
    plt.close()


def plot_child_aoa(input_dir: str, output_dir: str) -> None:
    """

    Args:
        input_dir (str): Directory containing the input file containg the proportion of children who know a word at a certain age.
        output_dir (str): Directory in which the learning curves are saved.

    Returns:
        None: The learning curves are saved in the output directory.

    """

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    aoa_df = pd.read_csv(f"{input_dir}/clean_wordbank_american.tsv", sep="\t")
    aoa_dict = aoa_df.drop_duplicates(subset="CleanedSingle", keep="first").set_index("CleanedSingle")["aoa"].to_dict()

    proportion_df = pd.read_csv(f"{input_dir}/child_american_proportion.csv")
    months = [int(m) for m in proportion_df.columns[3:]]

    summary_df = pd.DataFrame(columns=["word", "aoa", "L", "x0", "k", "b", "min", "max"])
    summary_df = summary_df.astype(
        {
            "word": str,
            "aoa": float,
            "L": float,
            "x0": float,
            "k": float,
            "b": float,
            "min": float,
            "max": float,
        }
    )

    for _, row in tqdm(proportion_df.iterrows(), total=proportion_df.shape[0]):
        item_definition = re.split("[.,(/*#\uff08]", row["item_definition"])[0].split(" ")[0].lower()
        proportion = row[3:].tolist()

        try:
            if aoa_dict[item_definition] < min(months) or aoa_dict[item_definition] > max(months):
                raise ValueError("Age of acquisition is out of range.")
        except (KeyError, ValueError):
            continue

        if min(proportion) < 0.5 and max(proportion) > 0.5:
            p0_sigmoid = initialize_parameters(months, proportion)
            popt, _ = curve_fit(sigmoid, months, proportion, p0=p0_sigmoid, maxfev=10000)
            x = np.linspace(min(months), max(months), 10000)

            curve_fit_data = sigmoid(x, *popt)

            intersection = np.argwhere(np.diff(np.sign(curve_fit_data - 0.5))).flatten()
            try:
                aoa = x[intersection[0]]
                plot_aoa(months, proportion, curve_fit_data, item_definition, aoa, output_dir)
                summary_df = pd.concat(
                    [
                        summary_df,
                        pd.DataFrame(
                            {
                                "word": item_definition,
                                "aoa": aoa,
                                "L": popt[0],
                                "x0": popt[1],
                                "k": popt[2],
                                "b": popt[3],
                                "min": min(months),
                                "max": max(months),
                            },
                            index=[0],
                        ),
                    ]
                )
                summary_df.to_csv(f"{output_dir}/aoa_summary.csv", index=False)
            except Exception:
                continue


def main():
    parser = argparse.ArgumentParser(description="Plot child AoA using wordbank data for American English.")
    parser.add_argument(
        "--input_dir",
        type=str,
        help="Path of the directory for the input file containg the proportion \
                         of children who know a word at a certain age.",
        default="src/wordbank/data",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Path of the directory in which the learning curves \
                         are saved.",
        default="src/wordbank/images",
    )
    args = parser.parse_args()

    plot_child_aoa(args.input_dir, args.output_dir)


if __name__ == "__main__":
    main()
