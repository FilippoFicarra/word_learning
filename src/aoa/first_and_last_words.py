import argparse
import os
import warnings

import pandas as pd

pd.options.display.max_colwidth = 3000
warnings.filterwarnings("ignore")


class FirstAndLast:
    def __init__(self, df_path: str, save_dir: str):
        self.df = pd.read_csv(df_path)
        self.df = self.df[(self.df.seed == 42) & (self.df.threshold == 0.07) & (self.df.dataset == "unified")]
        self.save_dir = save_dir

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

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

    def get_first_and_last(self):
        dd = pd.DataFrame()

        for task in list(self.task_map.keys()):
            task_df = self.df[self.df["task"] == task]
            task_df.sort_values("aoa_x", inplace=True)
            task_df = task_df[["word", "aoa_x"]]
            task_df = task_df.drop_duplicates("word")

            dd = pd.concat(
                [
                    dd,
                    pd.DataFrame(
                        {
                            "task": [self.task_map[task]],
                            "firsts": [",".join(task_df.head(10)["word"].values)],
                            "lasts": [",".join(task_df.tail(10)["word"].values[::-1])],
                        }
                    ),
                ]
            )
        # save to tex
        table = dd.to_latex(index=False, column_format="l|c|c", escape=False)

        with open(f"{self.save_dir}/aoa.tex", "w") as f:
            f.write(table)

    def get_specific_words_aoa(self, words):
        # create a dataframe with columns word, children, self.task_map.values
        save_df = pd.DataFrame(columns=["word", "children"] + list(self.task_map.values()))

        for word in words:
            word_df = self.df[self.df["word"] == word]
            dict_to_insert = {"word": word, "children": round(word_df["child_aoa"].values[0], 2)}

            for task in list(self.task_map.keys()):
                task_df = word_df[word_df["task"] == task]

                # print(task_df.columns)

                val = task_df["aoa_x"].values

                if val.size == 0:
                    val = "NA"

                else:
                    val = round(val[0], 2)

                dict_to_insert[self.task_map[task]] = val

            save_df = pd.concat([save_df, pd.DataFrame(dict_to_insert, index=[0])])

        table = save_df.to_latex(index=False, column_format="l|c|c|c|c|c|c|c|c|c|c", escape=False)

        with open(f"{self.save_dir}/aoa_specific.tex", "w") as f:
            f.write(table)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--df_path", type=str, default="src/signatures/results/aoa_and_predictors.csv")
    parser.add_argument("--save_dir", type=str, default="src/aoa/results/first_and_last")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    fa = FirstAndLast(**args.__dict__)

    words = ["the", "off", "water", "puzzle", "good", "orange", "go", "climb"]

    fa.get_first_and_last()
    fa.get_specific_words_aoa(["the", "off", "water", "puzzle", "good", "orange", "go", "climb"])
