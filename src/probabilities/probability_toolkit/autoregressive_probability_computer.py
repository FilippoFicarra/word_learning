from typing import Optional, Union

import pandas as pd
from minicons import scorer
from transformers import AutoModelForCausalLM

from tasks.probabilities.probability_toolkit import (
    ProbabilityComputer,
    ProbabilityDataModule,
)

TEXT_MAX_LENGTH_DEFAULT = 150


class AutoregressiveProbabilityComputer(ProbabilityComputer):
    def __init__(
        self,
        model_name: str,
        step: int,
        device: Optional[Union[int, str]] = "cuda",
        cache_dir: Optional[str] = "/cluster/scratch/fficarra",
    ):
        super().__init__(model_name, device, step)

        print(f"caching model in {cache_dir}")

        if model_name == "llama3":
            model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=cache_dir)
        else:
            model = model_name

        self.ilm_model = scorer.IncrementalLMScorer(
            model=model,
            device=self._device,
        )

    def get_probability_intermediate(self, dataloader: ProbabilityDataModule) -> pd.DataFrame:
        self.prob_map = {}
        self.intermediate_prob = pd.DataFrame(
            columns=[
                "word",
                "step",
                "sent",
                "p_c",
                "p_w_given_c",
            ]
        )
        self.intermediate_prob = self.intermediate_prob.astype(
            {
                "word": str,
                "step": float,
                "sent": int,
                "p_c": float,
                "p_w_given_c": float,
            }
        )

        dataloader.setup()

        for i, batch in enumerate(dataloader.test_dataloader()):
            scores = self.ilm_model.token_score(batch, prob=True)
            for j, sent in enumerate(scores):
                sent = sent[1:]
                p_c = 1
                p_w_given_c = 1

                for z in range(len(sent)):
                    if sent[z][0] == dataloader.word:
                        p_w_given_c *= sent[z][1]
                        break
                    p_c *= sent[z][1]

                self.intermediate_prob = pd.concat(
                    [
                        self.intermediate_prob,
                        pd.DataFrame(
                            {
                                "word": dataloader.word,
                                "step": self.step,
                                "sent": j + i * dataloader.batch_size,
                                "p_c": p_c,
                                "p_w_given_c": p_w_given_c,
                            },
                            index=[0],
                        ),
                    ],
                    ignore_index=True,
                )
        return self.intermediate_prob
