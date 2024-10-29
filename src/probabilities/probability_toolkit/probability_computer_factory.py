from typing import Optional, Union

from tasks.probabilities.probability_toolkit import (
    AutoregressiveProbabilityComputer,
    ProbabilityComputer,
)


class ProbabilityComputerFactory:
    @staticmethod
    def get_probability_computer(
        model_name: str,
        step: int,
        device: Optional[Union[int, str]] = "cuda",
        cache_dir: Optional[str] = "/cluster/scratch/fficarra",
    ) -> ProbabilityComputer:
        if "gpt2" in model_name or "llama" in model_name:
            return AutoregressiveProbabilityComputer(model_name, device, step, cache_dir)
        else:
            raise ValueError(f"Model type {model_name} not supported.")
