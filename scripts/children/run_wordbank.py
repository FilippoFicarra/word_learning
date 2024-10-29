import subprocess
from typing import Dict, List

from omegaconf import OmegaConf
from tqdm import tqdm


def create_command(command: List[str], args: Dict[str, str]) -> list:
    """
    Create the command to execute clean_wordbank_words.py

    Args:
        args (Dict): Dictionary of arguments

    Returns:
        list: The command as a list of strings
    """
    for key, value in args.items():
        if value:
            command.extend([key, value])
    return command


def run_tasks(config: Dict) -> None:
    """
    Run the tasks in the configuration file

    Args:
        config (Dict): The configuration file

    Returns:
        None
    """
    for script, tasks in tqdm(config["scripts"].items()):
        for _, task in tasks.items():
            file = task.get("file")
            args = task.get("args", {})
            command = create_command([script, file], args)

            print(f"Running {' '.join(command)}")
            subprocess.run(command)


if __name__ == "__main__":
    conf = OmegaConf.from_cli()
    yaml_conf = OmegaConf.load(conf.config)

    run_tasks(yaml_conf)
