import argparse
import sys
import os
import subprocess

import agent

environments = {
    "fitness-course": agent.Environments.FitnessCourseEnvironment.FitnessCourseEnvironment,
    "hoverboard": agent.Environments.HoverboardEnvironment.HoverboardEnvironment,
    "vidcomic": agent.Environments.VidcomicEnvironment.VidcomicEnvironment,
}


def main(args):
    """
    Spawns the learner and worker processes in separate processes.
    Responsible for catching user input and restart workers or learners separately if needed.

    For the selected environment, this script checks if the user has the game mod in the games/<game>/build directory.
    If it's not there, the script offers to download it from GitHub releases automatically.
    """
    environment = get_environment_from_string(args.environment)

    if not game_mod_exists(environment):
        choice = input(f"Game mod for {environment.game.game_key} not found. Do you want to download it now? (y/n)")

        if choice.lower() == "y":
            environment.download_game_mod()
        else:
            print("Exiting...")
            return

    # Spawn the learner process


def game_mod_exists(environment):
    """
    Check if the game mod exists in the games/<game>/build directory.
    """
    game_mod_path = f"games/{environment.game.game_key}/build"

    return os.path.exists(game_mod_path)


def download_game_mod(environment):
    """
    Download the game mod from the GitHub releases page.
    """
    print("Lol you can't")
    exit(0)


def get_environment_from_string(environment):
    if environment in environments:
        return environments[environment]
    else:
        raise ValueError(f"Environment '{environment}' not found.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--environment", type=str, required=True)
    parser.add_argument("--redis-host", type=str, default="localhost")
    parser.add_argument("--redis-port", type=int, default=6379)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--wandb", type=bool, action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--commit", type=bool, action=argparse.BooleanOptionalAction, default=False if "pydevd" in sys.modules else True)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--project-key", type=str, default="rac1.fitness-course")

    args = parser.parse_args()
    main(args)
