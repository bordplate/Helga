import time
import argparse
import sys

import torch
import numpy as np
import wandb

from PPO.PPOAgent import PPOAgent
from RedisHub import RedisHub

from util import update_graph_html


class Config:
    learning_rate       = 1e-4
    features            = 18 + 128
    actions             = 7
    batch_size          = 1024 * 8
    mini_batch_size     = int(batch_size / 8)
    sequence_length     = 8

    gamma = 0.99
    K_epochs = 10
    eps_clip = 0.2

    @staticmethod
    def serialize():
        return {key: Config.__dict__[key] for key in Config.__dict__ if not key.startswith("__")}


def start(args):
    commit = args.commit

    # redis = redis_from_url(f"redis://{args.redis_host}:{args.redis_port}")
    redis = RedisHub(f"redis://{args.redis_host}:{args.redis_port}", "rac1.fitness-course.rollout_buffer")

    # Create an agent
    agent = PPOAgent(
        Config.features,
        Config.actions,
        Config.learning_rate,
        Config.learning_rate,
        Config.batch_size,
        Config.mini_batch_size,
        Config.gamma,
        Config.K_epochs,
        Config.eps_clip
    )

    # Load existing model if load_model is set
    if args.model:
        redis.load_model_from_file(agent, args.model)
        print("Loaded model from file")

        # Set Redis model
        redis.save_model(agent)
    else:
        # Restores model from Redis if it exists
        redis.restore_model(agent)

    if args.wandb:
        current_run_id = redis.redis.get("rac1.fitness-course.wandb_run_id")
        current_run_id = current_run_id.decode() if current_run_id is not None else None

        wandb.init(
            project="rac1-fitness-course",
            id=current_run_id,
            config=Config.serialize(),
            resume="must" if current_run_id is not None else None
        )

        # Set current wandb run in Redis
        redis.redis.set("rac1.fitness-course.wandb_run_id", wandb.run.id)

        update_graph_html(wandb.run.get_url())

    # Start listening for messages on a separate thread
    redis.start_listening(agent)

    losses = []
    policy_losses = []
    value_losses = []
    entropy_losses = []

    steps = 0

    print("Starting training loop...")
    n_processed = 0

    while True:
        processed = False

        print("\r", end="")
        for i, replay_buffer in enumerate(agent.replay_buffers):
            print(f"[{i}:{replay_buffer.total}", end="")

            if replay_buffer.ready:
                print("*]", end="")
                n_processed += replay_buffer.total

                loss, policy_loss, value_loss, entropy_loss = agent.learn(replay_buffer)

                losses.append(loss)
                policy_losses.append(policy_loss)
                value_losses.append(value_loss)
                entropy_losses.append(entropy_loss)

                processed = True
                break
            else:
                print("]", end="")

        if not processed:
            time.sleep(0.1)
            continue

        steps += 1

        if commit:
            redis.save_model(agent)

        # Updating model in Redis, log stuff for debub, make backups
        if steps % 5 == 0:
            # Get the last 100 scores from Redis key "avg_scores" and cast them to floats
            scores = redis.redis.lrange("rac1.fitness-course.avg_scores", -100, -1)
            scores = [float(score) for score in scores]

            checkpoints = redis.redis.lrange("rac1.fitness-course.checkpoints", -100, -1)
            checkpoints = [float(checkpoint) for checkpoint in checkpoints]

            if len(losses) > 0:
                print('\rstep:', steps,
                      'avg loss: %.2f' % np.mean(losses[-100:]),
                      'avg_score: %.2f' % np.mean(scores),
                      'policy_loss: %.2f' % np.mean(policy_losses[-100:]),
                      'value_loss: %.2f' % np.mean(value_losses[-100:]),
                      'entropy_loss: %.2f' % np.mean(entropy_losses[-100:])
                )

            # Save the model every 15 steps
            if commit and steps % 15 == 0:
                redis.save_model_to_file(agent, f"models_bak/rac1_fitness-course_{steps}.pth")
                print(f"Saved model to models_bak/rac1_fitness-course_{steps}.pth")

            if args.wandb:
                wandb.log({
                    "avg_score": np.mean(scores),
                    "loss": np.mean(losses[-100:]),
                    "avg_checkpoints": np.mean(checkpoints),
                    "policy_loss": np.mean(policy_losses[-100:]),
                    "value_loss": np.mean(value_losses[-100:]),
                    "entropy_loss": np.mean(entropy_losses[-100:]),
                })


if __name__ == "__main__":
    torch.set_num_threads(1)

    try:
        # Parse arguments
        args = argparse.ArgumentParser()
        args.add_argument("--redis-host", type=str, default="localhost")
        args.add_argument("--redis-port", type=int, default=6379)
        args.add_argument("--model", type=str, default=None)
        args.add_argument("--wandb", type=bool, action=argparse.BooleanOptionalAction,
                          default=False if "pydevd" in sys.modules else True)
        args.add_argument("--commit", type=bool, action=argparse.BooleanOptionalAction,
                          default=False if "pydevd" in sys.modules else True)
        args = args.parse_args()

        start(args)
    except KeyboardInterrupt:
        print("Exiting...")
        exit(0)
