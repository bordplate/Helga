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
    learning_rate_critic    = 3e-4
    learning_rate_actor     = 3e-4
    features                = 27 + 128
    actions                 = 7
    buffer_size             = 1024 * 64
    batch_size              = 1024 * 16
    mini_batch_size         = 1024
    sequence_length         = 30

    gamma                   = 0.995
    K_epochs                = 10
    eps_clip                = 0.2
    log_std                 = -1.0
    ent_coef                = 0.001
    lambda_gae              = 0.9
    critic_loss_coeff       = 1.0
    # kl_threshold            = 0.025
    kl_threshold            = 0.01

    @staticmethod
    def serialize():
        return {key: Config.__dict__[key] for key in Config.__dict__ if not key.startswith("__") and not callable(Config.__dict__[key])}


def start(args):
    if not torch.cuda.is_available():
        print("CUDA not available, exiting...")
        exit(0)

    device = torch.device('cuda:0')
    torch.cuda.empty_cache()

    commit = args.commit

    # redis = redis_from_url(f"redis://{args.redis_host}:{args.redis_port}")
    redis = RedisHub(f"redis://{args.redis_host}:{args.redis_port}", "rac1.fitness-course.rollout_buffer", device=device)

    # Unblock potentially stale workers
    redis.unblock_workers()

    # Create an agent
    agent = PPOAgent(
        state_dim=Config.features,
        action_dim=Config.actions,
        lr_actor=Config.learning_rate_actor,
        lr_critic=Config.learning_rate_critic,
        buffer_size=Config.buffer_size,
        batch_size=Config.batch_size,
        mini_batch_size=Config.mini_batch_size,
        gamma=Config.gamma,
        K_epochs=Config.K_epochs,
        eps_clip=Config.eps_clip,
        log_std=Config.log_std,
        ent_coef=Config.ent_coef,
        cl_coeff=Config.critic_loss_coeff,
        lambda_gae=Config.lambda_gae,
        kl_threshold=Config.kl_threshold,
        device=device
    )

    agent.policy.actor.max_log_std = 0.8

    # agent.action_mask = redis.get_action_mask()

    # Load existing model if load_model is set
    if args.model:
        redis.load_model_from_file(agent, args.model)
        print("Loaded model from file")

        # Set Redis model
        redis.save_model(agent)
    else:
        # Restores model from Redis if it exists
        if not redis.restore_model(agent):
            print("No existing model found in Redis")

            # Save our current model to Redis
            redis.save_model(agent)
        else:
            print("Restored model from Redis")

            # Reapply learning rate to optimizer
            agent.optimizer.param_groups[0]['lr'] = Config.learning_rate_actor
            agent.optimizer.param_groups[1]['lr'] = Config.learning_rate_critic

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
    kl_divs = []

    steps = 0

    print("Starting training loop...")

    last_kl_div = 0

    while True:
        processed = False

        print("\r", end="")

        all_buffers_ready = all([replay_buffer.ready for replay_buffer in agent.replay_buffers])

        for i, replay_buffer in enumerate(agent.replay_buffers):
            print(f"[{i}:{replay_buffer.total}]", end="")

        if all_buffers_ready and len(agent.replay_buffers) > 0:
            loss, policy_loss, value_loss, entropy_loss, last_kl_div = agent.learn()

            losses.append(loss)
            policy_losses.append(policy_loss)
            value_losses.append(value_loss)
            entropy_losses.append(entropy_loss)

            processed = True

        if not processed:
            time.sleep(0.1)
            continue

        steps += 1

        if commit:
            redis.save_model(agent)

            # Clear the buffers
            for replay_buffer in agent.replay_buffers:
                replay_buffer.clear()

                # Notify workers
                redis.redis.publish(f"{replay_buffer.owner}.full", "False")

        # Updating model in Redis, log stuff for debub, make backups
        if steps % 1 == 0:
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
                      'entropy_loss: %.2f' % np.mean(entropy_losses[-100:]),
                      'kl_div: %.5f' % last_kl_div,
                )

                # log_std_params = [ "%.5f" % x for x in agent.policy.actor.log_std.squeeze().tolist() ]
                # print(f"log_std: {log_std_params}")

            # Save the model every 15 steps
            if commit and steps % 15 == 0:
                redis.save_model_to_file(agent, f"models_bak/rac1_fitness-course_{steps}.pth")
                print(f"Saved model to models_bak/rac1_fitness-course_{steps}.pth")

            if args.wandb:
                wandb.log({
                    "avg_score": np.mean(scores),
                    "loss": losses[-1] if len(losses) > 0 else 0,
                    "avg_checkpoints": np.mean(checkpoints),
                    "policy_loss": policy_losses[-1] if len(policy_losses) > 0 else 0,
                    "value_loss": value_losses[-1] if len(value_losses) > 0 else 0,
                    "entropy_loss": entropy_losses[-1] if len(entropy_losses) > 0 else 0,
                    "kl_div": last_kl_div,
                })

            # agent.action_mask = redis.get_action_mask()


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
