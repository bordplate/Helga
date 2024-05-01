from PPOAgent import PPOAgent
from ReplayBuffer import ReplayBuffer

from redis import Redis, ConnectionPool
from redis import from_url as redis_from_url

import torch
import numpy as np
import wandb

import pickle
import time
import argparse
import os
import sys

from threading import Thread, Lock

from learn import update_graph_html


def listen_for_messages(redis: Redis, agent: PPOAgent):
    # Subscribe to the "replay_buffer" channel
    pubsub = redis.pubsub()
    pubsub.subscribe("rac1.fitness-course.replay_buffer")

    buffers = {}

    for buffer in agent.replay_buffers:
        buffers[buffer.owner] = agent.replay_buffers

    # Start listening for messages
    try:
        i = 0
        while True:
            i += 1

            messages = []
            while True:
                message = pubsub.get_message(ignore_subscribe_messages=True)

                if message is None:
                    break

                messages.append(message)

                if len(messages) > 1000:
                    print("Flushing messages")

                    while pubsub.get_message(ignore_subscribe_messages=True) is not None:
                        pass

                    break

            for message in messages:
                if message["type"] == "message":
                    # Convert the message to states from bytes
                    data = message["data"]
                    data = pickle.loads(data)
                    transition = data.transition

                    if data.worker_name in buffers:
                        replay_buffer = buffers[data.worker_name]
                    else:
                        buffers[data.worker_name] = ReplayBuffer(data.worker_name, 10000)
                        agent.replay_buffers.append(buffers[data.worker_name])
                        replay_buffer = buffers[data.worker_name]

                    replay_buffer.add(
                        transition.state,
                        transition.action,
                        transition.reward,
                        transition.done,
                        transition.logprob,
                        transition.state_value,
                        transition.hidden_state,
                        transition.cell_state,
                    )
    except Exception as e:
        print(e)

    # Restart ourselves if we get here
    print("Restarting listener...")
    listen_for_messages(redis, agent)


def save_model(agent: PPOAgent, model_path: str):
    torch.save({
        'model_state_dict': agent.policy.state_dict(),
        'optimizer_state_dict': agent.optimizer.state_dict(),
        'action_std': agent.action_std,
    }, model_path)


def load_model(agent: PPOAgent, model_path: str):
    if os.path.exists(model_path):
        print(f"Loading model from {model_path}")
        checkpoint = torch.load(model_path)
        agent.load_policy_dict(checkpoint['model_state_dict'])
        agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        agent.action_std = checkpoint['action_std']

        agent.policy = agent.policy.to(agent.device)

        agent.policy.train()
    else:
        print(f"Model file {model_path} not found.")
        exit(0)


def start():
    # Parse arguments
    args = argparse.ArgumentParser()
    args.add_argument("--redis-host", type=str, default="localhost")
    args.add_argument("--redis-port", type=int, default=6379)
    args.add_argument("--model", type=str, default=None)
    args.add_argument("--wandb", type=bool, action=argparse.BooleanOptionalAction, default=False if "pydevd" in sys.modules else True)
    args.add_argument("--commit", type=bool, action=argparse.BooleanOptionalAction, default=False if "pydevd" in sys.modules else True)
    args = args.parse_args()

    commit = args.commit

    # Hyperparameters
    learning_rate = 7e-6
    features = 15 + 128
    batch_size = 256
    train_frequency = 4
    target_update_frequency = 10000  # How often we update the target network
    sequence_length = 8

    redis = redis_from_url(f"redis://{args.redis_host}:{args.redis_port}")

    # Create an agent
    # agent = Agent(gamma=0.99, epsilon=1.0, batch_size=batch_size, n_actions=13, eps_end=0.05,
    #               input_dims=features, lr=learning_rate, sequence_length=sequence_length)

    agent = PPOAgent(features, 7, 5e-6, 1e-5, 0.99, 10, 0.2)

    # Load existing model if load_model is set
    if args.model:
        load_model(agent, args.model)
        print("Loaded model from file")

        # Set Redis model
        model = pickle.dumps(agent.policy.state_dict())
        redis.set("rac1.fitness-course.model", model)
        redis.set("rac1.fitness-course.model_timestamp", time.time())

    existing_model = redis.get("rac1.fitness-course.model") if args.model is None else None
    if existing_model is not None:
        agent.action_std = float(redis.get("rac1.fitness-course.action_std"))
        agent.load_policy_dict(pickle.loads(existing_model))

        # Load optimizer if it exists
        existing_optimizer = redis.get("rac1.fitness-course.optimizer")
        if existing_optimizer is not None:
            agent.optimizer.load_state_dict(pickle.loads(existing_optimizer))

        print("Loaded existing model from Redis")

    if args.wandb:
        current_run_id = redis.get("rac1.fitness-course.wandb_run_id")
        current_run_id = current_run_id.decode() if current_run_id is not None else None

        wandb.init(project="rac1-fitness-course", id=current_run_id, config={
            "learning_rate": learning_rate,
            "sequence_length": sequence_length,
            "batch_size": batch_size,
            "features": features,
            "train_frequency": train_frequency,
            "target_update_frequency": target_update_frequency,
        }, resume="must" if current_run_id is not None else None)

        # Set current wandb run in Redis
        redis.set("rac1.fitness-course.wandb_run_id", wandb.run.id)

        update_graph_html(wandb.run.get_url())

    # Start listening for messages on a separate thread
    thread = Thread(target=listen_for_messages, args=(redis, agent))
    thread.daemon = True
    thread.start()

    losses = []

    steps = 0

    # For calculating samples per second
    last_time = time.time()
    last_samples = 0
    samples_history = []

    action_std = 0.6  # starting std for action distribution (Multivariate Normal)
    action_std_decay_rate = 0.025  # linearly decay action_std (action_std = action_std - action_std_decay_rate)
    min_action_std = 0.04  # minimum action_std (stop decay after action_std <= min_action_std)
    action_std_decay_freq = int(4000)  # action_std decay frequency (in num timesteps)

    print("Starting training loop...")
    n_processed = 0

    while True:
        processed = False

        print("\rProcessed samples: ", n_processed, end="")

        for replay_buffer in agent.replay_buffers:
            if replay_buffer.total >= 1500:
                n_processed += replay_buffer.total
                loss = agent.learn(replay_buffer)
                losses.append(loss)
                processed = True

                print("\rProcessed samples: ", n_processed, end="")

        if not processed:
            time.sleep(0.1)
            continue

        steps += 1

        # Calculate samples per second
        # if time.time() - last_time > 1:
        #     samples_per_second = last_samples / (time.time() - last_time)
        #     last_time = time.time()
        #     last_samples = 0
        #
        #     samples_history.append(samples_per_second)
        #     samples_history = samples_history[-10:]

        if commit:
            model = pickle.dumps(agent.policy.state_dict())
            optimizer = pickle.dumps(agent.optimizer.state_dict())

            redis.set("rac1.fitness-course.model", model)
            redis.set("rac1.fitness-course.optimizer", optimizer)
            redis.set("rac1.fitness-course.model_timestamp", time.time())

            redis.set("rac1.fitness-course.action_std", agent.action_std)

        # Updating model in Redis, log stuff for debub, make backups
        if steps % 5 == 0:
            agent.decay_action_std(action_std_decay_rate, min_action_std)

            # Get the last 100 scores from Redis key "avg_scores" and cast them to floats
            scores = redis.lrange("rac1.fitness-course.avg_scores", -100, -1)
            scores = [float(score) for score in scores]

            checkpoints = redis.lrange("rac1.fitness-course.checkpoints", -100, -1)
            checkpoints = [float(checkpoint) for checkpoint in checkpoints]

            if len(losses) > 0:
                print('\rstep:', steps, 'avg loss: %.2f' % np.mean(losses[-100:]), 'avg_score: %.2f' % np.mean(scores),
                      'epsilon: %.2f' % agent.action_std) # , 'samples/sec: %.2f' % np.mean(samples_history))

            # Save the model every 15 steps
            if commit and steps % 15 == 0:
                save_model(agent, f"models_bak/rac1_fitness-course_{steps}.pth")
                print(f"Saved model to models_bak/rac1_fitness-course_{steps}.pth")

            if args.wandb:
                wandb.log({
                    "avg_score": np.mean(scores),
                    "loss": np.mean(losses[-100:]),
                    "epsilon": agent.action_std,
                    # "samples_per_second": np.mean(samples_history),
                    "avg_checkpoints": np.mean(checkpoints),
                })


if __name__ == "__main__":
    try:
        start()
    except KeyboardInterrupt:
        print("Exiting...")
        exit(0)
