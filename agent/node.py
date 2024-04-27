from Agent import Agent
from ReplayBuffer import PrioritizedReplayBuffer

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


def listen_for_messages(redis: Redis, replay_buffer: PrioritizedReplayBuffer):
    # Subscribe to the "replay_buffer" channel
    pubsub = redis.pubsub()
    pubsub.subscribe("rac1.fitness-course.replay_buffer")

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

                    replay_buffer.add(
                        transition.state,
                        transition.action,
                        transition.reward,
                        transition.next_state,
                        transition.done,
                        transition.hidden_state,
                        transition.cell_state,
                    )

    except Exception as e:
        print(e)

    # Restart ourselves if we get here
    print("Restarting listener...")
    listen_for_messages(redis, replay_buffer)


def save_model(agent: Agent, model_path: str):
    torch.save({
        'model_state_dict': agent.Q_eval.state_dict(),
        'optimizer_state_dict': agent.Q_eval.optimizer.state_dict(),
        'epsilon': agent.epsilon,
    }, model_path)


def load_model(agent: Agent, model_path: str):
    if os.path.exists(model_path):
        print(f"Loading model from {model_path}")
        checkpoint = torch.load(model_path)
        agent.Q_eval.load_state_dict(checkpoint['model_state_dict'])
        agent.Q_eval.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        agent.epsilon = checkpoint['epsilon']

        agent.Q_eval = agent.Q_eval.to(agent.Q_eval.device)

        agent.Q_eval.train()
    else:
        print(f"Model file {model_path} not found.")
        exit(0)


def start():
    # Parse arguments
    args = argparse.ArgumentParser()
    args.add_argument("--redis-host", type=str, default="localhost")
    args.add_argument("--redis-port", type=int, default=6379)
    args.add_argument("--model", type=str, default=None)
    args.add_argument("--wandb", type=bool, default=False if "pydevd" in sys.modules else True)
    args.add_argument("--commit", type=bool, default=False if "pydevd" in sys.modules else True)
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
    agent = Agent(gamma=0.99, epsilon=1.0, batch_size=batch_size, n_actions=13, eps_end=0.05,
                  input_dims=features, lr=learning_rate, sequence_length=sequence_length)

    # Load existing model if load_model is set
    if args.model:
        load_model(agent, args.model)
        agent.update_target_network()
        print("Loaded model from file")

        # Set Redis model
        model = pickle.dumps(agent.Q_eval.state_dict())
        redis.set("rac1.fitness-course.model", model)
        redis.set("rac1.fitness-course.model_timestamp", time.time())

    existing_model = redis.get("rac1.fitness-course.model") if args.model is None else None
    if existing_model is not None:
        agent.epsilon = float(redis.get("rac1.fitness-course.epsilon"))
        agent.Q_eval.load_state_dict(pickle.loads(existing_model))

        # Load optimizer if it exists
        existing_optimizer = redis.get("rac1.fitness-course.optimizer")
        if existing_optimizer is not None:
            agent.Q_eval.optimizer.load_state_dict(pickle.loads(existing_optimizer))

        agent.update_target_network()

        agent.Q_eval.train()

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
    thread = Thread(target=listen_for_messages, args=(redis, agent.replay_buffer))
    thread.daemon = True
    thread.start()

    losses = []

    steps = 0

    # For calculating samples per second
    last_time = time.time()
    last_samples = 0
    samples_history = []

    while True:
        if agent.replay_buffer.total <= 10000:
            time.sleep(0.1)
            continue

        #if agent.replay_buffer.new_samples > train_frequency:
        last_samples += agent.replay_buffer.new_samples

        loss = agent.learn()

        losses.append(loss)

        # Calculate samples per second
        if time.time() - last_time > 1:
            samples_per_second = last_samples / (time.time() - last_time)
            last_time = time.time()
            last_samples = 0

            samples_history.append(samples_per_second)
            samples_history = samples_history[-10:]

        # Update target network used for calculating the target Q values
        if steps % target_update_frequency == 0:
            agent.update_target_network()

        if steps % 20 == 0:
            if commit:
                model = pickle.dumps(agent.Q_eval.state_dict())
                optimizer = pickle.dumps(agent.Q_eval.optimizer.state_dict())

                redis.set("rac1.fitness-course.model", model)
                redis.set("rac1.fitness-course.optimizer", optimizer)
                redis.set("rac1.fitness-course.model_timestamp", time.time())

                redis.set("rac1.fitness-course.epsilon", agent.epsilon)

        # Updating model in Redis, log stuff for debub, make backups
        if steps % 200 == 0:
            # Get the last 100 scores from Redis key "avg_scores" and cast them to floats
            scores = redis.lrange("rac1.fitness-course.avg_scores", -100, -1)
            scores = [float(score) for score in scores]

            checkpoints = redis.lrange("rac1.fitness-course.checkpoints", -100, -1)
            checkpoints = [float(checkpoint) for checkpoint in checkpoints]

            if len(losses) > 0:
                print('avg loss: %.2f' % np.mean(losses[-100:]), 'avg_score: %.2f' % np.mean(scores),
                      'epsilon: %.2f' % agent.epsilon, 'samples/sec: %.2f' % np.mean(samples_history))

            # Save the model every 10000 steps
            if commit and steps % 10000 == 0:
                save_model(agent, f"models_bak/rac1_fitness-course_{int(steps/1000)}k.pth")
                print(f"Saved model to models_bak/rac1_fitness-course_{int(steps/1000)}k.pth")

            if args.wandb:
                wandb.log({
                    "avg_score": np.mean(scores),
                    "loss": np.mean(losses[-100:]),
                    "epsilon": agent.epsilon,
                    "samples_per_second": np.mean(samples_history),
                    "avg_checkpoints": np.mean(checkpoints),
                })

        steps += 1

        time.sleep(0.001)


if __name__ == "__main__":
    try:
        start()
    except KeyboardInterrupt:
        print("Exiting...")
        exit(0)
