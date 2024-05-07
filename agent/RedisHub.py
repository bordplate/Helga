import os
import pickle
import string
from collections import namedtuple
import random
import time

import torch

from threading import Thread

from redis import from_url as redis_from_url

from PPOAgent import PPOAgent
from RolloutBuffer import RolloutBuffer


Transition = namedtuple('Transition', ('state', 'action', 'reward',
                                       'done',  'logprob', 'state_value', 'mu', 'log_std'))
TransitionMessage = namedtuple('TransitionMessage', ('transition', 'worker_name'))


class RedisHub:
    def __init__(self, redis_url, identifier):
        self.redis = redis_from_url(redis_url)
        self.key = identifier

        self.latest_model = 0
        self.model = None

        # Randomly generate worker ID
        self.worker_id = ''.join(random.choices(string.ascii_uppercase + string.digits, k=8))

    def add(self, state_sequence, actions, reward, last_done, logprob, state_value, mu, log_std):
        transition = Transition(state_sequence, actions, reward, last_done, logprob, state_value,
                                mu, log_std)
        message = TransitionMessage(transition, self.worker_id)

        # Pickle the transition and publish it to the "replay_buffer" channel
        data = pickle.dumps(message)
        self.redis.publish(self.key, data)

    def get_latest_model(self):
        model_timestamp = self.redis.get("rac1.fitness-course.model_timestamp")
        if model_timestamp is not None and float(model_timestamp) > self.latest_model:
            # Load the latest model from Redis
            model_pickled = self.redis.get("rac1.fitness-course.model")
            if model_pickled is not None:
                self.model = pickle.loads(model_pickled)
                self.latest_model = float(model_timestamp)

        return self.model

    def get_model(self):
        model_pickle = self.redis.get("rac1.fitness-course.model")
        if model_pickle is not None:
            return pickle.loads(model_pickle)

        return None

    def get_optimizer(self):
        optimizer_pickle = self.redis.get("rac1.fitness-course.optimizer")
        if optimizer_pickle is not None:
            return pickle.loads(optimizer_pickle)

        return None

    def get_new_model(self):
        """
        Gets the latest model if we don't already have it, otherwise None
        """
        model_timestamp = self.redis.get("rac1.fitness-course.model_timestamp")
        if model_timestamp is not None and float(model_timestamp) > self.latest_model:
            # Load the latest model from Redis
            model_pickled = self.redis.get("rac1.fitness-course.model")
            if model_pickled is not None:
                self.model = pickle.loads(model_pickled)
                self.latest_model = float(model_timestamp)
                return self.model

        return None

    def save_model(self, agent: PPOAgent):
        if self.model is None:
            self.model = pickle.dumps(agent.policy.state_dict())

        optimizer = pickle.dumps(agent.optimizer.state_dict())

        self.redis.set("rac1.fitness-course.model", self.model)
        self.redis.set("rac1.fitness-course.optimizer", optimizer)
        self.redis.set("rac1.fitness-course.model_timestamp", time.time())

        self.model = pickle.dumps(agent.policy.state_dict())

    def save_model_to_file(self, agent: PPOAgent, filename):
            torch.save({
                'model_state_dict': agent.policy.state_dict(),
                'optimizer_state_dict': agent.optimizer.state_dict()
            }, filename)

    def load_model_from_file(self, agent: PPOAgent, model_path):
        if os.path.exists(model_path):
            print(f"Loading model from {model_path}")
            checkpoint = torch.load(model_path)
            agent.load_policy_dict(checkpoint['model_state_dict'])
            agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            agent.policy = agent.policy.to(agent.device)

            agent.policy.train()
        else:
            print(f"Model file {model_path} not found.")
            exit(0)

    def restore_model(self, agent: PPOAgent):
        existing_model = self.get_model()
        if existing_model is not None:
            agent.load_policy_dict(existing_model)

            # Load optimizer if it exists
            existing_optimizer = self.get_optimizer()
            if existing_optimizer is not None:
                agent.optimizer.load_state_dict(existing_optimizer)

            print("Loaded existing model from Redis")

    def listen_for_messages(self, agent: PPOAgent):
        # Subscribe to the "replay_buffer" channel
        pubsub = self.redis.pubsub()
        pubsub.subscribe("rac1.fitness-course.rollout_buffer")

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
                            buffers[data.worker_name] = RolloutBuffer(data.worker_name, 1000000, agent.batch_size, agent.gamma,
                                                                      agent.lambda_gae)
                            agent.replay_buffers.append(buffers[data.worker_name])
                            replay_buffer = buffers[data.worker_name]

                        replay_buffer.add(
                            transition.state,
                            transition.action,
                            transition.reward,
                            transition.done,
                            transition.logprob,
                            transition.state_value,
                            transition.mu,
                            transition.log_std
                        )
        except IndexError as e:
            print(e)

        # Restart ourselves if we get here
        print("Restarting listener...")
        self.listen_for_messages(agent)

    def start_listening(self, agent: PPOAgent):
        thread = Thread(target=self.listen_for_messages, args=(agent,))
        thread.daemon = True
        thread.start()