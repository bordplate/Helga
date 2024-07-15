import time

import torch

from RunningStats import RunningStats
from Watchdog import Watchdog
from Environments.FitnessCourseEnvironment import FitnessCourseEnvironment
from Environments.GasparEnvironment import GasparEnvironment

import numpy as np

from PPO.PPOAgent import PPOAgent

from RedisHub import RedisHub

features = 28 + 256*5
sequence_length = 1

configuration = {
    "model": None,
    "epsilon": 0.005,
    "min_epsilon": 0.005,
}


def start_worker(args):
    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        torch.cuda.empty_cache()

    # Get paths from arguments
    rpcs3_path = args.rpcs3_path
    process_name = args.process_name
    render = args.render
    eval_mode = args.eval
    project_key = args.project_key
    device = "cpu" if args.cpu_only else device

    # If we're not being debugged in PyCharm mode, we start a new RPCS3 process using the watchdog, otherwise we connect to an existing one
    pid = 0
    import sys
    if not "pydevd" in sys.modules:
        # Make new environment and watchdog
        watchdog = Watchdog(render=eval_mode)
        if not watchdog.start(force=True):
            print("Damn, watchdog failed to start the process!")
            exit(-1)
        else:
            pid = watchdog.pid
    else:
        # Find the rpcs3 process
        import psutil
        for proc in reversed(list(psutil.process_iter())):
            if proc.name() == process_name:
                pid = proc.pid
                break

    env = FitnessCourseEnvironment(pid=pid, eval_mode=eval_mode, device=device)

    # Watchdog starts RPCS3 and the game for us if it's not already running
    env.start()

    # Connect to Redis
    redis = RedisHub(f"redis://{args.redis_host}:{args.redis_port}", f"{project_key}.rollout_buffer", device=device)

    # state_running_stats = RunningStats()

    # Agent that we will use only for inference, learning related parameters are not used
    agent = PPOAgent(features, 7, log_std=-0.5, device=device)

    if eval_mode:
        pass
        # Draws a visualization of the actions and other information
        import Visualizer

        if "pydevd" not in sys.modules:
            import Plotter
            Plotter.start_plotting()

        # agent.policy.actor.max_log_std = 0.0000001
        agent.policy.actor.max_log_std = 0.25
    else:
        agent.policy.actor.max_log_std = 0.8

    total_steps = 0
    episodes = 0
    scores = []

    # Start stepping through the environment
    while True:
        torch.cuda.empty_cache()

        agent.start_new_episode()
        state, _, _ = env.reset()

        # state_running_stats.update(state)
        # state = state_running_stats.normalize(state)

        state_sequence = torch.zeros((sequence_length, features), dtype=torch.bfloat16).to(device)
        state_sequence[-1] = state
    
        accumulated_reward = 0
        steps = 0

        last_done = True

        must_check_new_model = False

        agent.action_mask = redis.get_action_mask()

        while True:
            while redis.check_buffer_full():
                must_check_new_model = True
                time.sleep(0.1)

            if steps % 5 == 0 or must_check_new_model:
                new_model = redis.get_new_model()
                if new_model is not None:
                    agent.load_policy_dict(new_model)

            # old_hidden_state = agent.policy.actor.hidden_state.clone().detach()
            # old_cell_state = agent.policy.actor.cell_state.clone().detach()

            # actions, logprob, state_value = (torch.zeros(7), torch.zeros(1), torch.zeros(1))
            actions, logprob, state_value = agent.choose_action(state_sequence.unsqueeze(dim=0))
            actions = actions.to(dtype=torch.float32).squeeze().cpu()

            new_state, reward, done = env.step(actions)

            # state_running_stats.update(new_state)
            # new_state = state_running_stats.normalize(new_state)

            time_left = (30 * 30 - env.time_since_last_checkpoint) / 30

            # Give some run-in time before we start evaluating the model so state observations are normalized properly
            if not eval_mode:
                # redis.add(state_sequence, actions, reward, last_done, logprob, state_value, agent.policy.actor.hidden_state, agent.policy.actor.cell_state)
                redis.add(state_sequence, actions, reward, last_done, logprob, state_value, None, None)
                pass
            elif eval_mode:
                # Visualize the actions
                Visualizer.draw_state_value_face(state_value)
                Visualizer.draw_score_and_checkpoint(accumulated_reward, env.n_checkpoints)
                Visualizer.render_raycast_data(np.float16(env.game.get_collisions(normalized=False)))
                Visualizer.draw_bars(actions, state_value, time_left/30)

                if "pydevd" not in sys.modules:
                    Plotter.add_data(state_value.item(), reward)

            # Roll the state sequence and append the new normalized state
            new_state_sequence = torch.roll(state_sequence, -1, 0)
            new_state_sequence[-1] = new_state

            state_sequence = new_state_sequence
            last_done = done

            accumulated_reward += reward
            steps += 1
            total_steps += 1

            if eval_mode or steps % 5 == 0:
                print(f"Score: %6.2f    death: %05.2f checkpoint: %d  closest_dist: %02.2f  value: %3.2f  highest_z: %3.2f  from_ground: %3.2f         " % (
                    accumulated_reward,
                    time_left,
                    env.n_checkpoints,
                    env.closest_distance_to_checkpoint,
                    state_value.item() if state_value is not None else 0.0,
                    env.highest_grounded_z,
                    env.distance_from_ground
                ), end="\r")

            if done:
                break

        scores.append(accumulated_reward)
        avg_score = np.mean(scores[-100:])

        print('score: %.2f' % accumulated_reward, 'checkpoints: %d' % env.n_checkpoints,
              'avg score: %.2f' % avg_score)

        # Append score to Redis key "scores"
        if not eval_mode:
            redis.redis.rpush(f"{project_key}.avg_scores", accumulated_reward)
            redis.redis.rpush(f"{project_key}.checkpoints", env.n_checkpoints)

        episodes += 1


if __name__ == "__main__":
    # Catch Ctrl+C and exit gracefully
    try:
        import argparse

        parser = argparse.ArgumentParser()

        parser.add_argument("--rpcs3-path", type=str, required=False)
        parser.add_argument("--process-name", type=str, default="rpcs3")
        parser.add_argument("--redis-host", type=str, default="localhost")
        parser.add_argument("--redis-port", type=int, default=6379)
        parser.add_argument("--render", action="store_true", default=True)
        parser.add_argument("--force-watchdog", action="store_false")
        parser.add_argument("--eval", type=bool, action=argparse.BooleanOptionalAction, default=False)
        parser.add_argument("--project-key", type=str, default="rac1.fitness-course")
        parser.add_argument("--cpu-only", action="store_true", default=False)

        args = parser.parse_args()

        with torch.no_grad() and torch.autograd.no_grad() and torch.inference_mode():
            start_worker(args)
    except KeyboardInterrupt:
        print("Exiting...")
        exit(0)
