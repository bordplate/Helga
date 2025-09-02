import time
import numpy as np
import torch

# from ..Game.Game import Vector3
# from ..Game.RC3Game import RC3Game
from agent.Game.Game import Vector3
from agent.Game.RC3Game import RC3Game

from agent.Environments.RatchetEnvironment import RatchetEnvironment


class CaptureTheFlagEnvironment(RatchetEnvironment):
    def __init__(self, pid):
        super().__init__()
        self.game = RC3Game(pid)

        self.resets = 0

        self.timeout = 30 * 30

    def reset(self):
        self.game.reset(self.resets)
        self.timeout = 30 * 30

        # self.game.set_level(0)

        self.last_flag_holder = [-1, -1]

        self.closest_flag_distances = [9999.0, 9999.0, 9999.0, 9999.0]

        self.resets += 1

        self.game.frame_advance(frameskip=10)

        # Step once to get the first observation
        return self.step([
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ], True)

    def step(self, combined_actions, reset=False):
        data = [
            {}, {}, {}, {}
        ]

        for player_id in range(4):
            player = data[player_id]
            actions = combined_actions[player_id]
            # actions = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

            action = 0

            left_joy_x = 0.0
            left_joy_y = 0.0
            right_joy_y = 0.0
            right_joy_x = 0.0

            if abs(actions[0]) > 0.25:
                right_joy_x = max(-1, min(1, actions[0]))
            if abs(actions[1]) > 0.25:
                # right_joy_y = max(-1, min(1, actions[1]))
                right_joy_y = 0
            if abs(actions[2]) > 0.25:
                left_joy_x = max(-1, min(1, actions[1]))
            if abs(actions[3]) > 0.25:
                left_joy_y = max(-1, min(1, actions[2]))

            if actions[3] > 0.5:
                action |= 0x8  # R1
            if actions[4] > 0.5:
                action |= 0x40  # Cross
            if actions[5] > 0.5:
                action |= 0x80  # Square

            # self.game.set_controller_input(0x2, 0.0, 0.0, 0.75, 0.75)

            if not reset:
                action = action | 0x2

            self.game.set_controller_input(player_id, action, left_joy_x, left_joy_y, right_joy_x, right_joy_y)

            player["team_id"] = self.game.get_team(player_id)
            player["other_team_id"] = 1 if player["team_id"] == 0 else 0

            player["team_has_flag"] = self.game.team_has_flag(player["team_id"])
            player["enemy_has_flag"] = self.game.team_has_flag(player["other_team_id"])

            player["health"] = self.game.get_health(player_id)

            player["other_team_health"] = self.game.get_team_health(player["other_team_id"])

            player["team_score"] = self.game.get_team_score(player["team_id"])
            player["enemy_score"] = self.game.get_team_score(player["other_team_id"])

            player["flag_distance"] = self.game.get_distance_to_team_flag(player_id, player["team_id"])
            player["enemy_flag_distance"] = self.game.get_distance_to_team_flag(player_id, player["other_team_id"])

        if not self.game.frame_advance(frameskip=4):
            # If we can't frame advance, the game has probably crashed
            terminal = True
            # reward += self.reward("crash_penalty", -1.0)

        player_info = []

        for player_id in range(4):
            player_position = self.game.get_player_position(player_id)
            player_rotation = self.game.get_player_rotation(player_id)

            player_info += [[
                np.interp(player_position.x, (-500, 500), (-1, 1)),
                np.interp(player_position.y, (-500, 500), (-1, 1)),
                np.interp(player_position.z, (-500, 500), (-1, 1)),
                np.interp(player_rotation.x, (-4, 4), (-1, 1)),
                np.interp(player_rotation.y, (-4, 4), (-1, 1)),
                np.interp(player_rotation.z, (-4, 4), (-1, 1)),
                np.interp(self.game.get_health(player_id), (0, 15), (0, 1)),
                np.interp(self.game.get_player_state(player_id), (0, 256), (0, 1)),
            ]]

        returns = []

        self.timeout -= 1

        for player_id in range(4):
            state, reward, terminal = None, 0.0, False
            if self.timeout <= 0:
                terminal = True

            player = data[player_id]
            team_id = player["team_id"]
            other_team_id = player["other_team_id"]

            teammate_id = -1
            enemy_1 = -1
            enemy_2 = -1
            if team_id == 0:
                if player_id == 0:
                    teammate_id = 1
                else:
                    teammate_id = 0
                enemy_1 = 2
                enemy_2 = 3
            else:
                if player_id == 2:
                    teammate_id = 3
                else:
                    teammate_id = 2

                enemy_1 = 0
                enemy_2 = 1

            new_flag_distance = self.game.get_distance_to_team_flag(player_id, team_id)
            new_enemy_flag_distance = self.game.get_distance_to_team_flag(player_id, other_team_id)

            # The return point for a captured flag is your own team's flag.
            # We want to reward the flag holder for going back, but not the "idle" teammate, unless the enemy team has
            #   captured the flag, then we want to reward both agents for going after the flag.
            # if team_has_flag:
            flag_holder = self.game.get_flag_holder(other_team_id)
            enemy_flag_holder = self.game.get_flag_holder(team_id)

            if (player["enemy_has_flag"] or player_id == flag_holder) and new_flag_distance < player["flag_distance"]:
                reward += self.reward("return_flag_distance", 0.01)

            if (not player["team_has_flag"] and new_enemy_flag_distance < self.closest_flag_distances[player_id] and
                    new_enemy_flag_distance < player["enemy_flag_distance"]):
                self.closest_flag_distances[player_id] = new_flag_distance
                reward += self.reward("flag_distance", 0.05)

            got_flag = self.game.team_has_flag(team_id)
            enemy_got_flag = self.game.team_has_flag(other_team_id)

            if enemy_got_flag and not player["enemy_has_flag"]:
                reward += self.reward("flag_stolen", 0.0)

            if got_flag and not player["team_has_flag"] and flag_holder == player_id and player_id != self.last_flag_holder[team_id]:
                self.timeout = 30 * 30
                self.last_flag_holder[team_id] = player_id
                self.closest_flag_distances[player_id] = 9999.0
                print(f"\nPlayer {player_id} picked up {other_team_id} flag")
                reward += self.reward("flag_captured", 2.5)
            elif got_flag and not player["team_has_flag"] and player_id == self.last_flag_holder[team_id]:
                print(f"\nPlayer {player_id} picked up dropped flag")

            team_has_flag = got_flag

            if team_has_flag:  # Constant reward for having the flag
                reward += self.reward("team_flag_reward", 0.01)

            if player["enemy_has_flag"]:  # Constant penalty for the enemy having the flag
                reward += self.reward("enemy_flag_penalty", 0.0)

            if player["health"] > self.game.get_health(player_id):
                reward += self.reward("health_penalty", -0.1)

            if self.game.get_team_health(other_team_id) < player["other_team_health"]:
                self.timeout = 30 * 30
                print(f"\nSomeone on {other_team_id} has been hit")
                reward += self.reward("damage_dealt", 0.5)

            if self.game.get_team_score(player["team_id"]) > player["team_score"]:
                self.timeout = 30 * 30
                self.last_flag_holder[team_id] = -1
                self.closest_flag_distances[player_id] = 9999.0
                print(f"\nPlayer {player_id} scored on {other_team_id}")
                reward += self.reward("team_score", 10)

            if self.game.get_team_score(player["other_team_id"]) > player["enemy_score"]:
                reward += self.reward("enemy_score", 0)

            flag_position = self.game.get_team_flag_position(player["team_id"])
            enemy_flag_position = self.game.get_team_flag_position(player["other_team_id"])

            # Normalize all state values
            state = [
                player["team_id"],

                np.interp(new_flag_distance, (0, 200), (0, 1)),
                np.interp(new_enemy_flag_distance, (0, 200), (0, 1)),
                np.interp(flag_holder, (-1, 3), (-1, 1)),
                np.interp(enemy_flag_holder, (-1, 3), (-1, 1)),

                *player_info[player_id],
                *player_info[teammate_id],
                *player_info[enemy_1],
                *player_info[enemy_2],

                np.interp(flag_position.x, (-500, 500), (-1, 1)),
                np.interp(flag_position.y, (-500, 500), (-1, 1)),
                np.interp(flag_position.z, (-500, 500), (-1, 1)),
                np.interp(enemy_flag_position.x, (-500, 500), (-1, 1)),
                np.interp(enemy_flag_position.y, (-500, 500), (-1, 1)),
                np.interp(enemy_flag_position.z, (-500, 500), (-1, 1)),
            ]

            # Check that none of the items in state are above 1 or under -1
            # for val in state:
            #     if val < -1 or val > 1:
            #         print("Aah")
            #         exit(-1)

            returns += [[torch.tensor(state, dtype=torch.bfloat16, device=self.device), reward, terminal]]

        return returns


# Just used for various tests of the environment
if __name__ == '__main__':
    import psutil

    pid = None

    for proc in reversed(list(psutil.process_iter())):
        if proc.name() == "rpcs3":
            pid = proc.pid
            break

    if pid is None:
        raise RuntimeError(f"Could not find a running Rpcs3 process.")

    env = CaptureTheFlagEnvironment(pid)
    env.start()

    base_frame  = env.game.get_current_frame_count()    # may be non-zero
    start_time  = time.time()
    steps       = 0

    try:
        last_checkpoint = None
        reward = 0.0

        while True:
            _state = env.step([
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ])

            # env.game.frame_advance(frameskip=1)

            current_frame   = env.game.get_current_frame_count()
            frames_rendered = current_frame - base_frame
            elapsed         = time.time() - start_time
            fps             = frames_rendered / elapsed if elapsed > 0 else 0.0

            print(f"\rFrame count: {current_frame} | FPS: {fps:6.2f} | State: {env.game.get_player_state(0)}", end="")
            # print(f"\rState: {env.game.get_player_state(0)} | Health: {env.game.get_health(2)}", end="")

            # print(" ".join(["{:.1f}".format(x) for x in _state]), end="\r")
            # if done:
            #     print(f"Reward: %.3f" % reward)
            #     reward = 0.0
            #     env.reset()

            steps += 1

    except KeyboardInterrupt:
        env.stop()
