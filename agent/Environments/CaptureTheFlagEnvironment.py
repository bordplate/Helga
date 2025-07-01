import time
import numpy as np

# from ..Game.Game import Vector3
# from ..Game.RC3Game import RC3Game
from agent.Game.Game import Vector3
from agent.Game.RC3Game import RC3Game

from RatchetEnvironment import RatchetEnvironment


class CaptureTheFlagEnvironment(RatchetEnvironment):
    def __init__(self, player=0, process_name="rpcs3.exe"):
        super().__init__()
        self.player = player
        self.game = RC3Game()

    def reset(self):
        # Step once to get the first observation
        return self.step(0)

    def step(self, actions):
        state, reward, terminal = None, 0.0, False

        # left_joy_x = 0.0
        # left_joy_y = 0.0
        # right_joy_y = 0.0
        # right_joy_x = 0.0
        #
        # if abs(actions[0]) > 0.25:
        #     right_joy_x = max(-1, min(1, actions[0]))
        # if abs(actions[1]) > 0.25:
        #     right_joy_y = max(-1, min(1, actions[1]))
        # if abs(actions[2]) > 0.25:
        #     left_joy_x = max(-1, min(1, actions[2]))
        # if abs(actions[3]) > 0.25:
        #     left_joy_y = max(-1, min(1, actions[3]))
        #
        # if actions[4] > 0.5:
        #     action |= 0x8  # R1
        # if actions[5] > 0.5:
        #     action |= 0x40  # Cross
        # if actions[6] > 0.5:
        #     action |= 0x80  # Square

        team_id = self.game.get_team()
        other_team_id = 1 if team_id == 0 else 0

        team_has_flag = self.game.team_has_flag(team_id)
        enemy_has_flag = self.game.team_has_flag(other_team_id)

        health = self.game.get_health()

        other_team_health = self.game.get_team_health(other_team_id)

        team_score = self.game.get_team_score(team_id)
        enemy_score = self.game.get_team_score(other_team_id)

        # flag_distance = self.game.get_distance_to_team_flag()

        self.game.set_controller_input(0x2, 0.35, 0.0, 0.0, -1.0)

        if not self.game.frame_advance(frameskip=2):
            # If we can't frame advance, the game has probably crashed
            terminal = True
            # reward += self.reward("crash_penalty", -1.0)

        # new_flag_distance = self.game.get_distance_to_team_flag()

        # if not team_has_flag and new_flag_distance < flag_distance:
        #   reward += self.reward("flag_reached", 0.02)

        # got_flag = self.game.team_has_flag()
        # enemy_got_flag = self.game.other_team_has_flag()

        # if enemy_got_flag and not enemy_has_flag:
        #   reward += self.reward("flag_stolen", -1.0)

        # if got_flag and not team_has_flag:
        #   reward += self.reward("flag_captured", 1.0)

        # team_has_flag = got_flag

        # if team_has_flag:  # Constant reward for having the flag
        #   reward += self.reward("team_flag_reward", 0.02)

        # if enemy_has_flag:  # Constant penalty for the enemy having the flag
        #   reward += self.reward("enemy_flag_penalty", -0.02)

        # if health > self.game.get_health():
        #   reward += self.reward("health_penalty", -0.1)

        # if self.game.get_damage_dealt() > 0:
        #   reward += self.reward("damage_dealt", 0.2)

        # if self.game.get_team_score() > team_score:
        #   reward += self.reward("team_score", 1)

        # if self.game.get_enemy_score() > enemy_score:
        #   reward += self.reward("enemy_score", -1)

        # Normalize all state values
        state = [

        ]

        return np.array(state, dtype=np.float16), reward, terminal


# Just used for various tests of the environment
if __name__ == '__main__':
    env = CaptureTheFlagEnvironment(2)
    env.start()

    base_frame  = env.game.get_current_frame_count()    # may be non-zero
    start_time  = time.time()
    steps       = 0

    try:
        last_checkpoint = None
        reward = 0.0

        while True:
            _state, r, done = env.step(np.random.choice([0, 1, 2, 3, 4, 6, 8, 10]))

            current_frame   = env.game.get_current_frame_count()
            frames_rendered = current_frame - base_frame
            elapsed         = time.time() - start_time
            fps             = frames_rendered / elapsed if elapsed > 0 else 0.0

            reward += r

            print(f"Frame count: {current_frame} | FPS: {fps:6.2f} | Reward: {reward:.2f}", end="\r")

            # print(" ".join(["{:.1f}".format(x) for x in _state]), end="\r")
            # if done:
            #     print(f"Reward: %.3f" % reward)
            #     reward = 0.0
            #     env.reset()

            steps += 1

    except KeyboardInterrupt:
        env.stop()