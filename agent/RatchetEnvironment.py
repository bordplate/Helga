import time
import numpy as np

from Game import Vector3
from Game import Game


class RatchetEnvironment:
    def __init__(self, process_name="rpcs3.exe", eval_mode=False):
        self.game = Game(process_name=process_name)

        self.checkpoints_template = [
            Vector3(226, 143, 49.5),
            Vector3(213, 141, 57),
            Vector3(198, 140, 64),
            Vector3(198, 147, 77.5),
            Vector3(140, 161, 50.5),
            Vector3(114, 200, 63),
            Vector3(142, 197, 70),
            Vector3(130, 189, 89),
            Vector3(117, 86, 66),
            Vector3(136, 114, 70),
            Vector3(201, 128, 50),
        ]

        # [x, y] bounds of the level
        self.bounds = [
            [91, 234],
            [275, 40],
        ]

        self.eval_mode = eval_mode

        self.checkpoints = []

        self.distance = 0.0
        self.distance_traveled = 0.0
        self.timer = 0
        self.height_lost = 0.0

        self.checkpoint = 0
        self.n_checkpoints = 0

        self.distance_from_checkpoint_per_step = []
        self.time_since_last_checkpoint = 0
        self.closest_distance_to_checkpoint = 99999

        self.skid_address = 0
        self.jump_debounce = 0

        self.stalled_timer = 0

        self.frames_moving_away_from_checkpoint = 0

        self.reward_counters = {}

    def start(self):
        process_opened = self.game.open_process()
        while not process_opened:
            print("Waiting for process to open...")
            time.sleep(1)
            process_opened = self.game.open_process()
            
    def stop(self):
        self.game.close_process()

    def reset(self):
        # Check that we've landed on the right level yet
        while self.game.get_current_level() != 3:
            print("Waiting for Kerwan level change...")

            if self.game.must_restart:
                self.game.restart()

            time.sleep(1)

        # Reset variables that we use to keep track of the episode state and statistics
        self.stalled_timer = 0
        self.distance = 0.0
        self.distance_traveled = 0.0
        self.timer = 0
        self.height_lost = 0.0
        self.checkpoint = 0
        self.n_checkpoints = 0
        self.jump_debounce = 0
        self.frames_moving_away_from_checkpoint = 0

        self.time_since_last_checkpoint = 0
        self.closest_distance_to_checkpoint = 999999

        self.reward_counters = {
            'rewards/timeout_penalty': 0,
            'rewards/jump_penalty': 0,
            'rewards/distance_from_checkpoint_reward': 0,
            'rewards/distance_from_checkpoint_penalty': 0,
            'rewards/speed_reward': 0,
            'rewards/distance_traveled_reward': 0,
            'rewards/height_loss_penalty': 0,
            'rewards/death_penalty': 0,
            'rewards/crash_penalty': 0,
            'rewards/wall_crash_penalty': 0,
            'rewards/tnt_crash_penalty': 0,
            'rewards/void_penalty': 0,
            'rewards/stall_penalty': 0,
            'rewards/reached_checkpoint_reward': 0,
            'rewards/facing_checkpoint_reward': 0,
            'rewards/rotating_to_face_checkpoint_reward': 0,
        }

        # Create checkpoints from self.checkpoint_template and jitter them slightly to make them harder to memorize
        self.checkpoints = []
        for checkpoint in self.checkpoints_template:
            self.checkpoints.append(Vector3(checkpoint.x + np.random.uniform(-1, 1),
                                            checkpoint.y + np.random.uniform(-1, 1),
                                            checkpoint.z))

        # Reset game and player state to start a new episode
        # self.game.set_player_state(0)

        # Set player back to full health, just in case
        self.game.set_nanotech(4)

        spawn_position = Vector3(259, 143, 49.5)

        # 70% chance to spawn at random checkpoint, 30% in evaluation mode
        if np.random.rand() < (0.7 if not self.eval_mode else 0.3):
            checkpoint = np.random.randint(0, len(self.checkpoints))
            spawn_position = self.checkpoints_template[checkpoint]
            self.checkpoint = (checkpoint + 1) % len(self.checkpoints)

        self.game.set_player_position(spawn_position)
        self.game.set_player_rotation(Vector3(0, 0, -2.5))
        self.game.set_player_speed(0)
        self.game.set_item_unlocked(2)

        self.game.joystick_l_x = 0.0
        self.game.joystick_l_y = 0.0
        self.game.joystick_r_x = 0.0
        self.game.joystick_r_y = 0.0

        # Clear game inputs so we don't keep moving from the last episode
        self.game.set_controller_input(0, 0.0, 0.0, 0.0, 0.0)

        # while self.game.get_player_state() != 107:
        #     self.game.start_hoverboard_race()
        #
        #     self.game.frame_advance()

        self.distance_from_checkpoint_per_step = []

        # Step once to get the first observation
        return self.step([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

    def step(self, actions):
        state, reward, terminal = None, 0.0, False

        self.timer += 1

        # if self.timer > 30 * 60 * 5:  # 5 minutes
        #     terminal = True
        #     self.reward_counters['rewards/timeout_penalty'] += 1
        #     reward -= 1.0
        #     print("Timeout!")

        # actions_mapping = [
        #     0x0,     # No action
        #     0x8,     # R1
        #     0x40,    # Cross
        #     0x80,    # Square
        #     0x1000,  # Up
        #     0x2000,  # Right
        #     0x4000,  # Down
        #     0x8000,  # Left
        #
        #     0x8 | 0x40,  # R1 + Cross
        #
        #     0x40 | 0x1000,  # Cross + Up
        #     0x40 | 0x2000,  # Cross + Right
        #     0x40 | 0x4000,  # Cross + Down
        #     0x40 | 0x8000,  # Cross + Left
        # ]

        action = 0

        left_joy_x = 0.0
        left_joy_y = 0.0
        right_joy_y = 0.0
        right_joy_x = 0.0

        if actions[0] > 0.1 or actions[0] < -0.1:
            right_joy_x = max(-1, min(1, actions[0]))
        if actions[1] > 0.1 or actions[1] < -0.1:
            right_joy_y = max(-1, min(1, actions[1]))
        if actions[2] > 0.1 or actions[2] < -0.1:
            left_joy_x = max(-1, min(1, actions[2]))
        if actions[3] > 0.1 or actions[3] < -0.1:
            left_joy_y = max(-1, min(1, actions[3]))

        # if actions[0] > 0.5:
        #     action |= 0x1000  # Up
        # if actions[1] > 0.5:
        #     action |= 0x2000  # Right
        # if actions[2] > 0.5:
        #     action |= 0x4000  # Down
        # if actions[3] > 0.5:
        #     action |= 0x8000  # Left
        if actions[4] > 0.5:
            action |= 0x8   # R1
        if actions[5] > 0.5:
            action |= 0x40  # Cross
        if actions[6] > 0.5:
            action |= 0x80  # Square

        death_count = self.game.get_death_count()

        # Discourage excessive jumping because it's annoying
        # if self.jump_debounce > 0:
        #     self.jump_debounce -= 1
        #
        # if action & 0x40:
        #     if self.jump_debounce > 0:
        #         self.reward_counters['rewards/jump_penalty'] += self.jump_debounce * 0.00001
        #         reward -= self.jump_debounce * 0.00001
        #     self.jump_debounce += 5

        # Communicate game inputs with game
        self.game.set_controller_input(action, left_joy_x, left_joy_y, right_joy_x, right_joy_y)

        # Get current player position and distance to checkpoint before advancing to next frame so we can calculate
        #   how much the agent has moved towards the goal given the input it provided.
        pre_player_position = self.game.get_player_position()
        pre_player_rotation = self.game.get_player_rotation()

        checkpoint_position = self.checkpoints[self.checkpoint]

        # Check that agent is moving towards the next checkpoint
        pre_distance_from_checkpoint = pre_player_position.distance_to(checkpoint_position)

        pre_check_delta_x, pre_check_delta_y, pre_check_delta_z = (
            checkpoint_position.x - pre_player_position.x,
            checkpoint_position.y - pre_player_position.y,
            checkpoint_position.z - pre_player_position.z
        )

        pre_angle = np.arctan2(checkpoint_position.y - pre_player_position.y, checkpoint_position.x - pre_player_position.x) - pre_player_rotation.z

        # Frame advance the game
        if not self.game.frame_advance() or not self.game.frame_advance():
            # If we can't frame advance, the game has probably crashed
            reward -= 1.0
            self.reward_counters['rewards/crash_penalty'] += 1
            terminal = True

        if death_count != self.game.get_death_count():
            terminal = True
            self.reward_counters['rewards/death_penalty'] += 1.0
            reward -= 1.0

        # Get updated player info
        position = self.game.get_player_position()
        player_rotation = self.game.get_player_rotation()
        distance_from_ground = self.game.get_distance_from_ground()
        speed = self.game.get_player_speed()
        player_state = self.game.get_player_state()
        distance_delta = position.distance_to_2d(pre_player_position)

        # Calculate new distances, deltas and differences
        check_delta_x, check_delta_y, check_delta_z = (
            checkpoint_position.x - position.x,
            checkpoint_position.y - position.y,
            checkpoint_position.z - position.z
        )

        check_delta_x, check_delta_y, check_delta_z = (
            pre_check_delta_x - check_delta_x,
            pre_check_delta_y - check_delta_y,
            pre_check_delta_z - check_delta_z
        )

        check_diff_x, check_diff_y, check_diff_z = (
            max(-1, min(1, check_delta_x)), max(-1, min(1, check_delta_y)), max(-1, min(1, check_delta_z))
        )

        distance_from_checkpoint = self.game.get_player_position().distance_to(checkpoint_position)

        # Check that the player is within the bounds of the level
        if position.x < self.bounds[0][0] or position.y > self.bounds[0][1] or \
                position.x > self.bounds[1][0] or position.y < self.bounds[1][1]:
            terminal = True
            self.reward_counters['rewards/void_penalty'] += 1
            reward -= 1.0

        if distance_from_checkpoint < pre_distance_from_checkpoint:
            self.frames_moving_away_from_checkpoint -= 2

        # We want to discourage the agent from moving away from the checkpoint, but we don't want to penalize it
        #  too much for doing so because it's not always possible to move directly towards the checkpoint.
        if self.frames_moving_away_from_checkpoint < 0:
            self.frames_moving_away_from_checkpoint = 0
        elif self.frames_moving_away_from_checkpoint > 2000:
            self.frames_moving_away_from_checkpoint = 2000

        if distance_from_checkpoint < pre_distance_from_checkpoint and distance_from_checkpoint < self.closest_distance_to_checkpoint:
            dist = pre_distance_from_checkpoint - distance_from_checkpoint
            if dist < 2 and dist > 0.01:
                self.reward_counters['rewards/distance_from_checkpoint_reward'] += (pre_distance_from_checkpoint - distance_from_checkpoint) * 0.8
                reward += (pre_distance_from_checkpoint - distance_from_checkpoint) * 0.8

                self.frames_moving_away_from_checkpoint = 0
        elif distance_from_checkpoint < pre_distance_from_checkpoint:
            dist = pre_distance_from_checkpoint - distance_from_checkpoint
            if dist < 2 and dist > 0.01:
                self.reward_counters['rewards/distance_from_checkpoint_reward'] += (pre_distance_from_checkpoint - distance_from_checkpoint) * 0.1
                reward += (pre_distance_from_checkpoint - distance_from_checkpoint) * 0.1

        # else:
        #     self.frames_moving_away_from_checkpoint += 1
        #
        #     if self.frames_moving_away_from_checkpoint > 30:
        #         self.reward_counters['rewards/distance_from_checkpoint_penalty'] += (self.frames_moving_away_from_checkpoint * 0.00020)
        #         reward -= (self.frames_moving_away_from_checkpoint * 0.00020)

        if distance_from_checkpoint < self.closest_distance_to_checkpoint:
            self.closest_distance_to_checkpoint = distance_from_checkpoint

        self.time_since_last_checkpoint += 1

        # If agent is within 4 units of checkpoint, go to next checkpoint or loop around
        if distance_from_checkpoint < 4:
            self.time_since_last_checkpoint = 0

            self.checkpoint += 1
            self.n_checkpoints += 1
            if self.checkpoint >= len(self.checkpoints):
                self.checkpoint = 0

            self.reward_counters['rewards/reached_checkpoint_reward'] += 1.5 * self.n_checkpoints
            reward += 1.5 * self.n_checkpoints

            checkpoint_position = self.checkpoints[self.checkpoint]
            distance_from_checkpoint = self.game.get_player_position().distance_to(checkpoint_position)

            self.closest_distance_to_checkpoint = distance_from_checkpoint

        if distance_from_checkpoint > self.closest_distance_to_checkpoint + 10:
            # self.reward_counters['rewards/distance_from_checkpoint_penalty'] += 0.02
            reward -= 0.02

        # Various speed related rewards and penalties

        # Check that the agent hasn't stopped progressing
        if self.time_since_last_checkpoint > 30 * 30:  # 30 in-game seconds
            terminal = True
            self.reward_counters['rewards/timeout_penalty'] += 1
            reward -= 1.0

        # Discourage standing still
        if distance_delta <= 0.01 and self.timer > 30 * 5:
            if self.stalled_timer > 30 * 2:
                self.reward_counters['rewards/stall_penalty'] += 0.05
                reward -= 0.05

            self.stalled_timer += 1
        else:
            self.stalled_timer = 0

        # Check that agent is facing a checkpoint by calculating angle between player and checkpoint
        angle = np.arctan2(checkpoint_position.y - position.y, checkpoint_position.x - position.x) - player_rotation.z

        # Give reward for facing the checkpoint
        if abs(angle) < 0.1:
            self.reward_counters['rewards/facing_checkpoint_reward'] += 0.02
            reward += 0.02
        # elif abs(angle) < abs(pre_angle):
        #     # Give reward for moving to face the checkpoint
        #     self.reward_counters['rewards/rotating_to_face_checkpoint_reward'] += 0.01
        #     reward += 0.01
        #     print("Rotating")

        # We mostly collect position data to calculate distance traveled for metrics, and don't specifically use it for
        #   rewards or penalties.
        if pre_player_position is not None:
            distance = position.distance_to_2d(pre_player_position)
            if distance > 1.0:
                # Discard large distances because it probably just means the player respawned or was otherwise
                #   teleported by the game.
                distance = 0.0

            self.distance_traveled += distance

            if distance > 0:
                if pre_player_position.z > position.z:
                    self.height_lost += pre_player_position.z - position.z

            if distance_from_ground < 20:
                self.distance += distance

        # if player_state == 109 or player_state == 110 or player_state == 111:
        #     self.reward_counters['rewards/death_penalty'] += 1.0
        #     reward -= 1.0
        #     terminal = True

        self.distance_from_checkpoint_per_step.append(distance_from_checkpoint)

        # Penalize various collisions

        # Build observation state
        state = [
            # Position
            np.interp(position.x, (0, 500), (-1, 1)),  # 0
            np.interp(position.y, (0, 500), (-1, 1)),  # 1
            np.interp(position.z, (-150, 150), (-1, 1)),  # 2
            # Checkpoints
            np.interp(checkpoint_position.x, (0, 500), (-1, 1)),  # 3
            np.interp(checkpoint_position.y, (0, 500), (-1, 1)),  # 4
            np.interp(checkpoint_position.z, (-150, 150), (-1, 1)),  # 5
            check_diff_x,  # 6
            check_diff_y,  # 7
            check_diff_z,  # 8
            # Player data
            np.interp(player_rotation.z, (-20, 20), (-1, 1)),  # 9
            np.interp(distance_from_ground, (-64, 64), (-1, 1)),  # 10
            np.interp(speed, (0, 2), (-1, 1)),  # 11
            np.interp(distance_from_checkpoint, (0, 500), (-1, 1)),  # 12
            np.interp(player_state, (0, 255), (-1, 1)),  # 13

            # Joystick
            self.game.joystick_l_x,  # 14
            self.game.joystick_l_y,  # 15
            self.game.joystick_r_x,  # 16
            self.game.joystick_r_y,  # 17

            # Collision data
            *self.game.get_collisions()  # 64 collisions + 64 classes
        ]

        # Does the reward actually need to be normalized?
        if reward > 20 or reward < -20:
            #print(f"Danger! Reward out of bounds: {reward}")

            # Clamp
            reward = max(-20, min(20, reward))

        # Iterate through the state to check that none of the values are above 1 or below -1
        for s, state_value in enumerate(state):
            if state_value > 1.0 or state_value < -1.0:
                #print(f"Danger! State out of bounds: {s}. Value: {state_value}")
                exit(0)

        return np.array(state, dtype=np.float32), reward, terminal


# Just used for various tests of the environment
if __name__ == '__main__':
    try:
        env = RatchetEnvironment()
        env.start()

        while True:
            steps = 0
            next_frame_time = time.time()

            env.reset()

            last_checkpoint = None

            total_reward = 0

            while True:
                current_time = time.time()

                _, reward, terminal = env.step(0)

                total_reward += reward

                # if terminal:
                #     print(f"\nTotal reward: {total_reward}")
                #     break

                current_position = env.game.get_player_position()
                print(f"\r[{current_position.x}, {current_position.y}, {current_position.z}, {env.game.get_player_rotation().z}],", end="")

                # Schedule next frame
                next_frame_time += 1 / 60  # Schedule for the next 1/60th second
                time.sleep(0.016)

                steps += 1
    except KeyboardInterrupt:
        env.stop()
