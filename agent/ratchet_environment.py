import time
import numpy as np

from game import Vector3
from game import Game


class RatchetEnvironment:
    def __init__(self):
        self.game = Game()

        self.checkpoints_template = [
            Vector3(213.6437530517578, 232.98785400390625, 76.0),
            Vector3(168.9180450439453, 234.16783142089844, 75.97647857666016),
            Vector3(121.3829345703125, 262.2883605957031, 76.0),
            Vector3(73.7027587890625, 293.1028747558594, 68.0),
            Vector3(71.1695, 340.3349, 68.0),
            Vector3(81.04218292236328, 370.8876953125, 68.0),
            Vector3(100.66133117675781, 422.2431640625, 68.0),
            Vector3(129.94361877441406, 464.2681579589844, 68.0),
            Vector3(167.47509765625, 447.53363037109375, 70.28053283691406),
            Vector3(199.3822784423828, 425.5357360839844, 68.0),
            Vector3(251.841064453125, 437.9490966796875, 65.90913391113281),
            Vector3(304.7581481933594, 451.1502990722656, 67.72925567626953),
            Vector3(341.3725891113281, 435.3643798828125, 71.55968475341797),
            Vector3(320.5064697265625, 393.40325927734375, 75.61055755615234),
            Vector3(282.5606384277344, 346.9862365722656, 73.0411605834961),
        ]

        self.checkpoints = []

        self.distance = 0.0
        self.distance_traveled = 0.0
        self.timer = 0
        self.height_lost = 0.0

        self.checkpoint = 0
        self.n_checkpoints = 0

        self.distance_from_checkpoint_per_step = []

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
        while self.game.get_current_level() != 5:
            print("Waiting for Rilgar level change...")

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
        self.game.set_player_state(0)

        self.game.start_hoverboard_race()

        self.game.frame_advance()

        # Set player back to full health, just in case
        self.game.set_nanotech(4)

        self.distance_from_checkpoint_per_step = []

        # Clear game inputs so we don't keep moving from the last episode
        self.game.set_controller_input(0)

        # Frame advance a couple of frames before giving control
        for _ in range(10):
            self.game.frame_advance()

        # Step once to get the first observation
        return self.step(0)

    def step(self, action):
        state, reward, terminal = None, 0.0, False

        self.timer += 1

        if self.timer > 30 * 60 * 5:  # 5 minutes
            terminal = True
            self.reward_counters['rewards/timeout_penalty'] += 1
            reward -= 1.0
            print("Timeout!")

        actions_mapping = [
            0x0,     # No action
            0x40,    # Jump
            0x2000,  # Left
            0x8000,  # Right
        ]

        # Discourage excessive jumping because it's annoying
        if self.jump_debounce > 0:
            self.jump_debounce -= 1

        if action == 3:
            if self.jump_debounce > 0:
                self.reward_counters['rewards/jump_penalty'] += self.jump_debounce * 0.00001
                reward -= self.jump_debounce * 0.00001
            self.jump_debounce += 5

        # Communicate game inputs with game
        self.game.set_controller_input(actions_mapping[action])

        # Get current player position and distance to checkpoint before advancing to next frame so we can calculate
        #   how much the agent has moved towards the goal given the input it provided.
        pre_player_position = self.game.get_player_position()
        pre_player_rotation = self.game.get_player_rotation()

        checkpoint_position = self.checkpoints[self.checkpoint]

        # Check that agent is moving towards the next checkpoint
        pre_distance_from_checkpoint = pre_player_position.distance_to_2d(checkpoint_position)

        pre_check_delta_x, pre_check_delta_y = (checkpoint_position.x - pre_player_position.x,
                                                checkpoint_position.y - pre_player_position.y)

        pre_angle = np.arctan2(checkpoint_position.y - pre_player_position.y, checkpoint_position.x - pre_player_position.x) - pre_player_rotation.z

        # Frame advance the game
        if not self.game.frame_advance() or not self.game.frame_advance():
            # If we can't frame advance, the game has probably crashed
            reward -= 1.0
            self.reward_counters['rewards/crash_penalty'] += 1
            terminal = True

        # Get updated player info
        position = self.game.get_player_position()
        player_rotation = self.game.get_player_rotation()
        distance_from_ground = self.game.get_distance_from_ground()
        speed = self.game.get_player_speed()
        player_state = self.game.get_player_state()

        # Calculate new distances, deltas and differences
        check_delta_x, check_delta_y = (checkpoint_position.x - position.x,
                                        checkpoint_position.y - position.y)

        check_delta_x, check_delta_y = pre_check_delta_x - check_delta_x, pre_check_delta_y - check_delta_y

        check_diff_x, check_diff_y = max(-1, min(1, check_delta_x)), max(-1, min(1, check_delta_y))

        distance_from_checkpoint = self.game.get_player_position().distance_to_2d(checkpoint_position)

        # We want to discourage the agent from moving away from the checkpoint, but we don't want to penalize it
        #  too much for doing so because it's not always possible to move directly towards the checkpoint.
        if self.frames_moving_away_from_checkpoint < 0:
            self.frames_moving_away_from_checkpoint = 0
        elif self.frames_moving_away_from_checkpoint > 20:
            self.frames_moving_away_from_checkpoint = 20

        if distance_from_checkpoint < pre_distance_from_checkpoint:
            if pre_distance_from_checkpoint - distance_from_checkpoint < 2:
                self.reward_counters['rewards/distance_from_checkpoint_reward'] += (pre_distance_from_checkpoint - distance_from_checkpoint) * 0.005
                reward += (pre_distance_from_checkpoint - distance_from_checkpoint) * 0.005
                self.frames_moving_away_from_checkpoint -= 0
        else:
            self.frames_moving_away_from_checkpoint += 1

            self.reward_counters['rewards/distance_from_checkpoint_penalty'] += 0.005 + (self.frames_moving_away_from_checkpoint * 0.001)
            reward -= 0.005 + (self.frames_moving_away_from_checkpoint * 0.001)

        # If agent is within 15 units of checkpoint, go to next checkpoint or loop around
        if distance_from_checkpoint < 15:
            self.checkpoint += 1
            self.n_checkpoints += 1
            if self.checkpoint >= len(self.checkpoints):
                self.checkpoint = 0

            self.reward_counters['rewards/reached_checkpoint_reward'] += 1.5 * self.n_checkpoints
            reward += 1.5 * self.n_checkpoints

            checkpoint_position = self.checkpoints[self.checkpoint]
            distance_from_checkpoint = self.game.get_player_position().distance_to_2d(checkpoint_position)

        # Various speed related rewards and penalties

        # Check that the agent is moving and isn't stuck in place
        if speed < 0.1:
            self.stalled_timer += 1
            if self.stalled_timer > 30 * 5:  # 5 in-game seconds
                terminal = True
                self.reward_counters['rewards/timeout_penalty'] += 1
                reward -= 1.0
                print("Stalled!")
        else:
            self.stalled_timer = 0

        # Encourage higher speed
        if speed > 0.25:
            self.reward_counters['rewards/speed_reward'] += (speed - 0.25) * 0.05
            reward += (speed - 0.25) * 0.05

        # Discourage stalling
        if speed < 0.22 and self.timer > 30:
            self.reward_counters['rewards/stall_penalty'] += 0.05
            reward -= 0.05

        # Check that agent is facing a checkpoint by calculating angle between player and checkpoint
        angle = np.arctan2(checkpoint_position.y - position.y, checkpoint_position.x - position.x) - player_rotation.z

        # Give reward for facing the checkpoint
        if abs(angle) < 0.1:
            self.reward_counters['rewards/facing_checkpoint_reward'] += 0.02
            reward += 0.02
        elif abs(angle) < abs(pre_angle):
            # Give reward for moving to face the checkpoint
            self.reward_counters['rewards/rotating_to_face_checkpoint_reward'] += 0.01
            reward += 0.01

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

        if player_state == 109 or player_state == 110 or player_state == 111:
            self.reward_counters['rewards/death_penalty'] += 1.0
            reward -= 1.0
            terminal = True

        self.distance_from_checkpoint_per_step.append(distance_from_checkpoint)

        coll_f, coll_u, coll_d, coll_l, coll_r, coll_cf, coll_cu, coll_cd, coll_cl, coll_cr = self.game.get_collisions()

        # Penalize various collisions

        # Wall collision
        if coll_f != -32.0 and coll_u != -32.0 and coll_f < 10.5 and coll_u < 15.5 and coll_cf == 0:
            reward -= 0.1
            self.reward_counters['rewards/wall_crash_penalty'] += 0.1

        # TNT crate collision
        if (
                (coll_f != -32.0 and coll_f < 10.5 and coll_cf == 505) or
                (coll_d != -32.0 and coll_d < 5.5 and coll_cd == 505)
        ):
            reward -= 0.2
            self.reward_counters['rewards/tnt_crash_penalty'] += 0.2

        # Going over void, e.g. outside the course in many cases
        if distance_from_ground > 31 and coll_d <= -32.0:
            self.reward_counters['rewards/void_penalty'] += 0.2
            reward -= 0.2

        # Build observation state
        state = [
            np.interp(position.x, (0, 500), (-1, 1)),  # 0
            np.interp(position.y, (0, 500), (-1, 1)),  # 1
            np.interp(position.z, (-150, 150), (-1, 1)),  # 2
            np.interp(checkpoint_position.x, (0, 500), (-1, 1)),  # 3
            np.interp(checkpoint_position.y, (0, 500), (-1, 1)),  # 4
            check_diff_x,  # 5
            check_diff_y,  # 6
            np.interp(player_rotation.z, (-20, 20), (-1, 1)),  # 7
            np.interp(distance_from_ground, (-64, 64), (-1, 1)),  # 8
            np.interp(speed, (0, 2), (-1, 1)),  # 9
            np.interp(distance_from_checkpoint, (0, 500), (-1, 1)),  # 10
            np.interp(player_state, (0, 255), (-1, 1)),  # 11
            np.interp(self.distance_traveled, (0, 100000), (-1, 1)),  # 12
            *self.game.get_collisions_normalized()  # 13-23
        ]

        # Does the reward actually need to be normalized?
        if reward > 20 or reward < -20:
            print(f"Danger! Reward out of bounds: {reward}")

            # Clamp
            reward = max(-20, min(20, reward))

        # Iterate through the state to check that none of the values are above 1 or below -1
        for s, state_value in enumerate(state):
            if state_value > 1.0 or state_value < -1.0:
                print(f"Danger! State out of bounds: {s}. Value: {state_value}")
                exit(0)

        return np.array(state, dtype=np.float32), reward, terminal


# Just used for various tests of the environment
if __name__ == '__main__':
    env = RatchetEnvironment()
    env.start()

    try:
        steps = 0
        next_frame_time = time.time()

        env.reset()

        last_checkpoint = None

        while True:
            current_time = time.time()

            input()

            current_position = env.game.get_player_position()
            print(f"[{current_position.x}, {current_position.y}, {current_position.z}, {env.game.get_player_rotation().z}],", end="")

            # Schedule next frame
            next_frame_time += 1 / 60  # Schedule for the next 1/60th second
            time.sleep(0.016)

            steps += 1
    except KeyboardInterrupt:
        env.stop()
