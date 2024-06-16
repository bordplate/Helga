import time
import numpy as np
import torch

from Game.Game import Game, Vector3
from Game.RC1Game import RC1Game
from .RatchetEnvironment import RatchetEnvironment


class FitnessCourseEnvironment(RatchetEnvironment):
    def __init__(self, pid, eval_mode=False, device="cpu"):
        super().__init__(device=device)

        self.game = RC1Game(pid=pid)

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
            Vector3(269.9619445800781, 143.47598266601562, 50.0),
            Vector3(269.9619445800781, 143.47598266601562, 50.0),
            Vector3(298.1654052734375, 143.35801696777344, 48.0625),
            Vector3(301.29742431640625, 178.26695251464844, 45.689918518066406),
            Vector3(265.7449035644531, 195.6828155517578, 46.0),
            Vector3(228.53050231933594, 203.75091552734375, 46.25),
            Vector3(233.90478515625, 239.93067932128906, 36.0),
            Vector3(293.2232971191406, 244.1248321533203, 34.5),
            Vector3(320.4754943847656, 240.56504821777344, 42.0),
            Vector3(339.72320556640625, 243.81422424316406, 60.0),
            Vector3(329.45880126953125, 282.32977294921875, 47.75),
            Vector3(308.2227783203125, 302.4566650390625, 75.015625),
            Vector3(261.9341125488281, 349.85101318359375, 75.203125),
            Vector3(247.73629760742188, 374.39923095703125, 85.03430938720703),
            Vector3(262.2768249511719, 375.6791076660156, 87.02777099609375),
            Vector3(280.4011535644531, 392.7554016113281, 88.98082733154297),
            Vector3(247.2918243408203, 375.9893493652344, 85.04957580566406),
            Vector3(264.1728515625, 345.2733154296875, 75.0),
            Vector3(308.86572265625, 299.5409240722656, 75.015625),
            Vector3(326.81341552734375, 279.9330749511719, 47.75),
            Vector3(307.574951171875, 244.40528869628906, 34.0),
            Vector3(239.2870635986328, 244.0665740966797, 36.0),
            Vector3(228.46156311035156, 195.80406188964844, 46.0),
            Vector3(278.11517333984375, 195.728759765625, 45.89271545410156),
            Vector3(302.5223693847656, 146.34388732910156, 48.0625),
        ]

        self.more_checkpoints = [

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

    def reset(self):
        # Check that we've landed on the right level yet
        if self.eval_mode:
            self.game.set_should_render(True)

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
        # if np.random.rand() < (0.7 if not self.eval_mode else 0.25):
        if np.random.rand() < 0.7:
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

        self.game.set_checkpoint_position(self.checkpoints[self.checkpoint])

        # Clear game inputs so we don't keep moving from the last episode
        self.game.set_controller_input(0, 0.0, 0.0, 0.0, 0.0)

        self.distance_from_checkpoint_per_step = []

        # Step once to get the first observation
        return self.step([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

    def step(self, actions):
        state, reward, terminal = None, 0.0, False

        self.timer += 1

        action = 0

        left_joy_x = 0.0
        left_joy_y = 0.0
        right_joy_y = 0.0
        right_joy_x = 0.0

        if abs(actions[0]) > 0.25:
            right_joy_x = max(-1, min(1, actions[0]))
        if abs(actions[1]) > 0.25:
            right_joy_y = max(-1, min(1, actions[1]))
        if abs(actions[2]) > 0.25:
            left_joy_x = max(-1, min(1, actions[2]))
        if abs(actions[3]) > 0.25:
            left_joy_y = max(-1, min(1, actions[3]))

        if actions[4] > 0.5:
            action |= 0x8   # R1
        if actions[5] > 0.5:
            action |= 0x40  # Cross
        if actions[6] > 0.5:
            action |= 0x80  # Square

        death_count = self.game.get_death_count()

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
        if not self.game.frame_advance(frameskip=2):
            # If we can't frame advance, the game has probably crashed
            terminal = True
            reward += self.reward("crash_penalty", -1.0)

        if death_count != self.game.get_death_count():
            terminal = True
            reward += self.reward("death_penalty", -1.5)

        # Get updated player info
        looking_at_checkpoint = self.game.get_camera_position().is_looking_at(self.game.get_camera_rotation(), checkpoint_position)
        position = self.game.get_player_position()
        player_rotation = self.game.get_player_rotation()
        distance_from_ground = self.game.get_distance_from_ground()
        speed = self.game.get_player_speed()
        player_state = self.game.get_player_state()
        distance_delta = position.distance_to_2d(pre_player_position)

        # Give reward for looking towards the checkpoint
        # if looking_at_checkpoint:
        #     reward += self.reward("looking_at_checkpoint", 0.2)

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
        # if position.x < self.bounds[0][0] or position.y > self.bounds[0][1] or \
        #         position.x > self.bounds[1][0] or position.y < self.bounds[1][1]:
        #     terminal = True
        #     self.reward_counters['rewards/void_penalty'] += 1
        #     reward -= 1.0

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
            if dist < 10 and dist > 0.01:
                reward += self.reward("distance_from_checkpoint_reward", (pre_distance_from_checkpoint - distance_from_checkpoint) * 0.5)

                self.frames_moving_away_from_checkpoint = 0
        # elif distance_from_checkpoint < pre_distance_from_checkpoint:
        #     dist = pre_distance_from_checkpoint - distance_from_checkpoint
        #     if dist < 2 and dist > 0.01:
        #         self.reward_counters['rewards/distance_from_checkpoint_reward'] += (pre_distance_from_checkpoint - distance_from_checkpoint) * 0.1
        #         reward += (pre_distance_from_checkpoint - distance_from_checkpoint) * 0.1

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

            reward += self.reward("reached_checkpoint_reward", 5.0)

            checkpoint_position = self.checkpoints[self.checkpoint]
            distance_from_checkpoint = self.game.get_player_position().distance_to(checkpoint_position)

            self.closest_distance_to_checkpoint = distance_from_checkpoint

            self.game.set_checkpoint_position(self.checkpoints[self.checkpoint])

        if distance_from_checkpoint > self.closest_distance_to_checkpoint + 10:
            reward += self.reward("moving_away_from_checkpoint_penalty", -0.05)

        # Various speed related rewards and penalties

        # Check that the agent hasn't stopped progressing
        if self.time_since_last_checkpoint > 30 * 30:  # 30 in-game seconds
            terminal = True
            reward += self.reward("timeout_penalty", -1.0)

        # Discourage standing still
        # if distance_delta <= 0.01 and self.timer > 30 * 5:
        #     if self.stalled_timer > 30:
        #         self.reward_counters['rewards/stall_penalty'] += 0.05
        #         reward -= 0.1
        #
        #     self.stalled_timer += 1
        # else:
        #     self.stalled_timer = 0

        # Check that agent is facing a checkpoint by calculating angle between player and checkpoint
        angle = np.arctan2(checkpoint_position.y - position.y, checkpoint_position.x - position.x) - player_rotation.z

        # Give reward for facing the checkpoint
        # if abs(angle) < 0.1:
        #     self.reward_counters['rewards/facing_checkpoint_reward'] += 0.02
        #     reward += 0.02

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

        self.distance_from_checkpoint_per_step.append(distance_from_checkpoint)

        # Penalize various collisions
        camera_pos = self.game.get_camera_position()
        camera_rot = self.game.get_camera_rotation()

        camera_to_checkpoint_angle = np.arctan2(checkpoint_position.y - camera_pos.y, checkpoint_position.x - camera_pos.x) - camera_rot.z

        # Build observation state
        state = [
            # Position
            np.interp(position.x, (0, 500), (-1, 1)),  # 0
            np.interp(position.y, (0, 500), (-1, 1)),  # 1
            np.interp(position.z, (-150, 150), (-1, 1)),  # 2
            np.interp(player_rotation.z, (-4, 4), (-1, 1)),  # 3

            np.interp(camera_pos.x, (0, 500), (-1, 1)),  # 4
            np.interp(camera_pos.y, (0, 500), (-1, 1)),  # 5
            np.interp(camera_pos.z, (-150, 150), (-1, 1)),  # 6
            np.interp(camera_rot.z, (-4, 4), (-1, 1)),  # 7

            np.interp(camera_to_checkpoint_angle, (-4, 4), (-1, 1)),  # 8

            # Checkpoints
            np.interp(checkpoint_position.x, (0, 500), (-1, 1)),  # 8
            np.interp(checkpoint_position.y, (0, 500), (-1, 1)),  # 9
            np.interp(checkpoint_position.z, (-150, 150), (-1, 1)),  # 10
            check_diff_x,  # 11
            check_diff_y,  # 12
            check_diff_z,  # 13

            np.interp(pre_distance_from_checkpoint, (0, 500), (-1, 1)),  # 14
            np.interp(distance_from_checkpoint, (0, 500), (-1, 1)),  # 15
            np.interp(self.closest_distance_to_checkpoint, (0, 500), (-1, 1)),  # 16

            # Player data
            np.interp(distance_from_ground, (-64, 64), (-1, 1)),  # 17
            np.interp(speed, (0, 2), (-1, 1)),  # 18
            np.interp(player_state, (0, 255), (-1, 1)),  # 19

            np.interp(self.timer, (0, 1000 * 1000), (-1, 1)),  # 20

            # Joystick
            self.game.joystick_l_x,  # 21
            self.game.joystick_l_y,  # 22
            self.game.joystick_r_x,  # 23
            self.game.joystick_r_y,  # 24

            # Face buttons
            np.interp(action, (0, 0xFFFFFFFF), (-1, 1)),  # 25

            # Collision data
            *self.game.get_collisions()  # 64 collisions + 64 classes + 64*3 normals
        ]

        # state = [
        #     # Position
        #     position.x,
        #     position.y,
        #     position.z,
        #     player_rotation.z,
        #
        #     camera_pos.x,
        #     camera_pos.y,
        #     camera_pos.z,
        #     camera_rot.z,
        #
        #     camera_to_checkpoint_angle,
        #
        #     # Checkpoints
        #     checkpoint_position.x,
        #     checkpoint_position.y,
        #     checkpoint_position.z,
        #     check_diff_x,  # 11
        #     check_diff_y,  # 12
        #     check_diff_z,  # 13
        #
        #     pre_distance_from_checkpoint,
        #     distance_from_checkpoint,
        #     self.closest_distance_to_checkpoint,
        #
        #     # Player data
        #     distance_from_ground,
        #     speed,
        #     player_state,
        #
        #     self.timer,
        #
        #     # Joystick
        #     self.game.joystick_l_x,  # 21
        #     self.game.joystick_l_y,  # 22
        #     self.game.joystick_r_x,  # 23
        #     self.game.joystick_r_y,  # 24
        #
        #     # Collision data
        #     *self.game.get_collisions()  # 64 collisions + 64 classes
        # ]

        # Does the reward actually need to be normalized?
        if reward > 20 or reward < -20:
            #print(f"Danger! Reward out of bounds: {reward}")

            # Clamp
            reward = max(-20, min(20, reward))

        # Iterate through the state to check that none of the values are above 1 or below -1
        # for s, state_value in enumerate(state):
        #     if state_value > 1.0 or state_value < -1.0:
        #         print(f"Danger! State out of bounds: {s}. Value: {state_value}")
        #         exit(0)

        return torch.tensor(state, dtype=torch.float32, device=self.device), reward, terminal


# Just used for various tests of the environment
if __name__ == '__main__':
    try:
        env = FitnessCourseEnvironment(process_name="rpcs3.exe", eval_mode=False, device="cpu")
        env.start()

        while True:
            steps = 0
            next_frame_time = time.time()

            env.reset()

            last_checkpoint = None

            total_reward = 0

            while True:
                # current_time = time.time()

                _, reward, terminal = env.step([0, 0, 0, 0, 0, 0, 0])
                # env.game.frame_advance()

                # total_reward += reward

                # if terminal:
                #     print(f"\nTotal reward: {total_reward}")
                #     break

                # current_position = env.game.get_player_position()
                # print(f"\r[{current_position.x}, {current_position.y}, {current_position.z}, {env.game.get_player_rotation().z}],", end="")
                #
                # # Schedule next frame
                # next_frame_time += 1 / 60  # Schedule for the next 1/60th second
                # time.sleep(0.016)

                steps += 1
    except KeyboardInterrupt:
        env.stop()
