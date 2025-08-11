import ctypes
import time
import numpy as np
import torch

from Game.Game import Game, Vector3
from Game.RC1Game import RC1Game
from .RatchetEnvironment import RatchetEnvironment


class EudoraEnvironment(RatchetEnvironment):
    def __init__(self, pid, eval_mode=False, device="cpu"):
        super().__init__(device=device)

        self.game = RC1Game(pid=pid)
        self.total_steps = 0

        self.checkpoints_template = [
            RC1Game.BoltCrank(45),
            Vector3(221.25205993652344, 187.8348846435547, 53.5),
            Vector3(190.04981994628906, 189.90719604492188, 53.0),
            RC1Game.BoltCrank(56),
            Vector3(198.77659606933594, 226.0849609375, 53.06841278076172),
            Vector3(219.53329467773438, 226.20079040527344, 53.06841278076172),
            Vector3(259.1209716796875, 230.78616333007812, 56.0),
            Vector3(259.2367858886719, 250.417724609375, 60.0),
            Vector3(198.00863647460938, 253.7216796875, 61.0),
            Vector3(166.51422119140625, 254.73619079589844, 60.0),
            RC1Game.BoltCrank(160),
            Vector3(181.47677612304688, 293.73114013671875, 73.48265838623047),
            Vector3(209.99240112304688, 300.0426940917969, 64.0),
            Vector3(265.45831298828125, 309.39453125, 65.0),
            Vector3(315.7657775878906, 270.19781494140625, 62.25),
            Vector3(325.0148010253906, 245.27354431152344, 62.25),
            Vector3(291.9438171386719, 210.7085418701172, 56.0),
            Vector3(273.8605651855469, 187.17071533203125, 54.0),
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
        self.highest_grounded_z = 0.0
        self.start_z = 0.0

        self.checkpoint = 0
        self.n_checkpoints = 0

        self.distance_from_checkpoint_per_step = []
        self.time_since_last_checkpoint = 0
        self.closest_distance_to_checkpoint = 99999
        self.closest_distance_to_checkpoint_2d = 999999

        self.skid_address = 0
        self.jump_debounce = 0

        self.stalled_timer = 0

        self.frames_moving_away_from_checkpoint = 0

    def reset(self):
        # Check that we've landed on the right level yet
        if self.eval_mode:
            self.game.set_should_render(True)

        while self.game.get_current_level() != 4:
            print("Waiting for Eudora level change...")

            if self.game.must_restart:
                self.game.restart()

            time.sleep(1)

        self.game.reset_level_flags(4)

        self.game.zero_fill(0x96c00e, 2)

        while self.game.get_current_frame_count() < 850:
            self.total_steps += 1
            print("Waiting")
            self.game.frame_advance(100)

        # while self.game.get_death_count() <= 0:
        #     self.game.set_player_position(Vector3(0, 0, -10000))
        #     self.game.frame_advance(4)

        death_count = self.game.get_death_count()
        while self.game.get_death_count() <= death_count:
            self.game.set_player_position(Vector3(0, 0, -10000))
            self.game.frame_advance(4)

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
        self.closest_distance_to_checkpoint_2d = 999999

        # Create checkpoints from self.checkpoint_template and jitter them slightly to make them harder to memorize
        self.checkpoints = []
        for checkpoint in self.checkpoints_template:
            if checkpoint is Vector3:
                self.checkpoints.append(Vector3(checkpoint.x + np.random.uniform(-1, 1),
                                                checkpoint.y + np.random.uniform(-1, 1),
                                                checkpoint.z))
            else:
                self.checkpoints.append(checkpoint)

        # Set player back to full health, just in case
        self.game.set_nanotech(4)
        self.nanotech = 4

        spawn_position = Vector3(220.25, 162.03999329, 56)

        # 70% chance to spawn at random checkpoint, 30% in evaluation mode
        if np.random.rand() < (1 if not self.eval_mode else 1):
            checkpoint = np.random.randint(0, len(self.checkpoints_template))
            spawn_position = self.checkpoints_template[checkpoint]

            if type(spawn_position) is RC1Game.BoltCrank:
                spawn_position.start(self.game)

                spawn_position = Vector3(
                    spawn_position.moby.position.x,
                    spawn_position.moby.position.y,
                    spawn_position.moby.position.z + 1
                )

            self.checkpoint = (checkpoint + 1) % len(self.checkpoints)

        # checkpoint = 0
        # spawn_position = self.checkpoints_template[checkpoint]
        # self.checkpoint = (checkpoint + 1) % len(self.checkpoints)

        self.start_z = spawn_position.z
        self.highest_grounded_z = spawn_position.z
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

        # Reset game and player state to start a new episode
        self.game.set_player_state(0)

        self.distance_from_checkpoint_per_step = []

        # Step once to get the first observation
        return self.step([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

    def step(self, actions):
        if self.game.must_restart:
            return None, 0.0, True

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

        checkpoint = self.checkpoints[self.checkpoint]
        checkpoint_position = checkpoint

        if type(checkpoint) is RC1Game.BoltCrank:
            checkpoint_position = Vector3(
                checkpoint.moby.position.x,
                checkpoint.moby.position.y,
                checkpoint.moby.position.z
            )

        # checkpoint_position = self.checkpoints[self.checkpoint]

        # Check that agent is moving towards the next checkpoint
        pre_distance_from_checkpoint = pre_player_position.distance_to(checkpoint_position)
        pre_distance_from_checkpoint_2d = pre_player_position.distance_to_2d(checkpoint_position)

        pre_check_delta_x, pre_check_delta_y, pre_check_delta_z = (
            checkpoint_position.x - pre_player_position.x,
            checkpoint_position.y - pre_player_position.y,
            checkpoint_position.z - pre_player_position.z
        )

        pre_angle = np.arctan2(checkpoint_position.y - pre_player_position.y, checkpoint_position.x - pre_player_position.x) - pre_player_rotation.z

        pre_nanotech = self.game.get_nanotech()

        # Frame advance the game
        if not self.game.frame_advance(frameskip=2) or self.game.must_restart:
            # If we can't frame advance, the game has probably crashed
            terminal = True
            reward += self.reward("crash_penalty", -1.0)

        if death_count != self.game.get_death_count():
            terminal = True
            reward += self.reward("death_penalty", -10)

        # Get updated player info
        looking_at_checkpoint = self.game.get_camera_position().is_looking_at(self.game.get_camera_rotation(), checkpoint_position)
        position = self.game.get_player_position()
        player_rotation = self.game.get_player_rotation()
        distance_from_ground = self.game.get_distance_from_ground()
        speed = self.game.get_player_speed()
        player_state = self.game.get_player_state()
        distance_delta = position.distance_to_2d(pre_player_position)
        nanotech = self.game.get_nanotech()

        if nanotech < self.nanotech:
            reward += self.reward("nanotech_penalty", -1)

        if nanotech > self.nanotech and not terminal:
            reward += self.reward("nanotech_reward", 1)

        self.nanotech = nanotech

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
        distance_from_checkpoint_2d = pre_player_position.distance_to_2d(checkpoint_position)

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
                reward += self.reward("distance_from_checkpoint_reward", (pre_distance_from_checkpoint - distance_from_checkpoint))

                self.frames_moving_away_from_checkpoint = 0
        elif distance_from_checkpoint < pre_distance_from_checkpoint:
            dist = pre_distance_from_checkpoint - distance_from_checkpoint
            if dist < 10 and dist > 0.01:
                reward += self.reward("distance_from_checkpoint_reward", 0.01)
        # elif distance_from_checkpoint < pre_distance_from_checkpoint:
        #     dist = pre_distance_from_checkpoint - distance_from_checkpoint
        #     if dist < 2 and dist > 0.01:
        #         self.reward_counters['rewards/distance_from_checkpoint_reward'] += (pre_distance_from_checkpoint - distance_from_checkpoint) * 0.1
        #         reward += (pre_distance_from_checkpoint - distance_from_checkpoint) * 0.1

        # Reward agent for finding a higher spot where it is also grounded. To reward intermediate progress towards
        #   the checkpoint, we reward the agent for finding a higher spot where it is grounded.
        if checkpoint_position.z > position.z > self.highest_grounded_z and distance_from_ground < 0.01:
            # We reward a total maximum of 5 points between self.start_z and the checkpoint's z position for each checkpoint
            # to encourage the agent to find higher ground.
            # If the agent has found higher ground to get 2 points, there is a maximum of 3 points left in this checkpoint.
            height_diff = position.z - self.highest_grounded_z
            r = (height_diff / (checkpoint_position.z - self.start_z)) * 5.0

            reward += self.reward("higher_grounded_reward", r)

            self.highest_grounded_z = position.z

        if distance_from_checkpoint < self.closest_distance_to_checkpoint:
            self.closest_distance_to_checkpoint = distance_from_checkpoint

        if distance_from_checkpoint_2d < self.closest_distance_to_checkpoint_2d:
            self.closest_distance_to_checkpoint_2d = distance_from_checkpoint_2d

        self.time_since_last_checkpoint += 1

        # If agent is within 4 units of checkpoint, go to next checkpoint or loop around
        checkpoint_done = False

        if type(checkpoint) is Vector3:
            checkpoint_done = distance_from_checkpoint < 5

        if type(checkpoint) is RC1Game.BoltCrank:
            rew, checkpoint_done = checkpoint.step()
            reward += self.reward("crank_reward", rew)

        if checkpoint_done:
            self.time_since_last_checkpoint = 0

            self.checkpoint += 1
            self.n_checkpoints += 1
            if self.checkpoint >= len(self.checkpoints):
                self.checkpoint = 0

            reward += self.reward("reached_checkpoint_reward", 10.0)

            checkpoint = self.checkpoints[self.checkpoint]
            checkpoint_position = checkpoint

            if type(checkpoint) is RC1Game.BoltCrank:
                if not checkpoint.start(self.game):
                    self.checkpoint += 1
                    checkpoint = self.checkpoints[self.checkpoint]
                    checkpoint_position = checkpoint
                else:
                    checkpoint_position = Vector3(
                        checkpoint.moby.position.x,
                        checkpoint.moby.position.y,
                        checkpoint.moby.position.z
                    )

            distance_from_checkpoint = self.game.get_player_position().distance_to(checkpoint_position)

            self.closest_distance_to_checkpoint = distance_from_checkpoint

            self.highest_grounded_z = position.z - distance_from_ground
            self.start_z = position.z - distance_from_ground

            self.game.set_checkpoint_position(self.checkpoints[self.checkpoint])

        if distance_from_checkpoint_2d > self.closest_distance_to_checkpoint_2d + 10 and pre_distance_from_checkpoint_2d < distance_from_checkpoint_2d:
            reward += self.reward("moving_away_from_checkpoint_penalty", -0.1)

        # Various speed related rewards and penalties

        # Check that the agent hasn't stopped progressing
        if self.time_since_last_checkpoint > 30 * 30:  # 30 in-game seconds
            terminal = True
            reward += self.reward("timeout_penalty", -10.0)

        # Discourage standing still
        # if distance_delta <= 0.01 and self.timer > 30 * 5:
        #     if self.stalled_timer > 30:
        #         self.reward_counters['rewards/stall_penalty'] += 0.05
        #         reward -= 0.1
        #
        
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

        self.distance_from_ground = distance_from_ground

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
            np.interp((30 * 30 - self.time_since_last_checkpoint) / 30, (0, 30), (-1, 1)),
            np.interp(nanotech, (0, 8), (-1, 1)),

            np.interp(self.timer, (0, 1000 * 1000), (-1, 1)),  # 20

            # Joystick
            self.game.joystick_l_x,  # 21
            self.game.joystick_l_y,  # 22
            self.game.joystick_r_x,  # 23
            self.game.joystick_r_y,  # 24

            # Face buttons
            np.interp(action, (0, 0xFFFFFFFF), (-1, 1)),  # 25

            # Collision data
            *self.game.get_collisions_with_normals()  # 64 collisions + 64 classes + 64*3 normals
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
        #if reward > 20 or reward < -20:
        #print(f"Danger! Reward out of bounds: {reward}")

        # Clamp
        #reward = max(-20, min(20, reward))

        self.total_steps += 1

        # Iterate through the state to check that none of the values are above 1 or below -1
        # for s, state_value in enumerate(state):
        #     if state_value > 1.0 or state_value < -1.0:
        #         print(f"Danger! State out of bounds: {s}. Value: {state_value}")
        #         exit(0)

        return torch.tensor(state, dtype=torch.bfloat16, device=self.device), reward, terminal


# Just used for various tests of the environment
if __name__ == '__main__':
    try:
        # Find the rpcs3 process
        import psutil

        for proc in reversed(list(psutil.process_iter())):
            if proc.name() == "rpcs3":
                pid = proc.pid
                break

        env = EudoraEnvironment(pid=pid, eval_mode=False, device="cpu")
        env.start()

        env.game.set_should_render(True)

        save_data = env.game.get_save_data()

        while True:
            steps = 0
            next_frame_time = time.time()

            env.reset()

            last_checkpoint = None

            total_reward = 0
            print("")

            while True:
                # current_time = time.time()
                #
                # print("\r", end="")
                # for moby in env.game.find_mobys_by_oclass(280):
                #     class BoltVars(ctypes.Structure):
                #         _fields_ = [("x", ctypes.c_float),
                #                     ("y", ctypes.c_int),
                #                     ("z", ctypes.c_int)]
                #
                #     # if moby.UID == 45:
                #     #     moby.populate_pvars_with_ctype(env.game.process, BoltVars)
                #     #     print(f"\r45 z_rot: {moby.vars.x}", end="")
                #     #
                #     #     if moby.state == 4:
                #     #         print("OMG IT'S ALREADY CRANKED!")
                #
                #     print(f"{moby.UID}: {int.from_bytes(moby.state)}, ", end="")

                # for bridge_part in env.game.find_mobys_by_oclass(0x1b0):
                #     print(f"{bridge_part.UID}: {int.from_bytes(bridge_part.state)}, ", end="")

                # if env.game.get_player_state() == 59:
                #     print(f"{env.game.process.read_int(0x951540)}", end="")

                _, reward, terminal = env.step([0, 0, 0, 0, 0, 0, 0])
                # env.game.frame_advance(2)

                total_reward += reward

                print(f"\rTotal reward: {total_reward}", end="")

                if terminal:
                    break

                # env.game.frame_advance()

                # total_reward += reward

                # if terminal:
                #     print(f"\nTotal reward: {total_reward}")
                #     break

                # input()
                # current_position = env.game.get_player_position()
                # print(f"\rVector3({current_position.x}, {current_position.y}, {current_position.z}),", end="")
                #
                # # Schedule next frame
                # next_frame_time += 1 / 60  # Schedule for the next 1/60th second
                # time.sleep(0.016)

                steps += 1
    except KeyboardInterrupt:
        env.stop()
