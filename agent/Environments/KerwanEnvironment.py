import ctypes
import time
import numpy as np
import torch

from Game.Game import Game, Vector3
from Game.RC1Game import RC1Game
from .RatchetEnvironment import RatchetEnvironment


class KerwanEnvironment(RatchetEnvironment):
    def __init__(self, pid, eval_mode=False, device="cpu"):
        super().__init__(device=device)

        self.game = RC1Game(pid=pid)
        self.total_steps = 0

        self.checkpoints_template = [
            Vector3(300, 142, 50.5),
            Vector3(302, 172, 48),
            Vector3(276, 196, 48),
            Vector3(229, 200, 48),
            Vector3(237, 243, 38),
            Vector3(295, 244, 36),
            Vector3(313, 239, 36),
            Vector3(317, 239, 42),
            Vector3(323, 239, 43),
            Vector3(336, 240, 44),
            Vector3(336, 240, 60),
            Vector3(339, 251, 61),
            Vector3(338, 271, 50),
            Vector3(313, 297, 51),
            Vector3(313, 297, 77),
            Vector3(294, 315, 77),
            Vector3(263, 346, 77),
            Vector3(236, 368, 83),
            Vector3(206, 339, 83),

            Vector3(236, 368, 83),
            Vector3(263, 346, 77),
            Vector3(294, 315, 77),
            Vector3(313, 297, 77),
            Vector3(338, 271, 50),
            Vector3(336, 240, 44),
            Vector3(323, 239, 43),
            Vector3(317, 239, 42),
            Vector3(313, 239, 36),
            Vector3(295, 244, 36),
            Vector3(237, 243, 38),
            Vector3(229, 200, 48),
            Vector3(276, 196, 48),
            Vector3(302, 172, 48),
            Vector3(300, 142, 50.5),
            Vector3(264.25, 141, 51)

        ]

        self.more_checkpoints = [

        ]

        # [x, y] bounds of the level
        self.bounds = [
            [0, 10000],
            [0, 10000],
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
        self.remaining_frames = 30 * 30  # 30 seconds
        self.distance_from_spawn = 0

        self.skid_address = 0
        self.jump_debounce = 0

        self.stalled_timer = 0

    def reset(self):
        # Check that we've landed on the right level yet
        if self.eval_mode:
            self.game.set_should_render(True)

        while self.game.get_current_level() != 3:
            print("Waiting for Kerwan level change...")

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
        self.damage_cooldown = 0
        self.remaining_frames = 30 * 30
        self.distance_from_spawn = 0

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

        self.spawn_position = Vector3(264.25, 141, 51)

        # 70% chance to spawn at random checkpoint, 30% in evaluation mode
        # if np.random.rand() < (1 if not self.eval_mode else 0):
        #     checkpoint = np.random.randint(0, len(self.checkpoints_template))
        #     spawn_position = self.checkpoints_template[checkpoint]
        #
        #     self.checkpoint = (checkpoint + 1) % len(self.checkpoints)

        self.start_z = self.spawn_position.z
        self.highest_grounded_z = self.spawn_position.z
        self.game.set_player_position(self.spawn_position)
        self.game.set_player_rotation(Vector3(0, 0, -2.5))
        self.game.set_player_speed(0)
        self.game.set_item_unlocked(2)

        self.game.joystick_l_x = 0.0
        self.game.joystick_l_y = 0.0
        self.game.joystick_r_x = 0.0
        self.game.joystick_r_y = 0.0

        # Clear game inputs so we don't keep moving from the last episode
        self.game.set_controller_input(0, 0.0, 0.0, 0.0, 0.0)

        # Reset game and player state to start a new episode
        self.game.set_player_state(0)

        self.distance_from_checkpoint_per_step = []

        self.bolts = self.game.get_bolts()

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

        # Frame advance the game
        if not self.game.frame_advance(frameskip=2) or self.game.must_restart:
            # If we can't frame advance, the game has probably crashed
            terminal = True
            reward += self.reward("crash_penalty", -1.0)

        if death_count != self.game.get_death_count():
            terminal = True
            reward += self.reward("death_penalty", -50)

        # Get updated player info
        position = self.game.get_player_position()
        player_rotation = self.game.get_player_rotation()
        distance_from_ground = self.game.get_distance_from_ground()
        speed = self.game.get_player_speed()
        player_state = self.game.get_player_state()
        nanotech = self.game.get_nanotech()
        bolts = self.game.get_bolts()
        bolts_diff = bolts - self.bolts

        distance_from_spawn = position.distance_to_2d(self.spawn_position)

        self.bolts = bolts

        if bolts_diff > 0:
            reward += self.reward("bolt_reward", bolts_diff)

            self.remaining_frames += bolts_diff
            if self.remaining_frames > 30 * 30:
                self.remaining_frames = 30 * 30

        if distance_from_spawn > self.distance_from_spawn:
            reward += self.reward("distance_from_spawn_reward", distance_from_spawn - self.distance_from_spawn)
            self.distance_from_spawn = distance_from_spawn

        if nanotech < self.nanotech:
            reward += self.reward("nanotech_penalty", -15)

        if nanotech > self.nanotech and not terminal:
            reward += self.reward("nanotech_reward", 10)

        self.nanotech = nanotech

        self.damage_cooldown -= 1

        if self.game.get_did_damage() != 0 and self.damage_cooldown <= 0:
            reward += self.reward("damaged_enemy_reward", 2.0)
            self.game.reset_did_damage()
            self.damage_cooldown = 10

        # Various speed related rewards and penalties

        # Check that the agent hasn't stopped progressing
        if self.remaining_frames <= 0:
            terminal = True
            reward += self.reward("timeout_penalty", -50.0)

        self.remaining_frames -= 1

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

        # Penalize various collisions
        camera_pos = self.game.get_camera_position()
        camera_rot = self.game.get_camera_rotation()

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

            0,  # 8

            # Checkpoints
            0,  # 8
            0,  # 9
            0,  # 10
            0,  # 11
            0,  # 12
            0,  # 13

            0,  # 14
            0,  # 15
            0,  # 16

            # Player data
            np.interp(distance_from_ground, (-64, 64), (-1, 1)),  # 17
            np.interp(speed, (0, 2), (-1, 1)),  # 18
            np.interp(player_state, (0, 255), (-1, 1)),  # 19
            np.interp(self.remaining_frames / 30, (0, 30), (-1, 1)),
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

        env = KerwanEnvironment(pid=pid, eval_mode=True, device="cpu")
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
