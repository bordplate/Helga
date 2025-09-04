import ctypes
import time

import numpy as np

# from .WindowsProcess import Process
from .Game import Game, Vector3


class RC3Game(Game):
    offset = 0x300000000

    game_reset_address = 0xcc5188

    frame_count_address = 0xcc5180
    frame_progress_address = 0xcc5184
    input_address = 0xcc5200
    joystick_address = 0xcc5204

    player_position_address = 0xcd0000
    player_rotation_address = 0xcd000c
    player_health_address = 0xcd0018
    player_team_id_address = 0xcd001c
    player_moby_state_address = 0xcd0020
    player_speed_address = 0xcd0024
    player_camera_forward_address = 0xcd0028

    player_info_size = 0x34

    team_flag_position_address = 0xce0000
    team_flag_state_address = 0xce000c
    team_health_address = 0xce0010
    team_flag_holder_address = 0xce0014
    team_score_address = 0xce0018

    team_info_size = 0x1c

    collisions_address = 0xcd0000 + ((256 * 4) * 0)
    collisions_class_address = 0xcd0000 + ((256 * 4) * 1)
    collisions_normals_x = 0xcd0000 + ((256 * 4) * 2)
    collisions_normals_y = 0xcd0000 + ((256 * 4) * 3)
    collisions_normals_z = 0xcd0000 + ((256 * 4) * 4)

    headless_address = 0x1B00200

    game_state_address = 0xee9334
    game_frame_count_address = 0xD6B8D4
    current_planet_address = 0xC1E438

    hero_position_address = 0xDA2870
    hero_rotation_address = 0xDA2880
    hero_state_address = 0xda4db4
    health_address = 0xda5040
    ammo_address = 0xDA42F8

    load_level_address = 0xEE9310
    destination_level_address = 0xEE9314

    vidcomic_state_address = 0xda5122

    def __init__(self, pid):
        super().__init__(pid)
        self.game_key = "rc3"

        self.team = -1

    def get_team(self, player):
        """Get the team of the player."""
        return self.process.read_int(self.player_team_id_address + self.player_info_size * player)

    def get_health(self, player):
        return self.process.read_float(self.player_health_address + self.player_info_size * player)

    def get_team_health(self, team_id):
        return self.process.read_float(self.team_health_address + self.team_info_size * team_id)

    def get_hero_position(self) -> Vector3:
        """Player position is stored in big endian, so we need to convert it to little endian."""
        hero_position_buffer = self.process.read_memory(self.hero_position_address, 12)

        if hero_position_buffer is None:
            return Vector3()

        # Flip each 4 bytes to convert from big endian to little endian
        hero_position_buffer = (hero_position_buffer[3::-1] +
                                hero_position_buffer[7:3:-1] +
                                hero_position_buffer[11:7:-1])

        hero_position = Vector3()
        ctypes.memmove(ctypes.byref(hero_position), hero_position_buffer, ctypes.sizeof(hero_position))

        return hero_position

    def team_has_flag(self, team_id) -> bool:
        team = 1 if team_id == 0 else 0
        return self.get_flag_holder(team) != -1

    def get_flag_holder(self, team_id) -> int:
        return self.process.read_int(self.team_flag_holder_address + self.team_info_size * team_id, True)

    def get_team_flag_position(self, team_id) -> Vector3:
        """
        Flag position is stored in big endian, so we need to convert it to little endian
        """
        team_flag_position_buffer = self.process.read_memory(self.team_flag_position_address + self.team_info_size * team_id, 12)

        if team_flag_position_buffer is None:
            return Vector3()

        team_flag_position_buffer = (team_flag_position_buffer[3::-1] +
                                     team_flag_position_buffer[7:3:-1] +
                                     team_flag_position_buffer[11:7:-1])

        team_flag_position = Vector3()
        ctypes.memmove(ctypes.byref(team_flag_position), team_flag_position_buffer, ctypes.sizeof(team_flag_position))

        return team_flag_position

    def get_distance_to_team_flag(self, player_id, team_id) -> float:
        player_position = self.get_player_position(player_id)
        team_flag_position = self.get_team_flag_position(team_id)

        return player_position.distance_to(team_flag_position)

    def get_player_position(self, player) -> Vector3:
        """
        Player position is stored in big endian, so we need to convert it to little endian.
        """
        player_position_buffer = self.process.read_memory(self.player_position_address + self.player_info_size * player, 12)

        if player_position_buffer is None:
            return Vector3()

        player_position_buffer = (
            player_position_buffer[3::-1] +
            player_position_buffer[7:3:-1] +
            player_position_buffer[11:7:-1]
        )

        player_position = Vector3()
        ctypes.memmove(ctypes.byref(player_position), player_position_buffer, ctypes.sizeof(player_position))

        return player_position

    def get_player_camera_forward(self, player_id) -> Vector3:
        """
        Camera forward is stored in big endian, so we need to convert it to little endian.
        """
        camera_forward_buffer = self.process.read_memory(self.player_camera_forward_address + self.player_info_size * player_id, 12)

        if camera_forward_buffer is None:
            return Vector3()

        camera_forward_buffer = (
            camera_forward_buffer[3::-1] +
            camera_forward_buffer[7:3:-1] +
            camera_forward_buffer[11:7:-1]
        )

        camera_forward = Vector3()
        ctypes.memmove(ctypes.byref(camera_forward), camera_forward_buffer, ctypes.sizeof(camera_forward))

        return camera_forward

    def get_player_state(self, player) -> int:
        return self.process.read_int(self.player_moby_state_address + self.player_info_size * player)

    def get_player_speed(self, player) -> float:
        return self.process.read_float(self.player_speed_address + self.player_info_size * player)

    def get_team_score(self, team_id: int) -> int:
        """Get the score of the team."""
        return self.process.read_int(self.team_score_address + self.team_info_size * team_id)

    def get_player_rotation(self, player) -> Vector3:
        """
        Player rotation is stored in big endian, so we need to convert it to little endian.
        """
        player_rotation_buffer = self.process.read_memory(self.player_rotation_address + self.player_info_size * player, 12)

        if player_rotation_buffer is None:
            return Vector3()

        player_rotation_buffer = (
            player_rotation_buffer[3::-1] +
            player_rotation_buffer[7:3:-1] +
            player_rotation_buffer[11:7:-1]
        )

        player_rotation = Vector3()
        ctypes.memmove(ctypes.byref(player_rotation), player_rotation_buffer, ctypes.sizeof(player_rotation))

        return player_rotation


    def get_hero_rotation(self) -> Vector3:
        """Player rotation is stored in big endian, so we need to convert it to little endian."""
        hero_rotation_buffer = self.process.read_memory(self.hero_rotation_address, 12)

        if hero_rotation_buffer is None:
            return Vector3()

        # Flip each 4 bytes to convert from big endian to little endian
        hero_rotation_buffer = (hero_rotation_buffer[3::-1] +
                                hero_rotation_buffer[7:3:-1] +
                                hero_rotation_buffer[11:7:-1])

        hero_rotation = Vector3()
        ctypes.memmove(ctypes.byref(hero_rotation), hero_rotation_buffer, ctypes.sizeof(hero_rotation))

        return hero_rotation

    def get_hero_state(self):
        return self.process.read_int(self.hero_state_address)

    def set_controller_input(self, player, controller_input, left_joy_x, left_joy_y, right_joy_x, right_joy_y):
        self.process.write_int(self.input_address + 0x8 * player, controller_input)

        # self.joystick_l_x += left_joy_x
        # self.joystick_l_y += left_joy_y
        # self.joystick_r_x += right_joy_x
        # self.joystick_r_y += right_joy_y

        self.joystick_l_x = left_joy_x
        self.joystick_l_y = left_joy_y
        self.joystick_r_x = right_joy_x
        self.joystick_r_y = right_joy_y

        # Clamp all the joystick values between -1 and 1
        self.joystick_l_x = np.clip(self.joystick_l_x, -1, 1)
        self.joystick_l_y = np.clip(self.joystick_l_y, -1, 1)
        self.joystick_r_x = np.clip(self.joystick_r_x, -1, 1)
        self.joystick_r_y = np.clip(self.joystick_r_y, -1, 1)

        joystick = 0

        joystick = joystick | (int((self.joystick_l_x + 1) * 127) & 0xFF)
        joystick = joystick | ((int((self.joystick_l_y + 1) * 127) & 0xFF) << 8)
        joystick = joystick | ((int((self.joystick_r_x + 1) * 127) & 0xFF) << 16)
        joystick = joystick | ((int((self.joystick_r_y + 1) * 127) & 0xFF) << 24)

        self.process.write_int(self.joystick_address + 0x8 * player, joystick)

    def reset(self, resets: int):
        self.process.write_int(self.game_reset_address, resets)

    def get_current_frame_count(self):
        frames_buffer = self.process.read_memory(self.frame_count_address, 4)
        frame_count = 0
        if frames_buffer:
            frame_count = int.from_bytes(frames_buffer, byteorder='big', signed=False)

        return frame_count

    def get_game_state(self):
        return self.process.read_int(self.game_state_address)

    def set_game_state(self, state):
        self.process.write_int(self.game_state_address, state)

    def get_game_frame_count(self):
        return self.process.read_int(self.game_frame_count_address)

    def get_current_level(self):
        return self.process.read_int(self.current_planet_address)

    def set_level(self, level):
        self.process.write_int(self.destination_level_address, level)
        self.process.write_int(self.load_level_address, 1)

    def set_vidcomic_state(self, state):
        self.process.write_byte(self.vidcomic_state_address, state)

    def get_collisions(self, normalized=True):
        collisions = []
        classes = []

        # Fetch all the memory we need in one go

        # 8 rows, 8 columns
        for i in range(16):
            for j in range(16):
                offset = 4 * (i * 16 + j)

                collision_address = self.collisions_address + offset
                collision_value = self.process.read_float(collision_address + 0x500 * self.player)
                # collision_value = Process.read_float_from_buffer(collision_memory, offset)

                class_address = self.collisions_class_address + offset
                class_value = self.process.read_int(class_address + 0x500 * self.player, signed=True)
                # class_value = Process.read_int_from_buffer(collision_memory, 0x100 + offset, signed=True)

                if normalized:
                    collision_value = np.interp(collision_value, [-32, 64], [-1, 1])
                    class_value = np.interp(class_value, [-64, 4096], [-1, 1])

                collisions.append(collision_value)
                classes.append(class_value)

        return [*collisions, *classes]

    def get_collisions_with_normals(self, normalized=False) -> list:
        collisions = []
        normals_x = []
        normals_y = []
        normals_z = []
        classes = []

        memory = self.process.read_memory(self.collisions_address + 0x500 * self.player, ((64*4) * 5))

        # 8 rows, 8 columns
        for i in range(8):
            for j in range(8):
                offset = 4 * (i * 8 + j)

                collision_address = self.collisions_address + offset
                collision_value = self.process.read_float_from_buffer(memory, 0x100 * 0 + offset)

                class_address = self.collisions_class_address + offset
                class_value = self.process.read_int_from_buffer(memory, 0x100 * 1 + offset, signed=True)

                normal_x_address = self.collisions_normals_x + offset
                normal_x = np.float64(self.process.read_float_from_buffer(memory, 0x100 * 2 + offset) / (1024 * 1024))

                normal_y_address = self.collisions_normals_y + offset
                normal_y = np.float64(self.process.read_float_from_buffer(memory, 0x100 * 3 + offset) / (1024 * 1024))

                normal_z_address = self.collisions_normals_z + offset
                normal_z = np.float64(self.process.read_float_from_buffer(memory, 0x100 * 4 + offset) / (1024 * 1024))

                # Normalize the normal vector
                # magnitude = np.sqrt(normal_x ** 2 + normal_y ** 2 + normal_z ** 2)
                # if magnitude != 0:
                #     normal_x /= magnitude
                #     normal_y /= magnitude
                #     normal_z /= magnitude

                max = 1.0
                min = -1.0

                # assert min <= normal_x <= max, f"Normal x: {normal_x}"
                # assert min <= normal_y <= max, f"Normal y: {normal_y}"
                # assert min <= normal_z <= max, f"Normal z: {normal_z}"

                if normalized:
                    collision_value = np.interp(collision_value, [-32, 64], [-1, 1])
                    class_value = np.interp(class_value, [-128, 4096], [-1, 1])

                collisions.append(collision_value)
                normals_x.append(normal_x)
                normals_y.append(normal_y)
                normals_z.append(normal_z)
                classes.append(class_value)

        return [*collisions, *classes, *normals_x, *normals_y, *normals_z]

    def set_health(self, health):
        self.process.write_int(self.health_address, health)

    def get_ammo(self):
        return self.process.read_int(self.ammo_address)

    def set_headless(self, headless: bool):
        self.process.write_int(self.headless_address, 1 if headless else 0)

    def frame_advance(self, frameskip=1):
        frame_count = self.get_current_frame_count()
        target_frame = frame_count + frameskip

        self.process.write_int(self.frame_progress_address, target_frame)

        start_time = time.time()

        while frame_count < target_frame:
            frame_count = self.get_current_frame_count()

            # If we've waited 2 minutes, exit
            if time.time() - start_time > 120:
                exit(0)

        return True

