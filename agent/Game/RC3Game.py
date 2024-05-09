import ctypes

import numpy as np

from .Process import Process
from .Game import Game, Vector3


class RC3Game(Game):
    offset = 0x300000000

    frame_count_address = 0x1B00000
    frame_progress_address = 0x1B00004
    input_address = 0x1B00008

    collision_info_address = 0x1B00010

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

    def __init__(self, process_name="rpcs3.exe"):
        super().__init__(process_name)

        self.game_key = "rc3"

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

    def set_controller_input(self, controller_input):
        self.process.write_int(self.input_address, controller_input)

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

    # Collision info is an array of 8 floats
    def get_collision_info(self):
        collisions = []
        types = []

        for i in range(16):
            collision = self.process.read_float(self.collision_info_address + i * 4 * 2)
            type = self.process.read_int(self.collision_info_address + i * 4 * 2 + 4)

            # Normalize
            collision = np.interp(collision, [-10, 60], [-1.0, 1.0])
            type = np.interp(type, [0, 1024*16], [-1.0, 1.0])

            collisions.append(collision)
            types.append(type)

        return collisions, types

    def get_health(self):
        return self.process.read_int(self.health_address)

    def set_health(self, health):
        self.process.write_int(self.health_address, health)

    def get_ammo(self):
        return self.process.read_int(self.ammo_address)

    def set_headless(self, headless: bool):
        self.process.write_int(self.headless_address, 1 if headless else 0)

    def frame_advance(self, blocking=True):
        if self.must_restart:
            self.process.open_process()
            self.must_restart = False

        frame_count = self.get_current_frame_count()

        if blocking:
            while frame_count == self.last_frame_count:
                if self.must_restart:
                    self.process.open_process()
                    self.must_restart = False

                    return False

                frame_count = self.get_current_frame_count()

            self.last_frame_count = frame_count

        self.process.write_int(self.frame_progress_address, frame_count)

        return True

