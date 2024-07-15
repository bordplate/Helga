import ctypes
import struct

import numpy as np

from Game.LinuxProcess import Process
from Game.Game import Game, Vector3


class RC1Game(Game):
    offset = 0x300000000

    frame_count_address = 0xB00000
    frame_progress_address = 0xB00004
    input_address = 0xB00008
    joystick_address = 0xB00010
    hoverboard_lady_ptr_address = 0xB00020
    skid_ptr_address = 0xB00024

    player_state_address = 0x96bd64
    player_position_address = 0x969d60
    player_rotation_address = 0x969d70
    dist_from_ground_address = 0x969fbc
    player_speed_address = 0x969e74
    current_planet_address = 0x969C70
    nanotech_address = 0x96BF88
    max_nanotech_address = 0x71fb28
    items_address = 0x96C140

    collisions_address = 0xB00600 + ((256*4) * 0)
    collisions_class_address = 0xB00600 + ((256*4) * 1)
    collisions_normals_x = 0xB00600 + ((256*4) * 2)
    collisions_normals_y = 0xB00600 + ((256*4) * 3)
    collisions_normals_z = 0xB00600 + ((256*4) * 4)

    oscillation_offset_x_address = 0xB00060
    oscillation_offset_y_address = 0xB00064

    should_render_address = 0xB00030

    coll_forward_address = 0xB00030
    coll_up_address = 0xB00034
    coll_down_addrss = 0xB00050
    coll_left_address = 0xB00038
    coll_right_address = 0xB0003c

    coll_class_forward_address = 0xB00040
    coll_class_up_address = 0xB00044
    coll_class_down_address = 0xB00054
    coll_class_left_address = 0xB00048
    coll_class_right_address = 0xB0004c

    camera_position_address = 0x951500
    camera_rotation_address = 0x951510

    camera_forward_address = 0x9513c0
    camera_right_address = 0x9513d0
    camera_up_address = 0x9513e0

    checkpoint_position_address = 0xb00400

    death_count_address = 0xB00500

    joystick_l_x = 0.0
    joystick_l_y = 0.0
    joystick_r_x = 0.0
    joystick_r_y = 0.0

    def __init__(self, pid):
        super().__init__(pid)

        self.game_key = "rc1"

        self.collisions_render = np.zeros((8*4, 8*4))

        # Init mobys_render as -1
        self.mobys_render = np.zeros((8*4, 8*4))
        self.mobys_render.fill(-1)

    def set_controller_input(self, controller_input, left_joy_x, left_joy_y, right_joy_x, right_joy_y):
        self.process.write_int(self.input_address, controller_input)

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

        joystick = joystick | (int((self.joystick_l_x+1) * 127) & 0xFF)
        joystick = joystick | ((int((self.joystick_l_y+1) * 127) & 0xFF) << 8)
        joystick = joystick | ((int((self.joystick_r_x+1) * 127) & 0xFF) << 16)
        joystick = joystick | ((int((self.joystick_r_y+1) * 127) & 0xFF) << 24)

        self.process.write_int(self.joystick_address, joystick)

    def set_item_unlocked(self, item_id):
        self.process.write_byte(self.items_address + item_id, 1)

    def get_current_frame_count(self):
        frames_buffer = self.process.read_memory(self.frame_count_address, 4)
        frame_count = 0
        if frames_buffer:
            frame_count = int.from_bytes(frames_buffer, byteorder='big', signed=False)

        return frame_count

    def get_player_state(self):
        player_state_buffer = self.process.read_memory(self.player_state_address, 4)
        player_state = 0
        if player_state_buffer:
            player_state = int.from_bytes(player_state_buffer, byteorder='big', signed=False)

        return player_state

    def get_nanotech(self):
        nanotech = self.process.read_int(self.nanotech_address)

        return nanotech

    def get_max_nanotech(self):
        return self.process.read_int(self.max_nanotech_address)

    def set_nanotech(self, nanotech):
        """
        Nanotech is health in Ratchet & Clank.
        """
        self.process.write_int(self.nanotech_address, nanotech)

    def set_max_nanotech(self, value):
        self.process.write_int(self.max_nanotech_address, value)

    def set_player_state(self, state):
        self.process.write_int(self.player_state_address, state)

    def get_distance_from_ground(self):
        return self.process.read_float(self.dist_from_ground_address)

    def set_player_speed(self, speed):
        self.process.write_float(self.player_speed_address, speed)

    def get_player_speed(self):
        return self.process.read_float(self.player_speed_address)

    def get_current_level(self):
        return self.process.read_int(self.current_planet_address)

    def get_skid_position(self) -> Vector3:
        if self.skid_address == 0:
            self.skid_address = self.process.read_int(self.skid_ptr_address)
            if self.skid_address == 0:
                return Vector3()

        skid_position_buffer = self.process.read_memory(self.skid_address + 0x10, 12)

        if skid_position_buffer is None:
            return Vector3()

        # Flip each 4 bytes to convert from big endian to little endian
        skid_position_buffer = (skid_position_buffer[3::-1] +
                                skid_position_buffer[7:3:-1] +
                                skid_position_buffer[11:7:-1])

        skid_position = Vector3()
        ctypes.memmove(ctypes.byref(skid_position), skid_position_buffer, ctypes.sizeof(skid_position))

        return skid_position

    def set_player_position(self, position: Vector3):
        position_x = struct.pack('>f', position.x)
        position_y = struct.pack('>f', position.y)
        position_z = struct.pack('>f', position.z)

        self.process.write_memory(self.player_position_address, position_x)
        self.process.write_memory(self.player_position_address + 4, position_y)
        self.process.write_memory(self.player_position_address + 8, position_z)

    def set_player_rotation(self, rotation: Vector3):
        rotation_x = struct.pack('>f', rotation.x)
        rotation_y = struct.pack('>f', rotation.y)
        rotation_z = struct.pack('>f', rotation.z)

        self.process.write_memory(self.player_rotation_address, rotation_x)
        self.process.write_memory(self.player_rotation_address + 4, rotation_y)
        self.process.write_memory(self.player_rotation_address + 8, rotation_z)

    def get_player_position(self) -> Vector3:
        """Player position is stored in big endian, so we need to convert it to little endian."""
        player_position_buffer = self.process.read_memory(self.player_position_address, 12)

        if player_position_buffer is None:
            return Vector3()

        # Flip each 4 bytes to convert from big endian to little endian
        player_position_buffer = (player_position_buffer[3::-1] +
                                  player_position_buffer[7:3:-1] +
                                  player_position_buffer[11:7:-1])

        player_position = Vector3()
        ctypes.memmove(ctypes.byref(player_position), player_position_buffer, ctypes.sizeof(player_position))

        return player_position

    def get_player_rotation(self) -> Vector3:
        """Player rotation is stored in big endian, so we need to convert it to little endian."""
        player_rotation_buffer = self.process.read_memory(self.player_rotation_address, 12)

        if player_rotation_buffer is None:
            return Vector3()

        # Flip each 4 bytes to convert from big endian to little endian
        player_rotation_buffer = (player_rotation_buffer[3::-1] +
                                  player_rotation_buffer[7:3:-1] +
                                  player_rotation_buffer[11:7:-1])

        player_rotation = Vector3()
        ctypes.memmove(ctypes.byref(player_rotation), player_rotation_buffer, ctypes.sizeof(player_rotation))

        return player_rotation

    def get_camera_position(self) -> Vector3:
        camera_position_buffer = self.process.read_memory(self.camera_position_address, 12)

        if camera_position_buffer is None:
            return Vector3()

        # Flip each 4 bytes to convert from big endian to little endian
        camera_position_buffer = (camera_position_buffer[3::-1] +
                                  camera_position_buffer[7:3:-1] +
                                  camera_position_buffer[11:7:-1])

        camera_position = Vector3()
        ctypes.memmove(ctypes.byref(camera_position), camera_position_buffer, ctypes.sizeof(camera_position))

        return camera_position

    def get_camera_rotation(self) -> Vector3:
        camera_rotation_buffer = self.process.read_memory(self.camera_rotation_address, 12)

        if camera_rotation_buffer is None:
            return Vector3()

        # Flip each 4 bytes to convert from big endian to little endian
        camera_rotation_buffer = (camera_rotation_buffer[3::-1] +
                                  camera_rotation_buffer[7:3:-1] +
                                  camera_rotation_buffer[11:7:-1])

        camera_rotation = Vector3()
        ctypes.memmove(ctypes.byref(camera_rotation), camera_rotation_buffer, ctypes.sizeof(camera_rotation))

        return camera_rotation

    def get_camera_vectors(self):
        """
        Gets and returns the forward, right, and up vectors of the camera.
        """
        camera_forward_buffer = self.process.read_memory(self.camera_forward_address, 12)
        camera_right_buffer = self.process.read_memory(self.camera_right_address, 12)
        camera_up_buffer = self.process.read_memory(self.camera_up_address, 12)

        if camera_forward_buffer is None or camera_right_buffer is None or camera_up_buffer is None:
            return Vector3(), Vector3(), Vector3()

        # Flip each 4 bytes to convert from big endian to little endian
        camera_forward_buffer = (camera_forward_buffer[3::-1] +
                                 camera_forward_buffer[7:3:-1] +
                                 camera_forward_buffer[11:7:-1])

        camera_right_buffer = (camera_right_buffer[3::-1] +
                               camera_right_buffer[7:3:-1] +
                               camera_right_buffer[11:7:-1])

        camera_up_buffer = (camera_up_buffer[3::-1] +
                            camera_up_buffer[7:3:-1] +
                            camera_up_buffer[11:7:-1])

        camera_forward = Vector3()
        camera_right = Vector3()
        camera_up = Vector3()

        ctypes.memmove(ctypes.byref(camera_forward), camera_forward_buffer, ctypes.sizeof(camera_forward))
        ctypes.memmove(ctypes.byref(camera_right), camera_right_buffer, ctypes.sizeof(camera_right))
        ctypes.memmove(ctypes.byref(camera_up), camera_up_buffer, ctypes.sizeof(camera_up))

        return camera_forward, camera_right, camera_up

    def get_collisions(self, normalized=True):
        collisions = []
        classes = []

        # Fetch all the memory we need in one go

        # 8 rows, 8 columns
        for i in range(16):
            for j in range(16):
                offset = 4 * (i * 16 + j)

                collision_address = self.collisions_address + offset
                collision_value = self.process.read_float(collision_address)
                # collision_value = Process.read_float_from_buffer(collision_memory, offset)

                class_address = self.collisions_class_address + offset
                class_value = self.process.read_int(class_address, signed=True)
                # class_value = Process.read_int_from_buffer(collision_memory, 0x100 + offset, signed=True)

                if normalized:
                    collision_value = np.interp(collision_value, [-32, 64], [-1, 1])
                    class_value = np.interp(class_value, [-64, 4096], [-1, 1])

                collisions.append(collision_value)
                classes.append(class_value)

        return [*collisions, *classes]

    def get_collisions_oscillated(self, normalized=True):
        collisions = []
        classes = []

        # 8 rows, 8 columns
        for i in range(8):
            for j in range(8):
                offset = 4 * (i * 8 + j)

                collision_address = self.collisions_address + offset
                collision_value = self.process.read_float(collision_address)

                class_address = self.collisions_class_address + offset
                class_value = self.process.read_int(class_address, signed=True)

                if normalized:
                    collision_value = np.interp(collision_value, [-32, 64], [-1, 1])
                    class_value = np.interp(class_value, [-128, 4096], [-1, 1])

                (oscillation_offset_x, oscillation_offset_y) = self.get_oscillation_offset()

                x_index = int(i * 4 + oscillation_offset_x)
                y_index = int(j * 4 + oscillation_offset_y)

                self.collisions_render[x_index, y_index] = collision_value
                self.mobys_render[x_index, y_index] = class_value

        # Flatten and return self.collisions_render and self.mobys_render
        return [*self.collisions_render.copy().flatten(), *self.mobys_render.copy().flatten()]

    def get_collisions_with_normals(self, normalized=True) -> list:
        collisions = []
        normals_x = []
        normals_y = []
        normals_z = []
        classes = []

        memory = self.process.read_memory(self.collisions_address, ((256*4) * 5))

        # 16 rows, 16 columns
        for i in range(16):
            for j in range(16):
                offset = 4 * (i * 16 + j)

                collision_address = self.collisions_address + offset
                collision_value = self.process.read_float_from_buffer(memory, 0x400 * 0 + offset)

                class_address = self.collisions_class_address + offset
                class_value = self.process.read_int_from_buffer(memory, 0x400 * 1 + offset, signed=True)

                normal_x_address = self.collisions_normals_x + offset
                normal_x = np.float64(self.process.read_float_from_buffer(memory, 0x400 * 2 + offset) / (1024 * 1024))

                normal_y_address = self.collisions_normals_y + offset
                normal_y = np.float64(self.process.read_float_from_buffer(memory, 0x400 * 3 + offset) / (1024 * 1024))

                normal_z_address = self.collisions_normals_z + offset
                normal_z = np.float64(self.process.read_float_from_buffer(memory, 0x400 * 4 + offset) / (1024 * 1024))

                # Normalize the normal vector
                magnitude = np.sqrt(normal_x ** 2 + normal_y ** 2 + normal_z ** 2)
                if magnitude != 0:
                    normal_x /= magnitude
                    normal_y /= magnitude
                    normal_z /= magnitude

                max = 1.0
                min = -1.0

                assert min <= normal_x <= max, f"Normal x: {normal_x}"
                assert min <= normal_y <= max, f"Normal y: {normal_y}"
                assert min <= normal_z <= max, f"Normal z: {normal_z}"

                if normalized:
                    collision_value = np.interp(collision_value, [-32, 64], [-1, 1])
                    class_value = np.interp(class_value, [-128, 4096], [-1, 1])

                collisions.append(collision_value)
                normals_x.append(normal_x)
                normals_y.append(normal_y)
                normals_z.append(normal_z)
                classes.append(class_value)

        return [*collisions, *classes, *normals_x, *normals_y, *normals_z]

    def get_oscillation_offset(self):
        return (int(self.process.read_float(self.oscillation_offset_x_address)),
                (self.process.read_float(self.oscillation_offset_y_address)))

    def get_collisions_old(self):
        coll_forward = self.process.read_float(self.coll_forward_address)
        coll_up = self.process.read_float(self.coll_up_address)
        coll_down = self.process.read_float(self.coll_down_addrss)
        coll_left = self.process.read_float(self.coll_left_address)
        coll_right = self.process.read_float(self.coll_right_address)

        coll_class_forward = self.process.read_int(self.coll_class_forward_address)
        coll_class_up = self.process.read_int(self.coll_class_up_address)
        coll_class_down = self.process.read_int(self.coll_class_down_address)
        coll_class_left = self.process.read_int(self.coll_class_left_address)
        coll_class_right = self.process.read_int(self.coll_class_right_address)

        return (coll_forward, coll_up, coll_down, coll_left, coll_right,
                coll_class_forward, coll_class_up, coll_class_down, coll_class_left, coll_class_right)

    def get_death_count(self):
        return self.process.read_int(self.death_count_address)

    def start_hoverboard_race(self):
        """
        To start the hoverboard race, we have to find a specific NPC in the game and set two of its properties to 3.
        """
        hoverboard_lady_ptr = self.process.read_int(self.hoverboard_lady_ptr_address)

        self.process.write_byte(hoverboard_lady_ptr + 0x20, 3)
        self.process.write_byte(hoverboard_lady_ptr + 0xbc, 3)

    def set_checkpoint_position(self, position: Vector3):
        position_x = struct.pack('>f', position.x)
        position_y = struct.pack('>f', position.y)
        position_z = struct.pack('>f', position.z)

        self.process.write_memory(self.checkpoint_position_address, position_x)
        self.process.write_memory(self.checkpoint_position_address + 4, position_y)
        self.process.write_memory(self.checkpoint_position_address + 8, position_z)

    def set_should_render(self, should_render):
        self.process.write_int(self.should_render_address, 1 if should_render else 0)

    def frame_advance(self, frameskip=1):
        frame_count = self.get_current_frame_count()
        target_frame = frame_count + frameskip

        self.process.write_int(self.frame_progress_address, target_frame)

        while frame_count < target_frame:
            frame_count = self.get_current_frame_count()

        return True
