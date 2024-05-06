import ctypes
import ctypes.wintypes as wintypes
import psutil
import struct

import numpy as np


# Windows API functions
OpenProcess = ctypes.windll.kernel32.OpenProcess
ReadProcessMemory = ctypes.windll.kernel32.ReadProcessMemory
WriteProcessMemory = ctypes.windll.kernel32.WriteProcessMemory
CloseHandle = ctypes.windll.kernel32.CloseHandle

# Constants
PROCESS_ALL_ACCESS = 0x1F0FFF


# Vector3
class Vector3(ctypes.Structure):
    _fields_ = [("x", ctypes.c_float),
                ("y", ctypes.c_float),
                ("z", ctypes.c_float)]

    def distance_to_2d(self, other):
        return ((self.x - other.x) ** 2 + (self.y - other.y) ** 2) ** 0.5

    def distance_to(self, other):
        return ((self.x - other.x) ** 2 + (self.y - other.y) ** 2 + (self.z - other.z) ** 2) ** 0.5


class Process:
    def __init__(self, process_name, base_offset=0):
        self.process_name = process_name
        self.process = None
        self.process_handle = None
        self.base_offset = base_offset

    def open_process(self):
        self.process = None

        # Find the process in the process list
        while self.process is None:
            for process in psutil.process_iter(['pid', 'name']):
                if process.info['name'] == self.process_name:
                    self.process = process
                    break

            if self.process is None:
                print("RPCS3 process not found...")
                return False
            else:
                self.process_handle = OpenProcess(PROCESS_ALL_ACCESS, False, self.process.info['pid'])
                print(f"RPCS3 process found. Handle: {self.process_handle}")

                return True

    def close_process(self):
        CloseHandle(self.process_handle)

    def read_memory(self, address, size):
        buffer = ctypes.create_string_buffer(size)
        bytes_read = ctypes.c_size_t()
        address = ctypes.c_void_p(self.base_offset + address)

        if ReadProcessMemory(self.process_handle, address, buffer, size, ctypes.byref(bytes_read)):
            return buffer.raw
        else:
            return None

    def write_memory(self, address, data):
        size = len(data)
        c_data = ctypes.create_string_buffer(data)
        bytes_written = ctypes.c_size_t()
        address = ctypes.c_void_p(self.base_offset + address)

        result = WriteProcessMemory(self.process_handle, address, c_data, size, ctypes.byref(bytes_written))

        return result

    def write_int(self, address, value):
        value_bytes = value.to_bytes(4, byteorder='big')
        if not self.write_memory(address, value_bytes):
            print("Failed to write memory.")

    def write_byte(self, address, value):
        value_bytes = value.to_bytes(1, byteorder='big')
        if not self.write_memory(address, value_bytes):
            print("Failed to write memory.")

    def write_float(self, address, value):
        # Float doesn't have a to_bytes function, so we use struct.pack for this one
        value = struct.pack('>f', value)
        if not self.write_memory(address, value):
            print("Failed to write memory.")

    def read_int(self, address):
        buffer = self.read_memory(address, 4)

        value = 0
        if buffer:
            value = int.from_bytes(buffer, byteorder='big', signed=False)

        return value

    def read_float(self, address):
        buffer = self.read_memory(address, 4)

        if buffer is None:
            return 0.0

        buffer = buffer[::-1]
        value = 0
        if buffer:
            # There's no float.from_bytes function
            value = ctypes.c_float.from_buffer_copy(buffer).value
        return value


class Game:
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
    nanotech_address = 0x96BF8B
    items_address = 0x96C140

    collisions_address = 0xB00100
    collisions_class_address = 0xB00200

    oscillation_offset_address = 0xB00060

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

    death_count_address = 0xB00500

    joystick_l_x = 0.0
    joystick_l_y = 0.0
    joystick_r_x = 0.0
    joystick_r_y = 0.0

    def __init__(self, process_name="rpcs3.exe"):
        self.process_name = process_name

        self.process = Process(self.process_name, base_offset=self.offset)
        self.last_frame_count = 0
        self.must_restart = False

    def open_process(self):
        self.process = Process(self.process_name, base_offset=self.offset)
        return self.process.open_process()

    def close_process(self):
        self.process.close_process()
        self.process = None

    def restart(self):
        self.close_process()
        self.open_process()
        self.must_restart = False

    def set_controller_input(self, controller_input, left_joy_x, left_joy_y, right_joy_x, right_joy_y):
        self.process.write_int(self.input_address, controller_input)

        # self.joystick_l_x += left_joy_x
        # self.joystick_l_y += left_joy_y
        # self.joystick_r_x += right_joy_x
        # self.joystick_r_y += right_joy_y
        #
        # # Clamp joysticks to -1 and 1
        # self.joystick_l_x = max(-1.0, min(1.0, self.joystick_l_x))
        # self.joystick_l_y = max(-1.0, min(1.0, self.joystick_l_y))
        # self.joystick_r_x = max(-1.0, min(1.0, self.joystick_r_x))
        # self.joystick_r_y = max(-1.0, min(1.0, self.joystick_r_y))

        self.joystick_l_x = max(-1.0, min(1.0, left_joy_x))
        self.joystick_l_y = max(-1.0, min(1.0, left_joy_y))
        self.joystick_r_x = max(-1.0, min(1.0, right_joy_x))
        self.joystick_r_y = max(-1.0, min(1.0, right_joy_y))

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

    def set_nanotech(self, nanotech):
        """
        Nanotech is health in Ratchet & Clank.
        """
        self.process.write_byte(self.nanotech_address, nanotech)

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

    def get_collisions(self, normalized=True):
        collisions = []
        classes = []

        # 8 rows, 8 columns
        for i in range(8):
            for j in range(8):
                offset = 4 * (i * 8 + j)

                collision_address = self.collisions_address + offset
                collision_value = self.process.read_float(collision_address)

                class_address = self.collisions_class_address + offset
                class_value = self.process.read_int(class_address)

                if normalized:
                    collision_value = np.interp(collision_value, [-32, 64], [-1, 1])
                    class_value = np.interp(class_value, [-2, 4096], [-1, 1])

                collisions.append(collision_value)
                classes.append(class_value)

        return [*collisions, *classes]

    def get_oscillation_offset(self):
        return self.process.read_float(self.oscillation_offset_address)

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

    def frame_advance(self):
        frame_count = self.get_current_frame_count()

        while frame_count == self.last_frame_count:
            if self.must_restart:
                self.restart()

                return False

            frame_count = self.get_current_frame_count()

        self.last_frame_count = frame_count

        self.process.write_int(self.frame_progress_address, frame_count)

        return True
