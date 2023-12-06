import ctypes
import ctypes.wintypes as wintypes
import psutil
import time
import numpy as np

# Windows API functions
OpenProcess = ctypes.windll.kernel32.OpenProcess
ReadProcessMemory = ctypes.windll.kernel32.ReadProcessMemory
WriteProcessMemory = ctypes.windll.kernel32.WriteProcessMemory
CloseHandle = ctypes.windll.kernel32.CloseHandle

# Constants
PROCESS_ALL_ACCESS = 0x1F0FFF
frame_count_address = 0xB00000
frame_progress_address = 0xB00004
input_address = 0xB00008
hoverboard_lady_ptr_address = 0xB00020
skid_ptr_address = 0xB00024
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

offset = 0x300000000

player_state_address = 0x96bd64
player_position_address = 0x969d60
player_rotation_address = 0x969d70
dist_from_ground_address = 0x969fbc
player_speed_address = 0x969e74
current_planet_address = 0x969C70


def normalize_x(x, min, max):
    normalized_x = 2 * ((x - min) / (max - min)) - 1
    return normalized_x


# Vector3
class Vector3(ctypes.Structure):
    _fields_ = [("x", ctypes.c_float),
                ("y", ctypes.c_float),
                ("z", ctypes.c_float)]


class RatchetEnvironment:
    def __init__(self):
        self.checkpoints_template = [
            Vector3(211.14610290527344, 231.3751220703125, 76.0),
            Vector3(118.8655, 265.8282, 77.0),
            Vector3(157.5900, 242.4351, 78.0),
            Vector3(66.0811, 299.7586, 70.0),
            Vector3(84.4390, 375.6922, 78.0),
            Vector3(74.7129898071289, 292.6900329589844, 68.0),
            Vector3(119.23978424072266, 460.4266357421875, 77.04088592529297),
            Vector3(189.29335021972656, 429.5045471191406, 68.0),
            Vector3(282.83135986328125, 444.8203430175781, 65.76412200927734),
            Vector3(338.6705017089844, 401.8573303222656, 74.0),
            Vector3(246.55624389648438, 284.5892028808594, 76.0)
        ]

        self.checkpoints = []

        self.process_handle = None
        self.must_restart = False
        self.last_frame_count = 0

        self.last_position = None
        self.distance = 0.0
        self.distance_traveled = 0.0
        self.timer = 0
        self.last_rotation = 0.0
        self.height_lost = 0.0

        self.checkpoint = 0

        self.distance_from_skid_per_step = []

        self.skid_address = 0
        self.jump_debounce = 0

        self.currently_dead = False
        self.death_counter = 0

        self.stalled_timer = 0

        self.frames_moving_away_from_skid = 0

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
            'rewards/void_penalty': 0
        }

        self.skid_checkpoints = []

    # Function to read memory
    def read_memory(self, address, size):
        buffer = ctypes.create_string_buffer(size)
        bytesRead = ctypes.c_size_t()
        address = ctypes.c_void_p(address)
        if ReadProcessMemory(self.process_handle, address, buffer, size, ctypes.byref(bytesRead)):
            return buffer.raw
        else:
            return None

    def write_memory(self, address, data):
        size = len(data)
        c_data = ctypes.create_string_buffer(data)
        bytes_written = ctypes.c_size_t()
        address = ctypes.c_void_p(address)
        result = WriteProcessMemory(self.process_handle, address, c_data, size, ctypes.byref(bytes_written))
        return result

    def write_int(self, address, value):
        # Convert frame_count to bytes and write back
        value_bytes = value.to_bytes(4, byteorder='big')
        if not self.write_memory(address, value_bytes):
            print("Failed to write memory.")

    def write_byte(self, address, value):
        # Convert frame_count to bytes and write back
        value_bytes = value.to_bytes(1, byteorder='big')
        if not self.write_memory(address, value_bytes):
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

    def open_process(self):
        # Find RPCS3 process
        rpcs3_process = None

        while rpcs3_process is None:
            for process in psutil.process_iter(['pid', 'name']):
                if process.info['name'] == 'rpcs3.exe':
                    rpcs3_process = process
                    break

            if rpcs3_process is None:
                print("RPCS3 process not found...")
            else:
                self.process_handle = OpenProcess(PROCESS_ALL_ACCESS, False, rpcs3_process.info['pid'])
                print(f"RPCS3 process found. Handle: {self.process_handle}")

            time.sleep(1)

    def close_process(self):
        CloseHandle(self.process_handle)

    def get_current_frame_count(self):
        frames_buffer = self.read_memory(offset + frame_count_address, 4)
        frame_count = 0
        if frames_buffer:
            frame_count = int.from_bytes(frames_buffer, byteorder='big', signed=False)

        return frame_count

    def get_player_state(self):
        player_state_buffer = self.read_memory(offset + player_state_address, 4)
        player_state = 0
        if player_state_buffer:
            player_state = int.from_bytes(player_state_buffer, byteorder='big', signed=False)

        return player_state

    def set_nanotech(self, nanotech):
        self.write_byte(offset + 0x96BF8B, nanotech)

    def set_player_state(self, state):
        self.write_int(offset + player_state_address, state)

    def get_distance_from_ground(self):
        return self.read_float(offset + dist_from_ground_address)

    def get_player_speed(self):
        return self.read_float(offset + player_speed_address)

    def get_current_planet(self):
        return self.read_int(offset + current_planet_address)

    def get_skid_position(self):
        if self.skid_address == 0:
            self.skid_address = self.read_int(offset + skid_ptr_address)
            if self.skid_address == 0:
                return Vector3()

        skid_position_buffer = self.read_memory(offset + self.skid_address + 0x10, 12)

        if skid_position_buffer is None:
            return Vector3()

        # Flip each 4 bytes to convert from big endian to little endian
        skid_position_buffer = (skid_position_buffer[3::-1] +
                                skid_position_buffer[7:3:-1] +
                                skid_position_buffer[11:7:-1])

        skid_position = Vector3()
        ctypes.memmove(ctypes.byref(skid_position), skid_position_buffer, ctypes.sizeof(skid_position))

        return skid_position

    def get_player_position(self):
        """Player position is stored in big endian, so we need to convert it to little endian."""
        player_position_buffer = self.read_memory(offset + player_position_address, 12)

        if player_position_buffer is None:
            return Vector3()

        # Flip each 4 bytes to convert from big endian to little endian
        player_position_buffer = (player_position_buffer[3::-1] +
                                  player_position_buffer[7:3:-1] +
                                  player_position_buffer[11:7:-1])

        player_position = Vector3()
        ctypes.memmove(ctypes.byref(player_position), player_position_buffer, ctypes.sizeof(player_position))

        return player_position

    def get_player_rotation(self):
        """Player rotation is stored in big endian, so we need to convert it to little endian."""
        player_rotation_buffer = self.read_memory(offset + player_rotation_address, 12)

        if player_rotation_buffer is None:
            return Vector3()

        # Flip each 4 bytes to convert from big endian to little endian
        player_rotation_buffer = (player_rotation_buffer[3::-1] +
                                  player_rotation_buffer[7:3:-1] +
                                  player_rotation_buffer[11:7:-1])

        player_rotation = Vector3()
        ctypes.memmove(ctypes.byref(player_rotation), player_rotation_buffer, ctypes.sizeof(player_rotation))

        return player_rotation

    def get_collisions(self):
        coll_foward = self.read_float(offset + coll_forward_address)
        coll_up = self.read_float(offset + coll_up_address)
        coll_down = self.read_float(offset + coll_down_addrss)
        coll_left = self.read_float(offset + coll_left_address)
        coll_right = self.read_float(offset + coll_right_address)

        coll_class_forward = self.read_int(offset + coll_class_forward_address)
        coll_class_up = self.read_int(offset + coll_class_up_address)
        coll_class_down = self.read_int(offset + coll_class_down_address)
        coll_class_left = self.read_int(offset + coll_class_left_address)
        coll_class_right = self.read_int(offset + coll_class_right_address)

        return (coll_foward, coll_up, coll_down, coll_left, coll_right,
                coll_class_forward, coll_class_up, coll_class_down, coll_class_left, coll_class_right)

    def get_collisions_normalized(self):
        coll_forward = normalize_x(self.read_float(offset + coll_forward_address), -33, 256)
        coll_up = normalize_x(self.read_float(offset + coll_up_address), -33, 256)
        coll_down = normalize_x(self.read_float(offset + coll_down_addrss), -33, 256)
        coll_left = normalize_x(self.read_float(offset + coll_left_address), -33, 256)
        coll_right = normalize_x(self.read_float(offset + coll_right_address), -33, 256)

        coll_class_forward = normalize_x(self.read_int(offset + coll_class_forward_address), 0, 16384)
        coll_class_up = normalize_x(self.read_int(offset + coll_class_up_address), 0, 16384)
        coll_class_down = normalize_x(self.read_int(offset + coll_class_down_address), 0, 16384)
        coll_class_left = normalize_x(self.read_int(offset + coll_class_left_address), 0, 16384)
        coll_class_right = normalize_x(self.read_int(offset + coll_class_right_address), 0, 16384)

        return (coll_forward, coll_up, coll_down, coll_left, coll_right,
                coll_class_forward, coll_class_up, coll_class_down, coll_class_left, coll_class_right)

    def distance_between_positions_2d(self, position1, position2):
        return ((position1.x - position2.x) ** 2 + (position1.y - position2.y) ** 2) ** 0.5

    def frame_advance(self):
        frame_count = self.get_current_frame_count()

        while frame_count == self.last_frame_count:
            if self.must_restart:
                self.open_process()
                self.must_restart = False

                return False

            frame_count = self.get_current_frame_count()

        self.last_frame_count = frame_count

        self.write_int(offset + frame_progress_address, frame_count)

        return True

    def reset(self):
        # Check that planet is correct
        while self.get_current_planet() != 5:
            print("Waiting for Rilgar level change...")

            if self.must_restart:
                break

            time.sleep(1)

        hoverboard_lady_ptr = self.read_int(offset + hoverboard_lady_ptr_address)
        self.write_byte(offset + hoverboard_lady_ptr + 0x20, 3)
        self.write_byte(offset + hoverboard_lady_ptr + 0xbc, 3)
        self.set_player_state(0)
        self.frame_advance()
        self.set_nanotech(4)
        self.stalled_timer = 0

        self.last_position = None
        self.distance = 0.0
        self.distance_traveled = 0.0
        self.timer = 0
        self.last_frame_count = 0
        self.last_rotation = 0.0
        self.height_lost = 0.0
        self.checkpoint = 0
        self.jump_debounce = 0

        # Create checkpoints from self.checkpoint_template and jitter them slightly to make them harder to memorize
        self.checkpoints = []
        for checkpoint in self.checkpoints_template:
            self.checkpoints.append(Vector3(checkpoint.x + np.random.uniform(-1, 1),
                                            checkpoint.y + np.random.uniform(-1, 1),
                                            checkpoint.z))

        self.frames_moving_away_from_skid = 0

        self.currently_dead = False
        self.death_counter = 0

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
            'rewards/reached_checkpoint_reward': 0
        }

        self.distance_from_skid_per_step = []
        self.skid_checkpoints = []

        # Clear inputs
        self.write_int(offset + input_address, 0)

        # Frame advance a couple of frames before giving control
        for i in range(10):
            self.frame_advance()

        # Find Skid
        self.skid_address = self.read_int(offset + skid_ptr_address)

        return self.step(0)

    def step(self, action):
        state, reward, terminal = None, 0.0, False
        self.timer += 1

        if self.timer > 60 * 60 * 5:  # 5 minutes
            terminal = True
            self.reward_counters['rewards/timeout_penalty'] += 1
            reward -= 1.0
            print("Timeout!")

        if self.skid_address == 0:
            print("Skid address 0!")
            self.skid_address = self.read_int(offset + skid_ptr_address)

        actions_mapping = [
            0x0,     # No action
            0x8000,  # Left
            0x2000,  # Right
            0x40,    # Jump
        ]

        # Discourage excessive jumping
        if self.jump_debounce > 0:
            self.jump_debounce -= 1

        if action == 3:
            if self.jump_debounce > 0:
                self.reward_counters['rewards/jump_penalty'] += self.jump_debounce * 0.00001
                reward -= self.jump_debounce * 0.00001
            self.jump_debounce += 5

        self.write_int(offset + input_address, actions_mapping[action])

        next_checkpoint_position = self.checkpoints[self.checkpoint]

        # Check that player is moving towards skid
        old_distance_from_checkpoint = self.distance_between_positions_2d(self.get_player_position(), next_checkpoint_position)

        # Advance frame
        if not self.frame_advance() or not self.frame_advance():
            reward -= 1.0
            self.reward_counters['rewards/crash_penalty'] += 1
            terminal = True

        check_delta_x, check_delta_y = (next_checkpoint_position.x - self.get_player_position().x,
                                        next_checkpoint_position.y - self.get_player_position().y)

        check_delta_x, check_delta_y = max(-1, min(1, check_delta_x)), max(-1, min(1, check_delta_y))

        new_distance_from_checkpoint = self.distance_between_positions_2d(self.get_player_position(), next_checkpoint_position)

        if self.frames_moving_away_from_skid < 0:
            self.frames_moving_away_from_skid = 0
        elif self.frames_moving_away_from_skid > 20:
            self.frames_moving_away_from_skid = 20

        if new_distance_from_checkpoint < old_distance_from_checkpoint:
            self.reward_counters['rewards/distance_from_checkpoint_reward'] += 0.005 + (old_distance_from_checkpoint - new_distance_from_checkpoint) * 0.25
            reward += 0.005 + (old_distance_from_checkpoint - new_distance_from_checkpoint) * 0.25
            self.frames_moving_away_from_skid -= 0
        else:
            self.frames_moving_away_from_skid += 1

            self.reward_counters['rewards/distance_from_checkpoint_penalty'] += 0.005 + (self.frames_moving_away_from_skid * 0.01)
            reward -= 0.005 + (self.frames_moving_away_from_skid * 0.01)

        # If agent is within 20 units of checkpoint, go to next checkpoint or loop around
        if new_distance_from_checkpoint < 15:
            self.checkpoint += 1
            if self.checkpoint >= len(self.checkpoints):
                self.checkpoint = 0

            self.reward_counters['rewards/reached_checkpoint_reward'] += 0.8
            reward += 0.8

            next_checkpoint_position = self.checkpoints[self.checkpoint]

        position = self.get_player_position()
        player_rotation = self.get_player_rotation()
        distance_from_ground = self.get_distance_from_ground()
        speed = self.get_player_speed()

        if speed < 0.1:
            self.stalled_timer += 1
            if self.stalled_timer > 30 * 5:
                terminal = True
                self.reward_counters['rewards/timeout_penalty'] += 1
                reward -= 1.0
                print("Stalled!")
        else:
            self.stalled_timer = 0

        if speed > 0.25:
            self.reward_counters['rewards/speed_reward'] += (speed - 0.25) * 0.05
            reward += (speed - 0.25) * 0.05

        # Discourage stalling
        if speed < 0.25:
            self.reward_counters['rewards/stall_penalty'] += 0.1
            reward -= 0.1

        # Get distance player has moved since last frame to calculate reward
        if self.last_position is not None:
            distance = self.distance_between_positions_2d(position, self.last_position)
            if distance > 1.0:
                #print(f"Discarding unreasonable distance: {distance}")
                distance = 0.0

            self.distance_traveled += distance

            if distance > 0:
                if self.last_position.z > position.z:
                    self.height_lost += self.last_position.z - position.z

                # if distance_from_ground > 0:
                #     distance = 0.0
                #     reward -= 0.1

            # No reward if player has rotated
            #if player_rotation.z == self.last_rotation:
            #reward += distance * 2 #+ (self.timer * 0.01)
            if distance_from_ground < 20:
                self.distance += distance

        # if self.distance > 45:
        #     self.reward_counters['rewards/distance_traveled_reward'] += 0.9
        #     reward += 0.9
        #     self.distance = 0

        # Each 5 seconds the agent survives, give a reward
        # if self.timer % 60 == 0:
        #     reward += 0.1

        # Discount distance from skid
        # distance_from_skid = self.distance_between_positions_2d(position, skid_position)

        # if self.timer / 60 > 5:
        #     reward += (-distance_from_skid + 30) * 0.1

        # # Reward agent if it is pointing towards skid
        # angle = abs(player_rotation.z - np.arctan2(skid_position.y - position.y, skid_position.x - position.x))
        # if angle < 0.5:
        #     reward += 1.0

        # reward -= distance_from_ground * 0.2

        #print(f"Skid position: {skid_position.x}, {skid_position.y}, {skid_position.z}. Distance: {distance_from_skid}")

        self.last_position = position
        self.last_rotation = player_rotation.z

        player_state = self.get_player_state()

        if self.currently_dead and player_state == 107:
            self.currently_dead = False

        if not self.currently_dead and (
                player_state == 109 or player_state == 110 or player_state == 111
        ):
            self.reward_counters['rewards/death_penalty'] += 1
            reward -= 1.0
            self.currently_dead = True
            self.death_counter += 1

            if self.death_counter >= 1:
                terminal = True

        self.distance_from_skid_per_step.append(new_distance_from_checkpoint)

        coll_f, coll_u, coll_d, coll_l, coll_r, coll_cf, coll_cu, coll_cd, coll_cl, coll_cr = self.get_collisions()

        # Penalize various collisions
        if coll_f != -32.0 and coll_u != -32.0 and coll_f < 3.5 and coll_u < 4.5 and coll_cf == 0:
            reward -= 0.2
            self.reward_counters['rewards/wall_crash_penalty'] += 1

        # TNT crate collision
        if (
                (coll_f != -32.0 and coll_f < 4.5 and coll_cf == 505) or
                (coll_d != -32.0 and coll_d < 4.5 and coll_cd == 505)
        ):
            reward -= 0.2
            self.reward_counters['rewards/tnt_crash_penalty'] += 1

        if distance_from_ground > 31 and coll_d <= -32.0:
            self.reward_counters['rewards/void_penalty'] += 0.2
            reward -= 0.2

        state = [
            normalize_x(position.x, 0, 500),  # 0
            normalize_x(position.y, 0, 500),  # 1
            normalize_x(position.z, -150, 150),   # 2
            normalize_x(next_checkpoint_position.x, 0, 500),  # 7
            normalize_x(next_checkpoint_position.y, 0, 500),  # 8
            check_delta_x,
            check_delta_y,
            normalize_x(player_rotation.z, -20, 20), # 3
            normalize_x(distance_from_ground, -64, 64),  # 4
            normalize_x(speed, 0, 2),  # 5
            normalize_x(self.timer, 0, 60 * 60 * 60 * 24),  # 6 # 24 hours
            normalize_x(new_distance_from_checkpoint, 0, 500),  # 9
            normalize_x(player_state, 0, 255),  # 10
            normalize_x(self.distance_traveled, 0, 100000),  # 11
            *self.get_collisions_normalized()  # 12-21
        ]

        if reward > 1 or reward < -1:
            # Print if wildly out of bounds
            if reward > 2 or reward < -2:
                print(f"Danger! Reward out of bounds: {reward}")

            # Clamp
            reward = max(-1, min(1, reward))


        # Iterate through the state to check that none of the values are above 1 or below -1
        for s, state_value in enumerate(state):
            if state_value > 1.0 or state_value < -1.0:
                print(f"Danger! State out of bounds: {s}. Value: {state_value}")
                exit(0)

        return np.array(state, dtype=np.float32), reward, terminal


if __name__ == '__main__':
    env = RatchetEnvironment()
    env.open_process()

    #env.reset()

    # Get and print hoverboard lady address as hex
    # hoverboard_lady_ptr = env.read_int(offset + hoverboard_lady_ptr_address)
    # print(f"Hoverboard lady pointer: {hex(hoverboard_lady_ptr)}. With offset: {hex(hoverboard_lady_ptr + offset)}")

    try:
        steps = 0
        next_frame_time = time.time()

        last_checkpoint = None

        while True:
            current_time = time.time()

            # Busy-wait until it's time for the next frame
            # while current_time < next_frame_time:
            #     current_time = time.time()

            input()

            print(f"Collisions:", *env.get_collisions())

            # current_position = env.get_player_position()
            # print(f"Vector3({current_position.x}, {current_position.y}, {current_position.z}),", end="")

            # if last_checkpoint is None:
            #     last_checkpoint = env.get_player_position()

            # Schedule next frame
            next_frame_time += 1 / 60  # Schedule for the next 1/60th second
            time.sleep(0.016)

            # current_position = env.get_player_position()
            #
            # if env.distance_between_positions_2d(current_position, last_checkpoint) > 40:
            #     print(f"{current_position.x}, {current_position.y}, {current_position.z}")
            #     last_checkpoint = current_position

            # skid_pos = env.get_skid_position()
            # print(f"Skid position: {skid_pos.x}, {skid_pos.y}, {skid_pos.z}")

            #state__, reward__, terminal__ = env.step(0)
            # state__, reward__, terminal__ = env.step(np.random.choice(4))

            # print(f"Collisions:", *env.get_collisions())
            #
            # if terminal__:
            #     env.reset()

            # env.step(0x1040 if steps % 2 == 0 else 0x1000)

            steps += 1

    except KeyboardInterrupt:
        env.close_process()
