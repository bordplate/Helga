import ctypes
import math

from Game.Process import Process


# Vector3
class Vector3(ctypes.Structure):
    _fields_ = [("x", ctypes.c_float),
                ("y", ctypes.c_float),
                ("z", ctypes.c_float)]

    def distance_to_2d(self, other):
        return ((self.x - other.x) ** 2 + (self.y - other.y) ** 2) ** 0.5

    def distance_to(self, other):
        return ((self.x - other.x) ** 2 + (self.y - other.y) ** 2 + (self.z - other.z) ** 2) ** 0.5

    def angle(self):
        return math.atan2(self.y, self.x)

    def angle_to(self, other):
        return (other - self).angle()

    def is_looking_at(self, rotation, other, angle_threshold=0.1) -> bool:
        # Calculate the angle between the player's forward vector and the vector to the other object
        angle = self.angle_to(other)
        return abs(angle - rotation.z) < angle_threshold

    def __sub__(self, other):
        return Vector3(self.x - other.x, self.y - other.y, self.z - other.z)


class Game:
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
