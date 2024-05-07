import ctypes

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
