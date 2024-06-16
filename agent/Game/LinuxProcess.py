import os
import ctypes
import struct

import psutil

# Constants
PROCESS_ALL_ACCESS = os.O_RDWR | os.O_SYNC

class Process:
    def __init__(self, pid, base_offset=0):
        self.pid = pid
        self.process = None
        self.process_handle = None
        self.base_offset = base_offset

    def open_process(self):
        self.process = None

        # Find the process in the process list
        while self.process is None:
            for process in psutil.process_iter(['pid', 'name']):
                if process.info['pid'] == self.pid:
                    self.process = process
                    break

            if self.process is None:
                print(f"RPCS3 process not found...")
                return False
            else:
                self.process_handle = os.open(f"/proc/{self.process.info['pid']}/mem", PROCESS_ALL_ACCESS)
                print(f"{self.process.info['name']} process found. Handle: {self.process_handle}")

                return True

    def close_process(self):
        os.close(self.process_handle)

    def read_memory(self, address, size):
        buffer = ctypes.create_string_buffer(size)
        with open(f"/proc/{self.process.info['pid']}/mem", 'rb') as mem_file:
            mem_file.seek(self.base_offset + address)
            buffer.raw = mem_file.read(size)

        return buffer.raw

    def write_memory(self, address, data):
        with open(f"/proc/{self.process.info['pid']}/mem", 'wb') as mem_file:
            mem_file.seek(self.base_offset + address)
            mem_file.write(data)

        return True

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