import ctypes
import ctypes.wintypes as wintypes
import struct

import psutil

# Windows API functions
OpenProcess = ctypes.windll.kernel32.OpenProcess
ReadProcessMemory = ctypes.windll.kernel32.ReadProcessMemory
WriteProcessMemory = ctypes.windll.kernel32.WriteProcessMemory
CloseHandle = ctypes.windll.kernel32.CloseHandle

# Constants
PROCESS_ALL_ACCESS = 0x1F0FFF


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
