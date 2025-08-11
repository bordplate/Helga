import os
import ctypes
import struct
import time

import psutil

# Constants for process_vm_readv and process_vm_writev
PROCESS_VM_READV = 3102
PROCESS_VM_WRITEV = 3103

# Define the iovec structure
class Iovec(ctypes.Structure):
    _fields_ = [("iov_base", ctypes.c_void_p),
                ("iov_len", ctypes.c_size_t)]

class Process:
    def __init__(self, pid, base_offset=0):
        self.pid = pid
        self.process = None
        self.base_offset = base_offset
        self.libc = ctypes.cdll.LoadLibrary('libc.so.6')

    def open_process(self):
        self.process = None

        # Find the process in the process list
        for process in psutil.process_iter(['pid', 'name']):
            if process.info['pid'] == self.pid:
                self.process = process
                break

        if self.process is None:
            print(f"RPCS3 process not found...")
            return False
        else:
            print(f"{self.process.info['name']} process found.")
            return True

    def read_memory(self, address, size):
        while self.process is None:
            self.open_process()
            time.sleep(1)

        buffer = ctypes.create_string_buffer(size)
        local_iov = Iovec(ctypes.cast(ctypes.pointer(buffer), ctypes.c_void_p), size)
        remote_iov = Iovec(ctypes.c_void_p(self.base_offset + address), size)

        bytes_read = self.libc.process_vm_readv(
            self.pid,
            ctypes.byref(local_iov), 1,
            ctypes.byref(remote_iov), 1,
            0
        )

        if bytes_read == -1:
            # raise OSError("Failed to read memory")
            print("Failed to read memory")
            return [0 * size]

        return buffer.raw

    def write_memory(self, address, data):
        size = len(data)
        buffer = ctypes.create_string_buffer(data)
        local_iov = Iovec(ctypes.cast(ctypes.pointer(buffer), ctypes.c_void_p), size)
        remote_iov = Iovec(ctypes.c_void_p(self.base_offset + address), size)

        bytes_written = self.libc.process_vm_writev(
            self.pid,
            ctypes.byref(local_iov), 1,
            ctypes.byref(remote_iov), 1,
            0
        )

        if bytes_written == -1:
            raise OSError("Failed to write memory")

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
        value = struct.pack('>f', value)
        if not self.write_memory(address, value):
            print("Failed to write memory.")

    def read_int(self, address, signed=False):
        buffer = self.read_memory(address, 4)
        value = int.from_bytes(buffer, byteorder='big', signed=signed)
        return value

    def read_float(self, address):
        buffer = self.read_memory(address, 4)
        buffer = buffer[::-1]
        value = ctypes.c_float.from_buffer_copy(buffer).value
        return value

    @staticmethod
    def read_float_from_buffer(buffer, offset):
        buffer = buffer[offset:offset + 4][::-1]
        value = ctypes.c_float.from_buffer_copy(buffer).value
        return value

    @staticmethod
    def read_int_from_buffer(buffer, offset, signed=False):
        buffer = buffer[offset:offset + 4]
        value = int.from_bytes(buffer, byteorder='big', signed=signed)
        return value
