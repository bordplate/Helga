import ctypes
import struct

from agent.Game.PINE import PINE


class Process:
    def __init__(self, slot, base_offset=0):
        self.base_offset = 0
        self.pine = PINE(28020, 1000)

    def open_process(self):
        return True

    def close(self):
        self.pine.close()

    def read_memory(self, address, size):
        address += self.base_offset
        return self.pine.read(address, size)

    def write_memory(self, address, data):
        address += self.base_offset
        self.pine.write(address, data)

    def write_int(self, address, value):
        address += self.base_offset
        value_bytes = value.to_bytes(4, byteorder='big')
        self.write_memory(address, value_bytes)

    def write_byte(self, address, value):
        address += self.base_offset
        value_bytes = value.to_bytes(1, byteorder='big')
        self.write_memory(address, value_bytes)

    def write_float(self, address, value):
        address += self.base_offset
        value_bytes = struct.pack('>f', value)
        self.write_memory(address, value_bytes)

    def read_int(self, address, signed=False):
        address += self.base_offset
        buffer = self.read_memory(address, 4)
        if buffer:
            value = int.from_bytes(buffer, byteorder='big', signed=signed)
            return value
        return 0

    def read_float(self, address):
        address += self.base_offset
        buffer = self.read_memory(address, 4)
        if buffer:
            buffer = buffer[::-1]
            value = ctypes.c_float.from_buffer_copy(buffer).value
            return value
        return 0.0

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


if __name__ == '__main__':
    # Example usage
    slot = 28012  # Replace with the actual slot number
    process = Process(slot)

    # Example read/write operations
    address = 0x96BF88  # Replace with the actual address
    value = process.read_int(address)
    print(f"Value at {hex(address)}: {value}")

    new_value = 100
    process.write_int(address, new_value)
    print(f"New value at {hex(address)}: {process.read_int(address)}")

    process.close()
