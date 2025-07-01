import socket
import struct
import os
import sys

# Constants for PINE standard opcodes
OP_READ8 = 0
OP_READ16 = 1
OP_READ32 = 2
OP_READ64 = 3
OP_WRITE8 = 4
OP_WRITE16 = 5
OP_WRITE32 = 6
OP_WRITE64 = 7
OP_VERSION = 8
OP_SAVESTATE = 9
OP_LOADSTATE = 10
OP_GAMETITLE = 11
OP_GAMEID = 12
OP_GAMEUUID = 13
OP_GAMEVERSION = 14
OP_EMUSTATUS = 15


class EmulatorStatus:
    RUNNING = 0
    PAUSED = 1
    SHUTDOWN = 2


class PINE:
    def __init__(self, slot, timeout=1000):
        if sys.platform == 'win32':
            self.client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.client.settimeout(timeout / 1000)
            self.client.connect(('127.0.0.1', slot))
        else:
            self.client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            self.client.settimeout(timeout / 1000)
            socket_path = self._get_socket_path(slot)
            self.client.connect(socket_path)
        self.stream = self.client.makefile('rwb')

    def _get_socket_path(self, slot):
        target_name = "rpcs3"
        if sys.platform == 'darwin':
            tmp_dir = os.environ.get('TMPDIR', '/tmp')
        else:
            tmp_dir = os.environ.get('XDG_RUNTIME_DIR', '/tmp')

        if slot >= 28000 and slot <= 30000 and target_name == "pcsx2":
            return os.path.join(tmp_dir, f"{target_name}.sock")
        else:
            return os.path.join(tmp_dir, f"{target_name}.sock.{slot}")

    def close(self):
        self.stream.close()
        self.client.close()

    def __del__(self):
        self.close()

    def mkcmd(self, opcode, *args):
        cmd_list = [opcode] + list(args)
        cmd_buf = struct.pack(f'<I{len(cmd_list)}B', len(cmd_list) + 4, *cmd_list)
        return cmd_buf

    def runcmd(self, cmd):
        self.stream.write(cmd)
        self.stream.flush()

    def read_header(self):
        header = self.stream.read(5)
        length, return_code = struct.unpack('<IB', header)
        return length, return_code

    def read8(self, address):
        cmd = self.mkcmd(OP_READ8, *struct.pack('<I', address))
        self.runcmd(cmd)
        if self.read_header()[1] != 0:
            raise IOError()
        val = struct.unpack('<B', self.stream.read(1))[0]
        return val

    def read16(self, address):
        cmd = self.mkcmd(OP_READ16, *struct.pack('<I', address))
        self.runcmd(cmd)
        if self.read_header()[1] != 0:
            raise IOError()
        val = struct.unpack('<H', self.stream.read(2))[0]
        return val

    def read32(self, address):
        cmd = self.mkcmd(OP_READ32, *struct.pack('<I', address))
        self.runcmd(cmd)
        if self.read_header()[1] != 0:
            raise IOError()
        val = struct.unpack('<I', self.stream.read(4))[0]
        return val

    def read64(self, address):
        cmd = self.mkcmd(OP_READ64, *struct.pack('<I', address))
        self.runcmd(cmd)
        if self.read_header()[1] != 0:
            raise IOError()
        val = struct.unpack('<Q', self.stream.read(8))[0]
        return val

    def write8(self, address, value):
        cmd = self.mkcmd(OP_WRITE8, *struct.pack('<IB', address, value))
        self.runcmd(cmd)
        if self.read_header()[1] != 0:
            raise IOError()

    def write16(self, address, value):
        cmd = self.mkcmd(OP_WRITE16, *struct.pack('<IH', address, value))
        self.runcmd(cmd)
        if self.read_header()[1] != 0:
            raise IOError()

    def write32(self, address, value):
        cmd = self.mkcmd(OP_WRITE32, *struct.pack('<II', address, value))
        self.runcmd(cmd)
        if self.read_header()[1] != 0:
            raise IOError()

    def write64(self, address, value):
        cmd = self.mkcmd(OP_WRITE64, *struct.pack('<IQ', address, value))
        self.runcmd(cmd)
        if self.read_header()[1] != 0:
            raise IOError()

    def server_version(self):
        cmd = self.mkcmd(OP_VERSION)
        self.runcmd(cmd)
        if self.read_header()[1] != 0:
            raise IOError()
        strlen = struct.unpack('<I', self.stream.read(4))[0]
        chars = self.stream.read(strlen)
        return chars.decode('utf-8').rstrip('\0')

    def save_state(self, state):
        cmd = self.mkcmd(OP_SAVESTATE, state)
        self.runcmd(cmd)
        if self.read_header()[1] != 0:
            raise IOError()

    def load_state(self, state):
        cmd = self.mkcmd(OP_LOADSTATE, state)
        self.runcmd(cmd)
        if self.read_header()[1] != 0:
            raise IOError()

    def game_title(self):
        cmd = self.mkcmd(OP_GAMETITLE)
        self.runcmd(cmd)
        if self.read_header()[1] != 0:
            raise IOError()
        strlen = struct.unpack('<I', self.stream.read(4))[0]
        chars = self.stream.read(strlen)
        return chars.decode('utf-8').rstrip('\0')

    def game_id(self):
        cmd = self.mkcmd(OP_GAMEID)
        self.runcmd(cmd)
        if self.read_header()[1] != 0:
            raise IOError()
        strlen = struct.unpack('<I', self.stream.read(4))[0]
        chars = self.stream.read(strlen)
        return chars.decode('utf-8').rstrip('\0')

    def game_uuid(self):
        cmd = self.mkcmd(OP_GAMEUUID)
        self.runcmd(cmd)
        if self.read_header()[1] != 0:
            raise IOError()
        strlen = struct.unpack('<I', self.stream.read(4))[0]
        chars = self.stream.read(strlen)
        return chars.decode('utf-8').rstrip('\0')

    def game_version(self):
        cmd = self.mkcmd(OP_GAMEVERSION)
        self.runcmd(cmd)
        if self.read_header()[1] != 0:
            raise IOError()
        strlen = struct.unpack('<I', self.stream.read(4))[0]
        chars = self.stream.read(strlen)
        return chars.decode('utf-8').rstrip('\0')

    def status(self):
        cmd = self.mkcmd(OP_EMUSTATUS)
        self.runcmd(cmd)
        if self.read_header()[1] != 0:
            raise IOError()
        val = struct.unpack('<I', self.stream.read(4))[0]
        return EmulatorStatus(val)

    def read(self, address, size):
        cmd_list = []
        for i in range(size):
            cmd_list.append(OP_READ8)
            cmd_list.extend(struct.pack('<I', address + i))
        cmd_buf = struct.pack(f'<I{len(cmd_list)}B', len(cmd_list) + 4, *cmd_list)
        self.runcmd(cmd_buf)
        self.read_header()
        return self.stream.read(size)

    def write(self, address, bytes_data):
        cmd_list = []
        for i, byte in enumerate(bytes_data):
            cmd_list.append(OP_WRITE8)
            cmd_list.extend(struct.pack('<IB', address + i, byte))
        cmd_buf = struct.pack(f'<I{len(cmd_list)}B', len(cmd_list) + 4, *cmd_list)
        self.runcmd(cmd_buf)
        self.read_header()


# if __name__ == '__main__':
#     # Connect to RPCS3
#     pine = PINE(28012)
#
#     def oClass_to_mClass(oclass):
#         return pine.read8(0xa354c0 + oclass)
#
#     def mClass_for_oClass(oclass):
#         return pine.read32(0xa34c00 + oClass_to_mClass(oclass) * 4)
#
#     def give_bolts(amount):
#         pine.write32(0x969CA0, amount)
#
#     # Print game title
#     print(pine.game_title())
#     print("wow")