import _thread
import threading
import time
import os
import ctypes
import subprocess

import signal

import psutil

from Game.RC1Game import RC1Game

# Define the prctl function from the C library
#libc = ctypes.CDLL('libc.so.6')
#PR_SET_NAME = 15


def set_pdeathsig(sig=signal.SIGTERM):
    libc = ctypes.CDLL("libc.so.6")
    libc.prctl(1, sig)  # PR_SET_PDEATHSIG = 1


class Watchdog:
    """
    Watches the environment to make sure it hasn't stalled. In such a case, it will restart the environment and/or
        RPCS3.
    """
    def __init__(self,
                 game_path: str = r"./games/rc3/build/PS3_GAME",
                 render: bool = True
                 ):
        self.game_path = game_path
        self.render = render
        self.config_file = None

        self.last_frame_count = 0
        self.last_frame_count_time = 0

        self.process_name = None

    def start(self, force=False):
        # If we're running in PyCharm debug mode, don't start the watchdog, unless --force-watchdog is passed
        import sys
        if "pydevd" in sys.modules and "--force-watchdog" not in sys.argv:
            print("Watchdog: Not starting watchdog because we're running in PyCharm debug mode.")
            return False

        print("Starting RPCS3...")
        import subprocess

        self.config_file = "./rpcs3-config.yml" if not self.render else "./eval-config.yml"

        # Generate a unique process name for the RPCS3 process
        self.process_name = f"rpcs3-{time.time()}"

        if self.render:
            self.process_name = f"{self.process_name}-eval"

        # If Linux, open with bash
        if os.name == "posix":
            process = subprocess.Popen([
                    rf"/bin/bash", "-c",
                    f"exec -a {self.process_name} /usr/bin/rpcs3 --no-gui --config {self.config_file} {self.game_path}",
                ],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                preexec_fn=set_pdeathsig
            )

            self.rpcs3_path = "/usr/bin/rpcs3"
        elif os.name == "nt":
            rpcs3_paths = [
                rf"C:\rpcs3\rpcs3-test\rpcs3.exe",
                rf"C:\rpcs3\rpcs3-test2\rpcs3-test2.exe"
            ]

            rpcs3_path = None

            for path in rpcs3_paths:
                exe_name = path.split("\\")[-1]

                # Check if this exe is already running
                if exe_name in [p.name() for p in psutil.process_iter()]:
                    continue

                rpcs3_path = path

                break

            if rpcs3_path is None:
                raise Exception("Can't find a free RPCS3 instance to run.")

            self.rpcs3_path = rpcs3_path

            process = subprocess.Popen([
                    rpcs3_path,
                    self.game_path,
                    "--no-gui"
                ],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
        else:
            print("Watchdog: Unsupported OS.")
            return False

        # Get the PID of the process
        self.pid = process.pid

        # Change the name of the running process so we can distinguish it from

        time.sleep(1)

        # thread = threading.Thread(target=self.run, args=())
        # thread.daemon = True
        # thread.start()

        return True

    def watch(self, env):
        self.env = env

        print(f"Watching environment for stalls and crashes...")

        # Make a new thread to watch the environment
        # thread = threading.Thread(target=self.run, args=())
        # thread.daemon = True
        # thread.start()

    def run(self):
        if self.env is None:
            raise Exception

        # stalled_time = 0
        # last_total_steps = 0
        #
        # while True:
        #     if last_total_steps == self.env.total_steps:
        #         stalled_time += 1
        #     else:
        #         stalled_time = 0
        #
        #     last_total_steps = self.env.total_steps
        #
        #     if stalled_time > 120 and self.env.game.process.process is not None:
        #         stalled_time = 0
        #
        #         # RPCS3 has likely crashed, restart it
        #         print("Watchdog: Environment has stalled, restarting it...")
        #
        #         self.env.game.process.process = None
        #
        #         # Try to kill RPCS3 first
        #         if os.name == "posix":
        #             os.system(f"kill {self.pid}")
        #         elif os.name == "nt":
        #             os.system(f"taskkill /F /PID {self.pid}")
        #
        #         _thread.interrupt_main()
        #         exit(-1)
        #
        #         # Start RPCS3 again
        #         if os.name == "posix":
        #             process = subprocess.Popen([
        #                 rf"/bin/bash", "-c",
        #                 f"exec -a {self.process_name} /usr/bin/rpcs3 --config {self.config_file} {self.game_path}",
        #                 # f"exec -a {self.process_name} /usr/bin/rpcs3 --no-gui --config {config_file} {self.game_path}",
        #             ],
        #                 stdin=subprocess.DEVNULL,
        #                 stdout=subprocess.DEVNULL,
        #                 stderr=subprocess.DEVNULL
        #             )
        #         elif os.name == "nt":
        #             process = subprocess.Popen([
        #                 self.rpcs3_path,
        #                 self.game_path,
        #                 "--no-gui"
        #             ],
        #                 stdin=subprocess.DEVNULL,
        #                 stdout=subprocess.DEVNULL,
        #                 stderr=subprocess.DEVNULL
        #             )
        #         else:
        #             raise Exception("Watchdog: Unsupported OS")
        #
        #         self.pid = process.pid
        #
        #         # Signal to environment that it should restart and re-attach to RPCS3
        #         self.env.game.process.pid = self.pid
        #         # self.env.game.process.open_process()
        #         self.env.must_restart = True
        #
        #         self.last_frame_count = 0
        #         self.last_frame_count_time = 0
        #
        #     time.sleep(1)
