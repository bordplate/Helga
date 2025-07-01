import threading
import time
import os
import ctypes
import subprocess

from Game.RC1Game import RC1Game


class Watchdog:
    """
    Watches the environment to make sure it hasn't stalled. In such a case, it will restart the environment and/or
        RPCS3.
    """
    def __init__(self,
                 game_path: str = r"../games/rc1/build/PS3_GAME",
                 render: bool = True
                 ):
        self.game_path = game_path
        self.render = render

        self.last_frame_count = 0
        self.last_frame_count_time = 0

    def start(self, force=False):
        # If we're running in PyCharm debug mode, don't start the watchdog, unless --force-watchdog is passed
        import sys
        # if "pydevd" in sys.modules and "--force-watchdog" not in sys.argv:
        #     print("Watchdog: Not starting watchdog because we're running in PyCharm debug mode.")
        #     return

        print("Starting RPCS3...")
        import subprocess

        config_file = "../rpcs3-config.yml" if not self.render else "../eval-config.yml"

        # Generate a unique process name for the RPCS3 process
        process_name = f"rpcs3-{time.time()}"

        if self.render:
            process_name = f"{process_name}-eval"

        # If Linux, open with bash
        if os.name == "posix":
            process = subprocess.Popen([
                    rf"/bin/bash", "-c",
                    f"exec -a {process_name} /Applications/RPCS3.app/Contents/MacOS/rpcs3 --no-gui --config {config_file} {self.game_path}",
                    # f"exec -a {process_name} /usr/bin/rpcs3 --no-gui --config {config_file} {self.game_path}",
                ],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
        elif os.name == "nt":
            process = subprocess.Popen([
                    rf"C:\Program Files\RPCS3\rpcs3.exe",
                    self.game_path,
                    "--no-gui",
                    "--config", config_file
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

        time.sleep(10)

        # thread = threading.Thread(target=self.run, args=())
        # thread.daemon = True
        # thread.start()

        return True

    def run(self):
        if self.env is None:
            return

        while True:
            self.last_frame_count_time += 1

            if self.last_frame_count == self.env.last_frame_count and self.last_frame_count_time > 60 and False:  # Disabled
                # RPCS3 has likely crashed, restart it
                print("Watchdog: Environment has stalled, restarting it...")

                # Try to kill RPCS3 first using cmd.exe
                import subprocess
                subprocess.call(f"taskkill /IM {self.process_name} /F")

                # Start RPCS3 again
                subprocess.Popen([
                    rf"{self.rpcs3_path}\{self.process_name}",
                    self.game_path,
                    "--no-gui",
                    "--headless" if not self.render else ""]
                )

                # Signal to environment that it should restart and re-attach to RPCS3
                self.env.must_restart = True

                self.last_frame_count = 0
                self.last_frame_count_time = 0
            elif self.last_frame_count != self.env.last_frame_count:
                self.last_frame_count = self.env.last_frame_count
                self.last_frame_count_time = 0

            time.sleep(1)
