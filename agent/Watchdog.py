import threading
import time

from Game.RC1Game import RC1Game


class Watchdog:
    """
    Watches the environment to make sure it hasn't stalled. In such a case, it will restart the environment and/or
        RPCS3.
    """
    def __init__(self,
                 env: RC1Game,
                 process_name: str="rpcs3.exe",
                 rpcs3_path: str = "C:\\Users\\Vetle Hjelle\\Applications\\rpcs3-v0.0.15-12160-86a8e071_win64\\",
                 game_path: str = r"..\games\rc1\build\PS3_GAME",
                 render: bool = True
                 ):
        self.env = env
        self.process_name = process_name
        self.rpcs3_path = rpcs3_path
        self.game_path = game_path
        self.render = render

        self.env.process.process_name = process_name

        self.last_frame_count = 0
        self.last_frame_count_time = 0

    def start(self):
        # If we're running in PyCharm debug mode, don't start the watchdog, unless --force-watchdog is passed
        import sys
        if "pydevd" in sys.modules and "--force-watchdog" not in sys.argv:
            print("Watchdog: Not starting watchdog because we're running in PyCharm debug mode.")
            return

        # Check if the process is running, if not, we run it
        import psutil
        if not any(process.name() == self.process_name for process in psutil.process_iter()):
            print("Watchdog: RPCS3 is not running, starting it...")
            import subprocess
            subprocess.Popen([
                rf"{self.rpcs3_path}\{self.process_name}",
                self.game_path,
                "--no-gui",
                "--headless" if not self.render else ""]
            )

            time.sleep(10)

        thread = threading.Thread(target=self.run, args=())
        thread.daemon = True
        thread.start()

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
