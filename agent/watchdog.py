import threading
import time


class Watchdog:
    """
    Watches the environment to make sure it hasn't stalled. In such a case, it will restart the environment and/or
        RPCS3.
    """
    def __init__(self, env):
        self.env = env

        self.last_frame_count = 0
        self.last_frame_count_time = 0

    def start(self):
        # If we're running in PyCharm debug mode, don't start the watchdog
        import sys
        if "pydevd" in sys.modules:
            print("Watchdog: Not starting watchdog because we're running in PyCharm debug mode.")
            return

        thread = threading.Thread(target=self.run, args=())
        thread.daemon = True
        thread.start()

    def run(self):
        if self.env is None:
            return

        while True:
            self.last_frame_count_time += 1

            # Get RPCS3 from the process list
            # import psutil
            rpcs3 = None
            # for proc in psutil.process_iter():
            #     if proc.name() == "rpcs3.exe":
            #         rpcs3 = proc
            #         break

            if (self.last_frame_count == self.env.last_frame_count and self.last_frame_count_time > 15):
                # RPCS3 has likely crashed, restart it
                print("Watchdog: Environment has stalled, restarting it...")

                # Try to kill RPCS3 first using cmd.exe
                import subprocess
                subprocess.call("taskkill /IM rpcs3.exe /F")

                # Start RPCS3 again
                ## We need to run this command: "C:\Users\Vetle Hjelle\Applications\rpcs3-v0.0.15-12160-86a8e071_win64\rpcs3.exe" C:\StupidProjects\rac1-agent\build\PS3_GAME\
                subprocess.Popen([r"C:\Users\Vetle Hjelle\Applications\rpcs3-v0.0.15-12160-86a8e071_win64\rpcs3.exe", r"C:\StupidProjects\rac1-agent\build\PS3_GAME", "--no-gui"])

                # Signal to environment that it should restart and re-attach to RPCS3
                self.env.must_restart = True

                self.last_frame_count = 0
                self.last_frame_count_time = 0
            elif self.last_frame_count != self.env.last_frame_count:
                self.last_frame_count = self.env.last_frame_count
                self.last_frame_count_time = 0

            time.sleep(1)