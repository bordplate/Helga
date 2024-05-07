import time


class RatchetEnvironment:
    def start(self):
        process_opened = self.game.open_process()
        while not process_opened:
            print("Waiting for process to open...")
            time.sleep(1)
            process_opened = self.game.open_process()

    def stop(self):
        self.game.close_process()
