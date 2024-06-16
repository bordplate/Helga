import time


class RatchetEnvironment:
    def __init__(self, device="cpu"):
        self.reward_counters = {}

        self.stats = {}

        self.device = device

    def start(self):
        process_opened = self.game.open_process()
        while not process_opened:
            print("Waiting for process to open...")
            time.sleep(1)
            process_opened = self.game.open_process()

    def stop(self):
        self.game.close_process()

    def reward(self, name: str, value: float):
        key = f"rewards/{name}"

        if key not in self.reward_counters:
            self.reward_counters[key] = value
            return value

        self.reward_counters[key] += value

        return value

    def stat(self, name: str, value: {}) -> {}:
        key = f"stats/{name}"

        if key not in self.stats:
            self.stats[key] = value
            return value

        self.stats[key] = value

        return value