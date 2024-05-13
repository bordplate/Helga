from Game.RC1Game import RC1Game

from RatchetEnvironment import RatchetEnvironment

class EmptyEnvironment(RatchetEnvironment):
    def reset(self):
        pass

    def step(self, action):
        pass


if __name__ == "__main__":
    env = EmptyEnvironment()
    env.game = RC1Game(process_name="rpcs3.exe")
    env.start()

    while True:
        env.step(None)

        position = env.game.get_player_position()

        input("")

        print(f"Vector3({position.x}, {position.y}, {position.z}), ", end="")



    env.stop()
