# Helga
Helga is a reinforcement learning gym environment for the original Ratchet & Clank HD trilogy on PS3.

## Supported games
- Ratchet & Clank 1 (PAL) - NPEA00385
- Ratchet & Clank 3: Up Your Arsenal (PAL) - NPEA00387

## Requirements
- Windows
- RPCS3
- A legal copy of the PAL version of the game already installed in RPCS3
- At least Python 3.8
- Redis installed and running

## How to start training your agent
Install the Python requirements:
```shell
pip install -r requirements.txt
```

## Redis
Helga uses Redis to communicate between workers and the main learner process. The way the code is structured, Redis is a
hard requirement right now.  

You can have Redis installed locally on your machine, or you can use a different machine on the network for it. It's 
probably best not to use a cloud service for this, as the learner and worker communicate a lot, and the transfers might
be too slow.

### Redis on Windows
### Official Redis documentation
[Redis' official documentation on how to install Redis on Windows](https://redis.io/docs/latest/operate/oss_and_stack/install/install-redis/install-redis-on-windows/) 
involves using WSL or Docker. 

### Unofficial Redis Windows port
If you don't want to use WSL or Docker, you can use an unofficial Windows port of Redis. [This is the one that I use](https://github.com/tporadowski/redis),
and it has been working fine. 

## WandB
You can use Weights and Biases to track your training progress. Just follow their [quickstart guide here](https://docs.wandb.ai/quickstart)
until you're logged in. 

Add `--wandb` to the command line arguments to enable Weights and Biases tracking.

## Contributors
@davubnub on Discord for the Ratchet faces