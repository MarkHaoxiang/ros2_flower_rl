from typing import Tuple

from flwr.common.typing import Config
from florl.client.kitten.dqn import DQNKnowledge, DQNClientFactory
from gymnasium.spaces import Space

import kitten
from kitten.common.util import build_env
from kitten.policy import Policy
from kitten.rl import Algorithm
from kitten.rl.dqn import DQN

POLICY_SERVICE: str = "policy"
POLICY_UPDATE_TOPIC: str = "knowledge"
MEMORY_SERVICE: str = "memory"

DQN_SHARDS = ["critic", "critic_target"]

NUM_CLIENTS = 5
TOTAL_ROUNDS = 100
FRAMES_PER_ROUND = 50
EXPERIMENT_REPEATS = 20
SEED = 0

SERVER_ADDR = "[::]:8080"

config = {
    "rl": {
        "env": {
            "name": "CartPole-v1"
        },
        "algorithm": {
            "gamma": 0.99,
            "tau": 0.005,
            "lr": 0.001,
            "update_frequency": 1,
            "clip_grad_norm": 1,
            "critic": {
                "features": 64
            }
        },
        "memory": {
            "type": "experience_replay",
            "capacity": max(128, TOTAL_ROUNDS * FRAMES_PER_ROUND)
        },
        "train": {
            "initial_collection_size": 1024,
            "minibatch_size": 64
        }
    },
    "fl": {
        "train_config": {
            "frames": FRAMES_PER_ROUND,
        },
        "evaluate_config": {
            "evaluation_repeats": 1
        }
    }
}

client_factory = DQNClientFactory(config=config)
name = config["rl"]["env"]["name"]
env = build_env(name=name, seed=SEED)
action_space = env.action_space
default_knowledge_fn = lambda: DQNClientFactory(config).create_default_knowledge(config)

def build_dqn_algorithm(
    cfg: Config,
    knowledge: DQNKnowledge,
    action_space: Space,
    seed: int | None,
    device: str = "cpu",
) -> Tuple[Algorithm, Policy]:
    seed = seed if seed else 0
    rng = kitten.common.global_seed(seed).numpy
    cfg.get("algorithm", {}).pop("critic", None)  # type: ignore
    algorithm = DQN(
        critic=knowledge.critic.net,  # type: ignore
        device=device,
        **cfg.get("algorithm", {}),  # type: ignore
    )
    policy = kitten.policy.EpsilonGreedyPolicy(
        fn=algorithm.policy_fn,
        action_space=action_space,
        rng=rng,
        device=device,
    )
    # Synchronisation
    algorithm._critic = knowledge.critic  # type: ignore
    return algorithm, policy
