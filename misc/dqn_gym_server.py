import flwr as fl
from flwr.server import start_server

from florl.server.strategy.rl_fedavg import RlFedAvg
from florl.client.kitten.dqn import DQNClientFactory
from florl.common.util import aggregate_weighted_average, stateful_client

from toy_frl.dqn.common import *
from copy import deepcopy

from omegaconf import OmegaConf, DictConfig

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

config = DictConfig(config)
train_config = OmegaConf.to_container(config["fl"]["train_config"])
evaluate_config = OmegaConf.to_container(config["fl"]["evaluate_config"])
default_knowledge = default_knowledge_fn()

def _on_fit_config_fn(server_round: int):
        return train_config | {"server_round": server_round}
def _on_evaluate_config_fn(server_round: int):
    return evaluate_config | {"server_round": server_round}

strategy = RlFedAvg(
    knowledge=deepcopy(default_knowledge),
    min_fit_clients=1,
    min_available_clients=1,
    on_fit_config_fn = _on_fit_config_fn,
    on_evaluate_config_fn= _on_evaluate_config_fn,
    fit_metrics_aggregation_fn=aggregate_weighted_average,
    evaluate_metrics_aggregation_fn=aggregate_weighted_average,
    accept_failures=True,
    inplace=False
)

server_config = fl.server.ServerConfig(num_rounds=10, )
def main():
    start_server(server_address="[::]:8080", strategy=strategy, config=server_config)

if __name__ == "__main__":
    main()
