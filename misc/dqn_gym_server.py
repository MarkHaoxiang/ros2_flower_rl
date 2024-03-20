import flwr as fl
from flwr.server import start_server
from flwr.common import Parameters, EvaluateIns

from florl.server.strategy import FedAvg
from florl.common.util import aggregate_weighted_average

from toy_frl.dqn.common import config, default_knowledge_fn, client_factory
from copy import deepcopy

from omegaconf import OmegaConf, DictConfig

config = DictConfig(config)
train_config = OmegaConf.to_container(config["fl"]["train_config"])
evaluate_config = OmegaConf.to_container(config["fl"]["evaluate_config"])
default_knowledge = default_knowledge_fn()

evaluation_client = client_factory.create_client(0, config["rl"])
def evaluate(server_rounds: int, parameters: Parameters):
    ins = EvaluateIns(parameters=parameters, config={})
    evaluation_result = evaluation_client.evaluate(ins)
    return evaluation_result.loss, evaluation_result.metrics

def _on_fit_config_fn(server_round: int):
        return train_config | {"server_round": server_round}
def _on_evaluate_config_fn(server_round: int):
    return evaluate_config | {"server_round": server_round}

single_client_strategy = FedAvg(
    knowledge=deepcopy(default_knowledge),
    min_fit_clients=1,
    fraction_evaluate=0.0,
    min_available_clients=1,
    on_fit_config_fn = _on_fit_config_fn,
    on_evaluate_config_fn= _on_evaluate_config_fn,
    fit_metrics_aggregation_fn=aggregate_weighted_average,
    evaluate_metrics_aggregation_fn=aggregate_weighted_average,
    evaluate_fn=evaluate,
    accept_failures=True,
    inplace=False
)

server_config = fl.server.ServerConfig(num_rounds=20, )
def main():
    start_server(server_address="[::]:8080", strategy=single_client_strategy, config=server_config)

if __name__ == "__main__":
    main()
