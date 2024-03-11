from datasets import load_dataset
import torch
from torch.utils.data import DataLoader
import flwr as fl
from flwr.common import NDArrays
from florl.common.util import aggregate_weighted_average, set_torch_parameters

from toy_fl.nn import MnistClassifier

DATASET = "mnist"


def _on_fit_config_fn(server_round: int):
    return {"server_round": server_round}


def _on_evaluate_config_fn(server_round: int):
    return {"server_round": server_round}


def main():
    # Load Data
    dataset = load_dataset(DATASET).with_format("torch")["test"]
    dataloader = DataLoader(dataset, batch_size=32)
    cross_entropy_loss = torch.nn.CrossEntropyLoss()

    net = MnistClassifier()

    def evaluate(server_round: int, parameters: NDArrays, config):
        set_torch_parameters(net, parameters)
        # Run evaluation round
        average_loss, correct = 0, 0
        size, num_batches = len(dataset), len(dataloader)
        for batch in dataloader:
            X, y = batch["image"], batch["label"]
            # Predict Labels
            py = net(X)
            # Loss
            average_loss += cross_entropy_loss(py, y).item()
            # Correct
            correct += (py.argmax(1) == y).type(torch.float).sum().item()
        average_loss = average_loss / num_batches
        accuracy = correct / size
        # Return results
        return average_loss, {"accuracy": accuracy}

    strategy = fl.server.strategy.FedAvg(
        on_fit_config_fn=_on_fit_config_fn,
        on_evaluate_config_fn=_on_evaluate_config_fn,
        fit_metrics_aggregation_fn=aggregate_weighted_average,
        evaluate_metrics_aggregation_fn=aggregate_weighted_average,
        evaluate_fn=evaluate,
        accept_failures=False,
    )

    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=10),
        strategy=strategy,
    )


if __name__ == "__main__":
    main()
