from __future__ import annotations

from typing import List, Dict, Tuple, Any, Callable
from warnings import warn

import torch
from torch import Tensor, nn
from torch.nn import Module
from torch.utils.data import DataLoader
from collections import OrderedDict
from flwr.client import Client

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .fl_subscriber import FlowerSubscriber


class MnistClassifier(Module):
    """This class defines a basic convolutional neural nework"""

    def __init__(self):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, padding=3),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=3),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=3),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=3),
            nn.LeakyReLU(),
        )
        self.flatten = nn.Flatten(start_dim=-3, end_dim=-1)
        self.mlp = nn.Sequential(
            nn.LazyLinear(out_features=32),
            nn.LeakyReLU(),
            nn.Linear(in_features=32, out_features=10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float32)
        x = self.conv(x)
        x = self.flatten(x)
        x = self.mlp(x)
        return x


class RosClient(Client):
    flower_hook_t = Callable[[str], None]

    def __init__(
        self, cid: int, subscriber_node: FlowerSubscriber, entries_per_package: int
    ):
        super.__init__(self)
        self.cid: int = cid
        self.net: Module = MnistClassifier()
        self.entries_per_package: int = entries_per_package
        self.subscriber_node: FlowerSubscriber = subscriber_node
        self.training_data_dirs: List[str] = []
        self.subscriber_node.set_flower_hook(
            lambda training_data: self.training_data_dirs.append(training_data)
        )

        def get_device() -> str:
            device = "cpu"
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
                device = "mps"
            return device

        self.device = get_device()

    def _get_train_dataloader(self, config: Dict[str, Any]) -> DataLoader:
        """Generates the train set dataloader from currently received data

        Args:
            config: Dict[str, Any] Currently we support 2 keys:
                    batch_size: the training batch size
                    sample_cnt: the number of sample we use for training.
                                We always train on the freshest data, which may
                                be overlapped between iterations
        --------
        Returns:
            An iterable torch DataLoader object to be passed to the training function
        """
        sample_cnt = config["sample_cnt"]
        batch_size = config["batch_size"]

        trainset = []
        for entry_path in self.training_data_dirs.reverse():
            package: List[Tuple[Tensor, Tensor]] = torch.load(entry_path)
            trainset.append(package)
            if len(trainset >= sample_cnt):
                break

        if len(len(trainset) < sample_cnt):
            warn(
                f"Client does not currently have enough to train on,"
                f"expecting {sample_cnt} but got {len(trainset)}"
            )

        return DataLoader(trainset, batch_size=batch_size)

    def fit(self, config: Dict[str, Any]) -> Module:
        net = self.net
        net.to(self.device)
        train_loader = self._get_train_dataloader(config)
        self._train(net, train_dataloader=train_loader, config=config)
        return net

    def _train(self, net: Module, train_dataloader: DataLoader, config: Dict[str, Any]):
        # Train the Classifier
        torch.manual_seed(0)
        net = MnistClassifier()
        cross_entropy_loss = nn.CrossEntropyLoss()
        optim = torch.optim.Adam(net.parameters())

        losses = []
        for epoch in range(10):
            c = 0
            for batch in train_dataloader:
                c += 1
                if c > 50:
                    break
                X, y = batch["image"], batch["label"]
                # Normalisation step
                X = (X / 256).unsqueeze(-3)
                # Predict Labels
                py = net(X)
                # Loss
                loss = cross_entropy_loss(py, y)
                # Step
                optim.zero_grad()
                loss.backward()
                losses.append(loss.item())
                optim.step()

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.net.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.net.load_state_dict(state_dict, strict=True)
