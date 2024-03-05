import os, shutil

import rclpy
from rclpy.node import Node
from rclpy import logging
from rospkg import get_ros_home
from flwr_datasets import FederatedDataset


def generate_partition_path(dataset_dir, name, cid):
    return os.path.join(dataset_dir, f"{name}_{cid}.hf")


class DatasetDownloader(Node):

    def __init__(self):
        super().__init__("dataset_downloader")

        self.declare_parameter("experiment_dataset", "mnist")
        self.declare_parameter("n_partitions", 10)

        experiment_dataset = (
            self.get_parameter("experiment_dataset").get_parameter_value().string_value
        )
        n_partitions = (
            self.get_parameter("n_partitions").get_parameter_value().integer_value
        )

        dataset_dir = os.path.join(get_ros_home(), "fl_data")
        dataset_dir = os.path.join(dataset_dir, experiment_dataset)
        if not os.path.exists(dataset_dir):
            os.makedirs(dataset_dir)

        # Check if previously cached
        rebuild = False
        for i in range(n_partitions):
            if not os.path.exists(
                generate_partition_path(dataset_dir, experiment_dataset, i)
            ):
                rebuild = True
        if os.path.exists(
            generate_partition_path(dataset_dir, experiment_dataset, n_partitions)
        ):
            rebuild = True

        # Download and save dataset
        if rebuild:
            shutil.rmtree(dataset_dir)
            os.makedirs(dataset_dir)
            fds = FederatedDataset(
                dataset=experiment_dataset, partitioners={"train": n_partitions}
            )
            for i in range(n_partitions):
                partition = fds.load_partition(i, "train")
                partition.save_to_disk(
                    dataset_path=generate_partition_path(
                        dataset_dir, experiment_dataset, i
                    )
                )
        else:
            self.get_logger().info("No updates applied")

        # Shutdown
        raise SystemExit


def main(args=None):
    rclpy.init(args=args)
    node = DatasetDownloader()
    try:
        rclpy.spin(node)
    except SystemExit:
        logging.get_logger("Dataset").info("Dataset Preparation Complete")
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
