import os
import pandas as pd
from sklearn.datasets import fetch_openml


def download_and_save_openml(dataset_name: str, version: int, target_column: str, save_dir: str):
    print(f"Downloading {dataset_name}...")
    dataset = fetch_openml(dataset_name, version=version, as_frame=True)
    
    folder_path = os.path.join(save_dir, dataset_name)
    os.makedirs(folder_path, exist_ok=True)

    # Save features and target separately
    dataset.data.to_csv(os.path.join(folder_path, "features.csv"), index=False)
    dataset.target.to_frame(name=target_column).to_csv(os.path.join(folder_path, "labels.csv"), index=False)

    print(f"Saved to {folder_path}")
    return dataset


def downloadMNIST(path: str = "data/raw"):
    return download_and_save_openml("mnist_784", version=1, target_column="class", save_dir=path)


def downloadFashionMNIST(path: str = "data/raw"):
    return download_and_save_openml("Fashion-MNIST", version=1, target_column="class", save_dir=path)


if __name__ == "__main__":
    downloadFashionMNIST()
    downloadMNIST()
