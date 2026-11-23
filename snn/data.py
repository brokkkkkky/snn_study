"""下载并加载 Keras 发布的 MNIST 数据，保留原有的训练集和测试集。"""

import hashlib
import shutil
import tempfile
from pathlib import Path
from urllib.request import urlopen

import numpy as np


MNIST_URL = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"
MNIST_SHA256 = "731c5ac602752760c8e48fbffcf8c3b850d9dc2a2aedcf2cc48468fc17b673d1"
DEFAULT_DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "mnist.npz"


def _verify_archive(path: Path) -> None:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != MNIST_SHA256:
        raise ValueError(f"MNIST 文件校验失败：{path}。请使用来自 {MNIST_URL} 的文件。")


def _download_mnist(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # 先写临时文件，确保下载中断时缓存目录里不会留下可被误用的半成品。
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as temp:
        temp_path = Path(temp.name)
    try:
        with urlopen(MNIST_URL, timeout=30) as response, temp_path.open("wb") as target:
            shutil.copyfileobj(response, target)
        _verify_archive(temp_path)
        temp_path.replace(path)
    finally:
        temp_path.unlink(missing_ok=True)


def load_mnist(path: Path | None = None):
    """返回 ((训练图像, 训练标签), (测试图像, 测试标签))，像素保留 uint8。"""
    if path is None:
        path = DEFAULT_DATA_PATH
        if not path.exists():
            print(f"正在下载 MNIST 到 {path}", flush=True)
            _download_mnist(path)
        else:
            _verify_archive(path)
    else:
        _verify_archive(path)

    with np.load(path, allow_pickle=False) as archive:
        train = archive["x_train"], archive["y_train"]
        test = archive["x_test"], archive["y_test"]
    return train, test
