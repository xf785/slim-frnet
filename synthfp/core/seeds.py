"""确定性种子派生 —— 可复现性的基石。

设计文档要求「配置 + 种子 + 版本 + 校验和」四要素缺一不可。
本模块只做一件事：把任意层级的标识字符串稳定地映射为 64 位种子。

特点：
* 纯函数，跨进程 / 跨平台 / 跨 Python 版本一致（SHA-256，不依赖 ``hash()``）。
* 层级式：``master_seed(seed, dataset, identity, finger)`` 派生出主指纹种子，
  再由主指纹种子派生出每个捺印、每个 TTA 样本的种子。
* 与 ``numpy.random.SeedSequence`` 兼容，可直接喂给
  ``np.random.default_rng``。
"""

from __future__ import annotations

import hashlib
from typing import Iterable, Sequence

__all__ = ["stable_seed", "derive", "SeedTree"]

_MASK64 = (1 << 64) - 1


def stable_seed(*parts: object) -> int:
    """把任意数量的标识片段稳定映射为 ``[0, 2**64)`` 的整数种子。

    >>> stable_seed("ds", 42, "000001", "left_thumb") == stable_seed("ds", 42, "000001", "left_thumb")
    True
    >>> stable_seed("a") != stable_seed("b")
    True
    """
    payload = "\x1f".join(str(p) for p in parts).encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], "big") & _MASK64


def derive(base_seed: int, *parts: object) -> int:
    """在已有种子上继续派生（等价于 ``stable_seed`` 但强制带上父种子）。"""
    return stable_seed(base_seed, *parts)


class SeedTree:
    """层级种子树：``dataset -> identity -> finger -> impression -> tta``。

    所有节点种子都是父节点种子 + 节点标识的 SHA-256 派生，
    因此只要记住 dataset 级别的根种子，整棵树都可精确重建。
    """

    __slots__ = ("_root", "_dataset")

    def __init__(self, root_seed: int, dataset: str = "") -> None:
        self._root = int(root_seed)
        self._dataset = dataset

    # -- 根 ---------------------------------------------------------------
    @property
    def root(self) -> int:
        return self._root

    def dataset_seed(self) -> int:
        return derive(self._root, "dataset", self._dataset)

    # -- 层级 -------------------------------------------------------------
    def identity_seed(self, identity_id: str) -> int:
        return derive(self.dataset_seed(), "identity", identity_id)

    def finger_seed(self, identity_id: str, finger: str) -> int:
        return derive(self.identity_seed(identity_id), "finger", finger)

    def impression_seed(self, identity_id: str, finger: str, impression: int) -> int:
        return derive(self.finger_seed(identity_id, finger), "impression", impression)

    def tta_seed(self, identity_id: str, finger: str, impression: int, tta: int) -> int:
        return derive(self.impression_seed(identity_id, finger, impression), "tta", tta)

    # -- 通用 -------------------------------------------------------------
    def node(self, *parts: object) -> int:
        """任意子节点：``tree.node("stage", name, index)``。"""
        return derive(self.dataset_seed(), *parts)


def as_sequence(seed: int) -> Sequence[int]:
    """把单个 64 位种子转成 ``SeedSequence`` 可用的整数序列。"""
    return (int(seed) & _MASK64,)


def mix(seeds: Iterable[int]) -> int:
    """把多个种子混合为一个种子（用于组合多个随机源）。"""
    acc = 0
    for s in seeds:
        acc = stable_seed(acc, s)
    return acc
