"""pytest 公共 fixture。"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


# ---------------------------------------------------------------------------
# 受限沙箱下的临时目录适配
# ---------------------------------------------------------------------------
# pytest 的 ``tmp_path`` 会以 ``mode=0o700`` 创建 ``pytest-of-<user>/pytest-N``。
# 某些受限沙箱（Windows 上以受限令牌运行的自动化环境）对 0o700 目录连创建者
# 自己都拒绝 ``scandir``，于是所有用到 ``tmp_path`` 的测试在 setup 阶段就
# ``PermissionError`` ——这不是测试或被测代码的问题。
#
# 这里只在**探测到该限制**时，把「用 0o700 建目录」改写为 0o777。影响面被
# 严格限制在本次 pytest 进程内，且不改动任何测试语义（路径、命名、唯一性、
# 清理逻辑都不变）。普通环境探测通过后完全不做替换。
# ---------------------------------------------------------------------------


def _sandbox_denies_private_dirs() -> bool:
    """探测：能否正常读取一个以 0o700 创建的目录。"""
    probe = Path(tempfile.gettempdir()) / f"_synthfp_mode_probe_{os.getpid()}"
    created = False
    try:
        os.mkdir(probe, 0o700)
        created = True
        os.chmod(probe, 0o700)
        list(probe.iterdir())
        return False
    except OSError:
        return True
    finally:
        if created:
            try:
                os.chmod(probe, 0o777)
                os.rmdir(probe)
            except OSError:
                pass


def _permissive_mkdir(original):
    """把"没有其他用户权限位"的 mode 放宽为 0o777。"""

    def wrapper(path, mode=0o777, *, dir_fd=None):
        if not mode & 0o007:  # 0o700 / 0o750 这类
            mode = 0o777
        return original(path, mode, dir_fd=dir_fd)

    return wrapper


if _sandbox_denies_private_dirs():  # pragma: no cover - 取决于运行环境
    os.mkdir = _permissive_mkdir(os.mkdir)

    # 系统临时区里可能已经躺着上一轮遗留的 ``pytest-of-<user>``（0o700、不可读），
    # 而 pytest 会复用它。把临时根挪到工作区内的新目录，绕开这个污染。
    _WORK_TMP = _ROOT / ".pytest-work"
    try:
        _WORK_TMP.mkdir(parents=True, exist_ok=True)
        list(_WORK_TMP.iterdir())
        tempfile.tempdir = str(_WORK_TMP)
    except OSError:
        pass


@pytest.fixture(scope="session")
def rng() -> np.random.Generator:
    """确定性随机源。"""
    return np.random.default_rng(42)


@pytest.fixture(scope="session")
def small_size() -> tuple[int, int]:
    """小尺寸画布 (宽, 高)，用于快速测试。"""
    return (160, 240)


@pytest.fixture(scope="session")
def sine_image() -> np.ndarray:
    """理想正弦条纹（模拟完美脊线），float32 [0,1]。"""
    h, w = 256, 256
    xx = np.arange(w, dtype=np.float32)[None, :]
    return (0.5 + 0.45 * np.cos(2 * np.pi * xx / 10.0)).astype(np.float32) * np.ones(
        (h, 1), dtype=np.float32
    )


@pytest.fixture(scope="session")
def noise_image() -> np.ndarray:
    """均匀白噪声，float32 [0,1]。"""
    rng = np.random.default_rng(7)
    return rng.random((256, 256)).astype(np.float32)


@pytest.fixture(scope="session")
def ellipse_mask() -> np.ndarray:
    """椭圆手指掩模 uint8 {0,255}。"""
    import cv2

    mask = np.zeros((256, 256), dtype=np.uint8)
    cv2.ellipse(mask, (128, 128), (60, 100), 0, 0, 360, 255, -1)
    return mask
