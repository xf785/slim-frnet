import sys
from pathlib import Path

# 让测试直接以源码树运行（无需 pip install -e .）
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
