"""Route4-ex 的包执行入口。

本文件只负责把：

- `python -m qrng_routes.route4_ex`

转发到 `main.py` 中的 `main()`。它不承载任何算法逻辑，仅用于包级 CLI 启动。
"""

from .main import main


if __name__ == "__main__":
    main()
