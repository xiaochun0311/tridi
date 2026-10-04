"""查看可信 PKL 文件的字段、形状，以及可选的指定字段内容。"""

import argparse
import pickle
from pathlib import Path
from pprint import pprint


DEFAULT_PATH = (
    Path(__file__).resolve().parent
    / "data/raw/grab/grab_smplh/s1/banana_pass_1/sequence_data.pkl"
)


def print_structure(value, name="data", indent=0):
    prefix = " " * indent + name
    if isinstance(value, dict):
        print(f"{prefix}: dict ({len(value)} 个字段)")
        for key, item in value.items():
            print_structure(item, str(key), indent + 2)
    elif hasattr(value, "shape"):
        print(
            f"{prefix}: {type(value).__name__}, "
            f"shape={tuple(value.shape)}, dtype={value.dtype}"
        )
    elif isinstance(value, (list, tuple)):
        print(f"{prefix}: {type(value).__name__}, length={len(value)}")
    else:
        print(f"{prefix}: {value!r}")


def main():
    parser = argparse.ArgumentParser(
        description="查看 PKL 数据结构。仅打开可信文件，pickle 加载可能执行代码。"
    )
    parser.add_argument(
        "path", nargs="?", type=Path, default=DEFAULT_PATH,
        help="PKL 文件或包含 sequence_data.pkl 的目录",
    )
    parser.add_argument("--key", help="查看指定字段，例如 body.transl")
    parser.add_argument("--frame", type=int, help="查看指定字段的第几帧，从 0 开始")
    args = parser.parse_args()
    if args.frame is not None and not args.key:
        parser.error("--frame 需要同时指定 --key，例如 --key body.transl --frame 0")

    path = args.path.expanduser()
    if path.is_dir():
        path = path / "sequence_data.pkl"
    try:
        with path.open("rb") as file:
            data = pickle.load(file)
    except ModuleNotFoundError as error:
        parser.exit(
            1, f"缺少依赖 {error.name}，请在安装了 NumPy 和 PyTorch 的项目环境中运行。\n"
        )
    except (OSError, pickle.UnpicklingError, EOFError) as error:
        parser.exit(1, f"无法读取 {path}: {error}\n")

    print(f"文件: {path.resolve()}\n")
    print_structure(data)
    if args.key:
        value = data
        try:
            for key in args.key.split("."):
                value = value[key]
            if args.frame is not None:
                value = value[args.frame]
        except (KeyError, IndexError, TypeError) as error:
            parser.exit(1, f"无法访问字段或帧: {error}\n")
        print(f"\n{args.key}" + (f"[{args.frame}]" if args.frame is not None else "") + ":")
        pprint(value)


if __name__ == "__main__":
    main()
