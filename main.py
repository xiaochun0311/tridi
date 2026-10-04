import os

# Pin to E-cores (16-23) BEFORE importing torch: torch 2.9.0+cu128 segfaults
# (exit 139) on i9-14900K P-cores. Equivalent to `taskset -c 16-23`.
try:
    os.sched_setaffinity(0, set(range(16, 24)))
except (OSError, AttributeError):
    pass

import torch

from config.config import ProjectConfig
from tridi.core.evaluator import Evaluator
from tridi.core.sampler import Sampler
from tridi.core.trainer import Trainer
from tridi.model import get_model
from tridi.utils import training as training_utils
from tridi.utils.exp import init_exp, init_wandb, init_logging, parse_arguments


def main():  # 定义程序入口，根据配置选择训练、生成或评估。
    torch.multiprocessing.set_sharing_strategy('file_system')  # 设置多进程共享 CPU 张量的方式，供数据加载等环节使用。
    torch.set_float32_matmul_precision('high')  # 允许 float32 矩阵乘法使用支持的加速方式。

    # Parse arguments
    arguments = parse_arguments()  # 读取命令行传入的配置文件和覆盖参数。

    # Initialzie run
    cfg: ProjectConfig = init_exp(arguments)  # 合并配置并初始化本次运行；cfg 保存最终配置， 是完整的模型配置表。

    # Logging
    init_logging(cfg)  # 设置日志，记录程序运行信息。
    if cfg.logging.wandb:  # 判断是否启用 WandB 实验记录服务。用于记录实验数据的工具，可以可视化。
        init_wandb(cfg)  # 初始化本次运行的 WandB 记录。

    # Set random seed
    training_utils.set_seed(cfg.run.seed)  # 设置随机种子，方便复现实验结果。

    if cfg.run.job in ['train', 'sample']:  # 训练和生成任务都需要先创建模型。
        # Model
        model = get_model(cfg)  # 根据配置创建 TriDi 模型的网络结构。

        if cfg.run.job == 'train':  # 如果任务是训练，进入这个分支。
            trainer = Trainer(cfg, model)  # 创建训练器，准备模型、数据和优化器。

            trainer.train()  # 开始训练，通过数据更新模型权重。
        elif cfg.run.job == 'sample':  # 如果任务是生成，进入这个分支。
            sampler = Sampler(cfg, model)  # 创建采样器，加载模型权重并准备数据等资源。
            if cfg.sample.target == 'meshes':  # 判断是否将结果保存为三维网格。
                sampler.sample()  # 运行扩散采样并导出网格结果。
            elif cfg.sample.target == 'hdf5':  # 判断是否将结果保存为 HDF5 数据文件。
                sampler.sample_to_hdf5()  # 运行扩散采样并将结果写入 HDF5。
            else:  # 处理不支持的输出格式。
                raise ValueError(f"Invalid target {cfg.sample.target}")  # 抛出异常，指出无效的输出格式并停止运行。
    elif cfg.run.job == 'eval':  # 如果任务是评估，进入这个分支。
        evaluator = Evaluator(cfg)  # 创建评估器，准备评估所需的资源。
        evaluator.evaluate()  # 计算生成结果的评估指标。
    else:  # 处理不支持的任务类型。
        raise ValueError(f"Invalid job type {cfg.run.job}")  # 抛出异常，指出无效的任务类型并停止运行。


if __name__ == '__main__':
    main()
