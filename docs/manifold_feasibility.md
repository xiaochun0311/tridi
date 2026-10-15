# 七篇流形扩散论文与 TriDi：首轮可行性报告

日期：2026-10-12。实际仓库：`/home/yanghc/tridi`；分支：`manifold-test`；基线提交：`86e0e9de1c63acb87d41cba2540863f45ac0e04c`。

本轮完成七项顺序数值探针。论文 1、3、5 实现推理适配，论文 4 实现局部诊断；论文 2、6、7 目前验证表示、密度建模或指导组件的前提。后者尚未完成各论文的新模型训练或规划实验。本轮不能判定七篇完整方法均已复现或有效。

## 执行通道恢复

完整粘贴的管道、引号、下划线回显已通过 Python 断言。现有 SSH 连接与 `tridi-offline` 解释器可稳定执行实验；全部源文件与结果都写入上面的真实仓库。Trae 的逐键输入/焦点同步问题采用完整粘贴及 SSH 执行通道处理，没有修改 Trae 安全设置。

`.gitignore` 内容在实验前后校验一致（SHA256：`6d5bb16a095ca26b3f789de4214dfc46f3f996c72e5640e9c6092a8ef6cb01f2`）。本轮新增实验脚本和报告，训练模型源码没有改动。

## 实验范围与成功检查

- 官方 `assets/gb_main.pth` 严格加载成功；人体 325 维、物体 9 维、接触潜变量 128 维。
- 数据：`data/preprocessed/grab_smplh`。训练主体 s1：529 帧、10 fps；测试主体 s9：41 帧、1 fps。仅香蕉，包含 eat/lift/pass/peel 等动作，训练还有 offhand。相邻帧相关，不能视为独立统计样本。
- 原调试数据目录在首轮后已不存在；本报告使用现有正式预处理数据上的最终成功复跑结果。
- 随机种子：42、43、44。MAD 和平滑各 20 步；相同调用预算的 Euler 为 40 步。三者均含末端去噪，合计 41 次网络调用。
- 同种子共用 Euler 基线接触 mask，分别为 38、16、23 个顶点；另记录各方法自己的解码接触数量。这样接触距离不会通过改变 mask 人为改善。
- 接触距离为模型坐标单位，测量人体接触顶点到物体 1500 关键点的距离，不是穿透、真实连续表面距离或总体质量指标。原始 6D 偏差只反映冗余旋转坐标离 canonical 表示的距离；现有 decoder 已做 Gram–Schmidt。
- 数据序列的 prep_s 均为 1；代表帧 `s9/banana_eat_1/0` 的保存顶点与 MeshModel 重建 RMS 为 5.92e-08。
- 最终进程退出码 0，日志末尾为 `ALL_SEVEN_PILOTS_COMPLETE`；JSON 中 `completed=true`。PyTorch 峰值分配显存约 176.6 MB，未计其他进程和 CUDA 上下文。

## 七个方向的判断

| 论文 | 本轮做了什么 | 观察结果 | 当前判断 |
|---|---|---|---|
| 1 MAD | 论文 Algorithm 1 的 VP→VE 适配；固定 z 做不同噪声尺度的 score 差分，p=2、a=b=1 | 相同接触集合下平均距离 0.0583 → 0.1063（增加 82.4%）；原始 6D 偏差降低 | 接口能运行，当前未调参版本损害该接触指标，不能直接采用 |
| 2 Efficient Diffusion Models for Symmetric Manifolds | SO(3) Expmap 与普通 6D 扰动的表示检查及角度响应比较 | 内在扰动保留 canonical 旋转；普通 6D 扰动产生冗余偏差；现有 decoder 本身也能正交化 | 仅表示前提可行。原论文的 OU 投影、Itô 漂移/协方差及训练目标需要另行实现 |
| 3 Log-domain Smoothing | 对固定尺度 score 做 z±0.03v 对称平均，v 为随机单位方向 | 平均接触距离 0.0593；与 20 步普通 Euler 几乎相同 | 可接入，当前小扰动没有观察到明显收益；不能宣称几何自适应已证实 |
| 4 Local Geometry | 16 个正交输入方向的去噪 Jacobian 有限差分，h=0.002/0.001 检查 | 投影有效秩约 15.31 → 4.44；差分相对变化 0.50%–1.02% | 可用于诊断，建议先扩充沿真实采样轨迹的分析；这不是完整流形维数 |
| 5 Rate Separations | 固定偏移下的旋转切向/法向 score 导数；固定 σ 的标准/tempered Langevin 20 步 | 切向和法向导数均明显随 σ 变化，未观察到预期分离；tempered 的原始 6D 偏差 0.181 → 0.257 | 更新接口稳定，但当前局部指标不支持直接用它改采样；短链不能证明均匀分布 |
| 6 Multi-Subspace Multi-Modal | train-only 单 Gaussian、共享全局子空间 Gaussian mixture、局部子空间 mixture；s9 测试似然 | 单组件 -1578.6；共享基 2 组件 -1412.0；局部基 2 组件 -2384.9（越大越好） | 多模态前提值得继续；当前局部子空间迁移变差。尚未训练论文完整两层 MoE 扩散 |
| 7 STRiDE | clean-input yaw 一致性；相同 3 次梯度下的普通/旋转切向接触指导 | 普通指导 0.0760 → 0.0484；切向指导 → 0.0487，未优于普通指导 | 静态接触指导组件可借鉴；SMPL SO(3) 与机器人 torus 不同，当前没有轨迹/限位/规划验证 |

## 采样比较原始指标

| 方法 | 每种子网络调用次数 | 共同 mask 平均接触距离 ↓ | 原始 6D RMS ↓ |
|---|---:|---:|---:|
| Euler 40 步（调用预算基线） | 41 | 0.058252 | 0.014424 |
| Euler 20 步（离散步数控制） | 21 | 0.059297 | 0.013516 |
| 原始 DDIM 40 步（参考） | 40 | 0.058124 | 0.014519 |
| MAD 20 步 | 41 | 0.106267 | 0.011478 |
| score 平滑 20 步 | 41 | 0.059296 | 0.013516 |

Euler 20 步控制用于判断 score 平滑效果是否来自积分步数变化；该行使用更少网络调用。原始 DDIM 是原模型参考，其 VP 离散起点与 VE Euler 的 t999 起点不同，不能视为完全匹配的算法控制。3 个种子不足以作统计显著性结论。

MAD 在最大 σ 处采用向较小 σ 的边界差分，其余位置采用不同训练 timestep 的向前差分。参数 a=b=1 未拟合。TriDi 人体/物体的 x0 预测使用 L1，其最优预测对应条件中位数；VP→VE 的 Tweedie score 恒等式要求条件均值，因此当前 score 适配是近似。这同时限制论文 1、3、5 的理论解释。

第 4 项只计算混合参数坐标下的去噪网络投影谱，测量点为 √α x0，没有注入随机噪声；不能据此推断整个生成器流形的维度、体积或质量原因。第 5 项的切空间来自单个全局 SO(3) 旋转，并非完整交互数据流形。第 6 项使用参数列 10:334，排除人体 shape 与接触潜变量；每组件 rank=4，局部基引入额外参数，不能将似然变化归因于完整论文结构。第 7 项用嵌入空间切向投影与 canonical 6D retraction，未实现 STRiDE 的 wrapped torus diffusion 和规划策略。

优先顺序：先用第 4 项做模型诊断；第 6 项先扩展训练主体并检查共享子空间多模态；第 7 项保留普通接触指导作为控制；第 1/3/5 项需要参数或 score 目标校准后再判断。第 2 项完整方法需要独立训练实验。

## 复跑

在 Trae 现有远程终端内：

```bash
cd /home/yanghc/tridi
env CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2   /home/yanghc/miniconda3/envs/tridi-offline/bin/python -u scripts/manifold_feasibility.py
```

源代码：`scripts/manifold_feasibility.py`；逐项指标：`docs/manifold_feasibility_results.json`；完整日志：`docs/manifold_feasibility_run.log`。

脚本 SHA256：`d503def32cfe79c4eb49c41958856e3af96d8323ff7c1bc5f85d13a4eb564e5c`。上述结果来自当前脚本的最终成功运行。

## 对应论文

1. MAD Manifold Attracted Diffusion。
2. Efficient Diffusion Models for Symmetric Manifolds。
3. Diffusion Models and the Manifold Hypothesis: Log-Domain Smoothing is Geometry Adaptive。
4. What Secrets Do Your Manifolds Hold? Understanding the Local Geometry of Generative Models。
5. When Scores Learn Geometry: Rate Separations under the Manifold Hypothesis。
6. Multi-Subspace Multi-Modal Modeling for Diffusion Models。
7. STRiDE: STate-space Riemannian Diffusion for Equivariant Planning。

方法定义已核对用户提供的 PDF 方法页。没有依靠论文标题把普通后处理当作原文算法。

## 查看生成模型

种子42的完整人体与物体模型已导出到 `outputs/manifold_preview/seed42/`。`euler_*`、`mad_*`、`smooth_*` 分别对应同调用预算基线、MAD、score平滑；`ddim_*` 是原始采样器参考。每组含 human.ply 和 object.ply，应同时打开两者。

复跑时添加 `--preview-dir outputs/manifold_preview` 可重新导出。
