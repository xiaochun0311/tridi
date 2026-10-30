# HIMO 双物体、两帧接入

输入为两物体的初始姿态、规范几何和两个有顺序的动作；输出为S1、S2两帧人体、两物体姿态及逐物体接触。S0人体和真实未来姿态不作为生成条件。

本分支复用TriDi的Transformer核心、人体／物体／接触三分支、七类稀疏噪声时钟及x0预测。HIMO人体采用原生SMPL-X neutral和完整手指，接触改为对应SMPL-X的690顶点新头。12维几何描述是规范网格的extents、mean、absmean、std各3维，**不是PointNeXt**；此扩展不等同于原论文配置复现，也没有证明生成质量。

## 目录与环境

在服务器HIMO独立分支工作目录执行，`/home/yanghc/tridi`保留干净的`main`：

```bash
cd /home/yanghc/tridi-himo
TRIDI_PY=/home/yanghc/miniconda3/envs/tridi-offline/bin/python
```

`config/env.yaml`默认路径可直接使用：HIMO源数据位于`data/raw/himo/unpacked-20261029/data/`；人体资产为`data/smplx_models/smplx/SMPLX_NEUTRAL.npz`；TriDi初始化权重为`assets/gb_main.pth`。预处理输出为`data/preprocessed/himo_ordered.pth`和`himo_smplx_indices.npy`。

当前分支为`codex/himo-integration`；`data`、`outputs`与`assets/gb_main.pth`通过软链接复用主目录中的资源。

以下901—906是示例run名，应选未使用的名字。scenario默认CPU、batch size 4、训练预算2000步、采样40步；不加载HIMO官方生成网络权重。

## 1. 检查数据与诊断链路

```bash
"$TRIDI_PY" main.py -c config/env.yaml scenarios/himo_ordered.yaml -- \
  run.job=inspect run.name=901_himo_inspect

"$TRIDI_PY" main.py -c config/env.yaml scenarios/himo_ordered.yaml -- \
  run.job=diagnose run.name=902_himo_diagnose himo.condition_index=0
```

库存写入`experiments/901_himo_inspect/himo_inspection.json`。诊断结果位于`experiments/902_himo_diagnose/diagnostic/`，包括真实网格、梯度检查和两步未训练采样的尺寸检查。诊断不执行优化器更新、不保存已训练权重，`untrained_shape_check`不是模型效果展示。

## 2. 审核阶段标签并准备训练数据

当前`outputs/himo-small-stage-20261029/stage_candidates.json`只有3条`pending_human_review`候选，均未分配train/val split；**正式训练尚未启动**。直接prepare或train会拒绝这些标签。

人工逐条确认动作名称、完成帧、物体slot和用手后，将接受的完整记录另存为`stage_verified.json`。每条被明确接受的记录需要：

```json
{
  "status": "verified",
  "review_status": "accepted_by_human",
  "split": "train"
}
```

上面仅展示要修改的字段，不能替代完整记录。保留原来的`sequence`、`object_slots`、`keyframes`、`actions`、来源SHA和限制说明。字段记录已经发生的人工确认，不能用批量替换状态来代替审核。若修改帧号，须同步动作起止：`S0 < start1 <= end1=S1 < start2 <= end2=S2`。

至少需要来自不同源序列的train和val；test可另分配。同一原始序列不可跨split，三条S01样本也不能支持跨人物泛化结论。预处理记录验证／测试中未在train出现的动作ID，不应将这些动作视为已学会。

```bash
"$TRIDI_PY" main.py -c config/env.yaml scenarios/himo_ordered.yaml -- \
  run.job=prepare run.name=903_himo_prepare \
  himo.annotations=./outputs/himo-small-stage-20261029/stage_verified.json
```

人体模型、旋转布局、单位及690采样索引SHA写入数据和后续checkpoint，防止SMPL-H与SMPL-X同维参数混用。准备好的数据路径已存在时会拒绝覆盖；需要新版本可覆盖配置`himo.prepared_data=新路径`，后续命令也须使用同一路径。

## 3. 训练

标签和划分就绪后，选择实际空闲GPU，例如GPU 0：

```bash
CUDA_VISIBLE_DEVICES=0 "$TRIDI_PY" main.py -c config/env.yaml scenarios/himo_ordered.yaml -- \
  run.job=train run.name=904_himo_train run.cpu=false \
  himo.annotations=./outputs/himo-small-stage-20261029/stage_verified.json \
  train.max_steps=2000 dataloader.batch_size=4 optimizer.lr=0.0001
```

`run.cpu=true`可改为CPU。训练输出为`experiments/904_himo_train/fit/`，包含`protocol.json`、`history.json`和`best.pth`；初始化采用TriDi权重，新增条件及接触头需要学习。当前不支持HIMO训练resume，每次使用新run名；2000步仅为小规模预算，不保证动作语义或泛化已学会。

## 4. 导出初始条件并采样

```bash
"$TRIDI_PY" main.py -c config/env.yaml scenarios/himo_ordered.yaml -- \
  run.job=conditions run.name=905_himo_conditions himo.condition_index=0

"$TRIDI_PY" main.py -c config/env.yaml scenarios/himo_ordered.yaml -- \
  run.job=sample run.name=906_himo_sample \
  resume.checkpoint=./experiments/904_himo_train/fit/best.pth \
  himo.conditions=./experiments/905_himo_conditions/initial_conditions.pth \
  sample.num_inference_steps=40 sample.target=meshes
```

`conditions`从准备好的数据取一条**仅初态**条件，保存物体名、动作词表和表示协议，不包含未来目标。新场景可按相同结构构造条件文件，动作ID和表示协议必须与checkpoint一致。

采样输出为`experiments/906_himo_sample/generation.pth`及`meshes/`：每帧一份人体、两份物体网格，另有正确52关节和网格清单。设`sample.target=tensors`可只保存参数。只接受已训练的`tridi-himo-ordered-v2`权重；旧TriDi、旧pair或诊断权重不能直接用于此采样入口。

本接入的接触标签采用5厘米近邻阈值，不代表精细抓握或无穿透。未建模静态支撑、液体和材料变化须保留为评估限制；采样没有IK、动作规则或逐样本姿态修正。
