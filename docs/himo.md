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

人工审核路径中的`outputs/himo-small-stage-20261029/stage_candidates.json`仍保留3条`pending_human_review`候选，未分配train/val split。直接prepare或train会拒绝这些标签。另有用户授权的自动弱监督路径，见下节。

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

`run.cpu=true`可改为CPU。训练输出为`experiments/904_himo_train/fit/`，包含`protocol.json`、`history.json`、`best.pth`和可恢复的`last.pth`；初始化采用TriDi权重，新增条件及接触头需要学习。恢复时使用原run名并指定`resume.checkpoint=该目录/fit/last.pth`，数据SHA、监督类型及训练／评估参数必须保持一致；`train.max_steps`是累计总预算。2000步仅为小规模预算，不保证动作语义或泛化已学会。

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

采样输出为`experiments/906_himo_sample/generation.pth`及`meshes/`：每帧一份人体、两份物体网格，另有正确52关节和网格清单。设`sample.target=tensors`可只保存参数。接受已训练的原生v2／v3权重；旧TriDi、旧pair或诊断权重不能直接用于此采样入口。v2固定恢复legacy激活，v3必须明确记录`model_profile`。

本接入的接触标签采用5厘米近邻阈值，不代表精细抓握或无穿透。未建模静态支撑、液体和材料变化须保留为评估限制；采样没有IK、动作规则或逐样本姿态修正。

## 自动弱监督首轮训练

2026-10-30用户明确接受先自动扩充阶段标签再训练。此路径使用独立文件，不改动上述3条待人工审核候选；标签为`weak_temporal_v1`／`machine_checked`，不写成人工审核通过。

统一算法在平滑后的真实两物体轨迹中寻找两个相邻、主物体明确的活动阶段，过滤跟踪跳变、复杂往返和含糊阶段，并用真实目标端点复核。动作仅为`move_object`、`rotate_object`、`move_and_rotate_object`；源文本仅保留为来源，不能据此声称模型学会喝水、抓握、放入或穿戴等细粒度语义。阈值、源文件SHA和选段证据随每条标签保存。

本轮实际筛选605条：训练512、验证35、测试58。其中75条前后切换主物体，其余主要是同一物体的两个运动阶段；全部`target_slot=None`，没有监督“对目标物体B执行动作”的关系。权重记录训练中见过的目标引用，并拒绝没有监督的目标slot0／1，不能将此基线当作完整双物体任务语义模型。

固定留出S01／S08／S16作验证、S02／S09／S20作测试，其余人物训练。每源序列至多一条；同一人物不跨集合。验证及测试动作必须在训练中出现。

```bash
"$TRIDI_PY" -m tridi.data.himo_weak \
  --root data/raw/himo/unpacked-20261029/data \
  --output outputs/himo-weak-20261030/stage_weak.json --seed 42 \
  --max-train 512 --max-val 64 --max-test 64

"$TRIDI_PY" main.py -c config/env.yaml scenarios/himo_weak_train.yaml -- \
  run.job=prepare run.cpu=true

CUDA_VISIBLE_DEVICES=0 "$TRIDI_PY" main.py \
  -c config/env.yaml scenarios/himo_weak_train.yaml
```

新scenario明确设置`himo.supervision=weak_temporal_v1`，人体／物体／接触和动作条件全部参与学习，预算5000步、batch16、学习率1e-4，AdamW betas=(.95,.999)、weight_decay=1e-6。每50步保存进度与断点；每200步用固定最多32条验证样本、20步DDIM选择`best.pth`。训练后用40步DDIM在留出人物上比较初始化网络、训练网络与静止物体对照，报告姿态误差和接触指标。动作顺序交换仅检查条件敏感性，不能代替顺序正确性证明。

共享GPU上实际运行使用输出目录内的启动脚本限制本进程显存；不要停止其他任务。接触预处理仍使用完整三角表面的5厘米精确阈值查询，仅减少不可能接触的三角候选，已与旧全局最近查询做过逐点等价检查。

## 定位修复与目标关系弱标签v2

`root_affine_v1`保留全部170个初始化权重，仅将人体平移解码头中的SiLU改为Identity，避免窄输出层在驻点附近失去输入敏感性。新训练的配置和CLI默认使用此profile；显式`himo.model_profile=legacy`可重现对照。v3权重保存激活profile，不能只改旧v2文件的标签来替代重新训练。

`oracle`提供真实轴角→原生解码→网格导出重载检查；`smallfit`提供16／32条原训练样本的低噪声、高噪声和完整采样诊断。smallfit权重采用独立schema，不能当成正式生成模型。

目标关系v2沿用605条记录完全相同的状态和划分，仅新增`approach_other_object`、`move_away_from_other_object`与另一物体的目标引用。标签基于真实规范网格表面质心、主物体贡献和距离路径筛选，保留全部v1证据；仍为`weak_temporal_v2 / machine_checked`，不代表放置、包含或接触语义。

已有服务器v2准备文件，可运行：

```bash
"$TRIDI_PY" main.py -c config/env.yaml scenarios/himo_weak_train.yaml \
  scenarios/himo_relations.yaml -- run.name=新训练名称
```

`himo.contact_weight`默认0。可微相对接触项只用于训练，以GT接触mask约束人体点在物体局部坐标中的位置；没有推理姿态修正。权重0.1的小样本对照损害了定位和几何接触，因此当前保留关闭，不能仅凭接触头F1提高就启用。

本次完整修复实验、固定5000步权重与未修正网格位于服务器`outputs/himo-repair-20261030/`。对照分别报告人体位置、物体误差、实际几何接触和关系方向遵从，不将原综合分数的`best.pth`等同于所有指标最佳。
