# TriDi 远程调试

在 Trae 运行和调试中选配置，F5 启动后在入口暂停；F10 逐过程，F11 进入函数。
Python 固定为 tridi-offline，GPU0，数据加载 workers=0。修改 CUDA_VISIBLE_DEVICES 可换 GPU。

- codex/debugger：从 main 创建，GRAB/BEHAVE train/test 预处理、2 步训练、2 样本生成。
- codex/hand-contact：上述配置加穿透引导生成。
- codex/tri-direction-prototype：上述配置加穿透引导和 landing 引导生成。

launch.json 随分支提交，切换分支后选名称带当前分支的配置。
预处理使用单主体单物体、10fps，输出到 data/debug_preprocessed/<分支名>/，不覆盖训练数据。
GRAB 已转换的 SMPL+H 数据需在 data/raw/grab/grab_smplh；BEHAVE 使用原始 split.json。
训练/生成仍读取 data/preprocessed。训练使用现有可用序列拆分文件，限制 2 步。
20 步生成仅用于执行流程检查；完整生成保留 1000 步。
联合流形投影需要训练过的编码器/解码器权重，未把原始 gb_main 权重配置成可用投影模型。
main 基线与当前服务器依赖/数据可能仍有兼容性问题；配置字段验证不等于完整运行成功。
