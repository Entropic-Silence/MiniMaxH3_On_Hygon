# MiniMax H3 海光八卡复现指南

这份指南把 MiniMax-H3 的海光八卡运行流程收拢为四个动作：构建、生成、生成参考视频和比较结果。模型权重由使用者自行准备，代码包与输入图像均提供校验值，便于复查环境差异。

## 开始之前

请先确认机器具备 8 张 BW1000，架构为 gfx936。当前验证使用 Linux、Python 3.11.9、PyTorch 2.9.0+das.opt1.dtk2604、DTK 26.04、DCC 25.10 和 Triton 3.5.1。FlashAttention、HIP/BLAS/CCL 与 Python 依赖必须和这套海光环境匹配；普通 CUDA wheel 不能直接替换 DAS 组件。迁移到其他架构时需要重新构建和检查数值结果。

## 从源码到视频

```bash
git clone --branch dcu8-reproducible https://github.com/Entropic-Silence/MiniMaxH3_On_Hygon.git
cd MiniMaxH3_On_Hygon
export H3_MODEL=/path/to/MiniMax-H3
bash run_dcu8_workflow.sh build
bash run_dcu8_workflow.sh generate
bash run_dcu8_workflow.sh reference
bash run_dcu8_workflow.sh compare
```

四个动作依次完成本地构建、优化版生成、PyTorch 参考生成和视频比较。模型目录由 H3_MODEL 指定，仓库不包含权重。构建使用包内的离线依赖，但仍以已有匹配的 Torch、FlashAttention 和 DTK 为前提。新内核已接入 build_all.sh；现有构建证据来自 Core 桥接层和注意力库的分阶段重建，尚未覆盖全新操作系统上的完整安装流程。

首次运行可以直接使用随仓库提供的 assets/first_frame.png 和提示词 A dog falls asleep。默认输出为 768×768、124 帧，随机种子为 42，steps 为 50，对应 49 次主去噪前向。更换输入时设置 H3_IMAGE 和 H3_TEXT；更换结果目录时设置 H3_OUTPUT。默认结果写入 outputs，该目录与临时环境均不纳入版本管理。

## 怎样理解测试数据

局部性能采用固定输入 ABBA 对照测量：同一八卡实例上，DiT 前向中位耗时从 0.987319 秒变为 0.965104 秒，减少 2.250065%。注意力算子的已测形状覆盖八个设备，中位降幅为 4.080995%。对照对象是上一版已优化实现，因此这组数字描述新增改动的收益，不能直接用于推算完整视频生成速度。

新内核保留稠密 BF16 计算，省去注意力累加器乘以精确 1 的操作，同时维持 softmax 分母的 FP32 融合乘加与逐行加法顺序。没有稀疏化、量化、跳步或跨步输出缓存。旧 astronaut 与 cat 用例的四次完整生成、196 对逐步视频/音频结果均与优化前基线一致；这些历史相似度不能作为本仓库小狗图像的成绩。仓库内小狗原图已完成八卡验证：124 帧、768×768、seed 42、steps 50。官方比较脚本给出的平均 SSIM 为 98.996%；优化版生成流水线耗时 110.697 秒，PyTorch 参考版为 441.477 秒。该计时排除模型构建、预热和最终媒体编码，单次运行会受节点状态影响。包含加载、预热等开销的入口总耗时分别为优化版 577.959 秒、参考版 639.842 秒。原始视频、组件计时和比较结果保存在 validation 目录。

## 复查源码与回退

完整计算实现位于 source/minimax_dcu8_bundle.tar.gz，包含 InfiniCore、InfiniLM、NineToothed 和构建材料。两个环境分别重建的新注意力库与实际测试二进制逐字节一致。排查兼容性时，可设置 H3_DENSE_TILE_ENABLE=0 切回原注意力路径，再对同一输入进行比较。

源码包 SHA256：`cfc760d64eeef6d3b47c16b1be36856a3ecd7e6448183dd7b3e8aa85d1457512`。
