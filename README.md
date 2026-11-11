# MiniMax H3 BW1000 推理实现

本项目提供 MiniMax-H3 在海光 BW1000 八卡节点上的推理实现和构建入口。优化集中于序列并行的数据布局、GEMM 调度、残差与归一化融合，以及保持数值一致的稠密注意力内核。

## 环境

已验证环境为 Linux、8 张 BW1000（gfx936）、Python 3.11.9、PyTorch 2.9.0+das.opt1.dtk2604、DTK 26.04、DCC 25.10 和 Triton 3.5.1。需要匹配该环境的海光 FlashAttention、HIP/BLAS/CCL 及 Python 依赖。不要用普通 CUDA wheel 替换 DAS 组件。其他架构需要重新验证。

## 使用

```bash
git clone --branch hygon-bw1000 https://github.com/Entropic-Silence/MiniMaxH3_On_Hygon.git
cd MiniMaxH3_On_Hygon
export H3_MODEL=/path/to/MiniMax-H3
bash start_bw1000.sh build
bash start_bw1000.sh generate
bash start_bw1000.sh reference
bash start_bw1000.sh compare
```

需要已有匹配的 Torch、FlashAttention 和 DTK 环境。构建脚本使用包内离线依赖；权重不随仓库分发。完整 build_all.sh 集成了新内核，但本次验证采用分阶段重建：Core 桥接层与注意力库分别编译和验证，未宣称在全新操作系统上一键完成全部安装。

默认输入为 assets/first_frame.png，提示词为 A dog falls asleep。生成配置是 768×768、124 帧、seed 42、steps 50，实际主去噪前向为 49 次。通过 H3_IMAGE、H3_TEXT、H3_OUTPUT 指定自己的输入与输出。运行文件位于 outputs，不应把模型权重、凭据或临时环境加入 Git。

## 实测范围

同一八卡实例的固定输入 ABBA 对照中，DiT 前向中位耗时由 0.987319 秒降至 0.965104 秒，下降 2.250065%；注意力算子在已测形状及八个设备上的中位降幅为 4.080995%。这是相对先前已优化原生实现的局部增量，不能换算成完整视频生成的等比例提速。

新内核保留稠密 BF16 计算，省去注意力累加器乘以精确 1 的操作，同时维持 softmax 分母的 FP32 融合乘加与逐行加法顺序。没有稀疏化、量化、跳步或跨步输出缓存。旧 astronaut 与 cat 用例的四次完整生成、196 对逐步视频/音频结果均与优化前基线一致；这些历史相似度不能作为本仓库小狗图像的成绩。仓库内小狗原图已完成八卡验证：124 帧、768×768、seed 42、steps 50。官方比较脚本给出的平均 SSIM 为 98.996%；优化版生成流水线耗时 110.697 秒，PyTorch 参考版为 441.477 秒。该计时排除模型构建、预热和最终媒体编码，单次运行会受节点状态影响。包含加载、预热等开销的入口总耗时分别为优化版 577.959 秒、参考版 639.842 秒。原始视频、组件计时和比较结果保存在 validation 目录。

## 源码与复现

source/h3_bw1000_source.tar.gz 是可重建源码快照，包含 InfiniCore、InfiniLM、NineToothed 及相关构建材料。新注意力库在两个环境独立重建后与被测二进制逐字节一致。显式设置 H3_DENSE_TILE_ENABLE=0 可回退旧注意力路径。

源码包 SHA256：`cfc760d64eeef6d3b47c16b1be36856a3ecd7e6448183dd7b3e8aa85d1457512`。
