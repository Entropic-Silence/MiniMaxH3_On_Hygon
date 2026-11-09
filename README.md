# MiniMax H3 在海光 DCU 上运行

这里提供 MiniMax-H3 首帧视频生成的海光适配代码、构建说明和数值验证资料。模型权重需要单独获取。

## 选择合适的版本

| 分支 | 用途 | 入口 |
| --- | --- | --- |
| [hygon-bw1000](https://github.com/Entropic-Silence/MiniMaxH3_On_Hygon/tree/hygon-bw1000) | BW1000 八卡源码与工程说明 | start_bw1000.sh |
| [dcu8-reproducible](https://github.com/Entropic-Silence/MiniMaxH3_On_Hygon/tree/dcu8-reproducible) | 同一八卡实现的复现指南 | run_dcu8_workflow.sh |
| main | 保留既有 K100 复现文件及历史工具 | 查看对应环境脚本 |

两个 BW1000 分支共享经过验证的计算实现，说明结构和发布文件命名不同。BW1000 用户应切换到其中一个分支，按照其 README 准备环境；不要把 main 中的 K100 历史路径直接当作八卡部署配置。

## 八卡实现

已验证的平台是 gfx936、Python 3.11.9、DAS PyTorch 2.9.0 和 DTK 26.04。主要改动涉及 QKV 序列并行布局、GEMM 长度覆盖、残差与 RMS 融合，以及保持原有计算顺序的稠密注意力内核。

固定输入的同实例 ABBA 测试中，最近一次注意力改动使原生 DiT 前向耗时从 0.987319 秒降至 0.965104 秒。该数据是相对此前已优化实现的局部增量，不是完整视频生成的端到端加速比。完整输入、日志和比较方式以发布分支的验证材料为准。

## 快速开始

```bash
git clone --branch hygon-bw1000 https://github.com/Entropic-Silence/MiniMaxH3_On_Hygon.git
cd MiniMaxH3_On_Hygon
export H3_MODEL=/path/to/MiniMax-H3
bash start_bw1000.sh build
bash start_bw1000.sh generate
```

构建需要预先准备匹配的海光软件栈；不要混装普通 CUDA wheel。源码快照保留上游许可证与版权声明，权重按原项目的许可获取。
