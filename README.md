# MiniMax H3 在海光 DCU 上运行

这里提供 MiniMax-H3 首帧视频生成的海光适配代码、构建说明和数值验证资料。模型权重需要单独获取。

## 选择合适的版本

| 分支 | 用途 | 入口 |
| --- | --- | --- |
| [hygon-bw1000](https://github.com/Entropic-Silence/MiniMaxH3_On_Hygon/tree/hygon-bw1000) | BW1000 八卡源码与工程说明 | start_bw1000.sh |
| [dcu8-reproducible](https://github.com/Entropic-Silence/MiniMaxH3_On_Hygon/tree/dcu8-reproducible) | 八卡构建、生成与结果比较 | run_dcu8_workflow.sh |
| main | 保留既有 K100 复现文件及历史工具 | 查看对应环境脚本 |

BW1000 的完整源码、固定依赖和运行入口位于上述八卡分支。各分支 README 提供环境要求、组件仓库版本和分阶段运行方法。

## 八卡实现

已验证的平台是 gfx936、Python 3.11.9、DAS PyTorch 2.9.0 和 DTK 26.04。主要改动涉及 QKV 序列并行布局、GEMM 长度覆盖、残差与 RMS 融合，以及保持原有计算顺序的稠密注意力内核。

使用仓库中的小狗首帧、提示词 `A dog falls asleep`、768×768 分辨率、124 帧、seed 42、steps 50，优化实现与 PyTorch 参考输出的平均 SSIM 为 98.996%，尺寸和长度检查均通过。2026-11-09 的一次完整生成中，优化流水线耗时 110.697 秒，参考流水线耗时 441.477 秒。输入、输出视频及计时日志保存在八卡分支的 `validation` 目录。

## 快速开始

```bash
GIT_CONFIG_NOSYSTEM=1 git clone --branch hygon-bw1000 https://github.com/Entropic-Silence/MiniMaxH3_On_Hygon.git
cd MiniMaxH3_On_Hygon
export H3_MODEL=/path/to/MiniMax-H3
bash start_bw1000.sh reproduce
```

基础环境提供匹配的海光 PyTorch、DTK 和编译工具；入口自动校验源码、准备 Python 依赖、编译并完成优化与参考生成及比较。首次构建需要网络连接，模型权重通过 `H3_MODEL` 指定。各组件保留上游许可证与版权声明，模型权重按原项目许可获取。
