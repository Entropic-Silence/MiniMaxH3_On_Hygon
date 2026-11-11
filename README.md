# MiniMax H3 八卡构建与运行

通过固定版本源码完成 MiniMax-H3 八卡构建、视频生成及参考结果比较。运行入口自动准备工作目录并保存各阶段日志。

## 准备环境

Linux x86_64，8 张 BW1000（gfx936），DTK 26.04 / DCC 25.10，Python 3.11.9，PyTorch 2.9.0+das.opt1.dtk2604，Triton 3.5.1，HIP BLAS 和 CCL 由 DTK 提供。基础镜像需提供匹配的 PyTorch、DTK、Git、C/C++ 编译工具和 Python venv。入口根据公开依赖记录自动安装海光 Triton 3.5.1 与 FlashAttention 2.8.3，安装前校验 SHA256。获取集成仓库后，首次构建需要访问 Python 包索引；PIP_INDEX_URL 可指定镜像站。模型权重目录通过 H3_MODEL 设置。

## 从源码生成视频

```bash
GIT_CONFIG_NOSYSTEM=1 git -c checkout.workers=1 clone --branch dcu8-reproducible https://github.com/Entropic-Silence/MiniMaxH3_On_Hygon.git
cd MiniMaxH3_On_Hygon
export H3_MODEL=/path/to/MiniMax-H3
bash run_dcu8_workflow.sh reproduce
```

`reproduce` 会从源码校验开始，建立虚拟环境并安装固定依赖，随后编译 AOT 内核及原生扩展。构建完成后，它用相同首帧和提示词分别执行优化路径与 PyTorch 参考路径，再比较两份输出。完整构建日志在 `outputs/build.log`，视频、生成日志和比较结果也保存在 `outputs`。需要分步运行时，按 `fetch`、`build`、`generate`、`reference`、`compare` 的顺序调用入口。

## 组件与版本

| 组件 | Public 仓库 | Branch | Commit |
| --- | --- | --- | --- |
| InfiniRT | https://github.com/Entropic-Silence/InfiniRT_Hygon | bw1000 | `3234df032a2628238a8150194540d79a30563ef5` |
| InfiniOps | https://github.com/Entropic-Silence/InfiniOps_Hygon | bw1000 | `0c259c45d3d6fab9fdb6420a81b2feee91dcca59` |
| InfiniCCL | https://github.com/Entropic-Silence/InfiniCCL_Hygon | bw1000 | `e6a2062e20f1b01e003757daa7aa15ecf7831323` |
| H3Attention | https://github.com/Entropic-Silence/H3Attention_Hygon | bw1000 | `4c6c744f8ac2dd8491507ba2cb241a9b98647581` |
| InfiniCore | https://github.com/Entropic-Silence/InfiniCore_Hygon | bw1000 | `a94aff62a970a3184b1043f11cdf92123c67c6b7` |
| InfiniLM | https://github.com/Entropic-Silence/InfiniLM_Hygon | bw1000 | `39d318fda7b5ae395f24edf0107c24f385d153c8` |
| NineToothed | https://github.com/Entropic-Silence/NineToothed_Hygon | bw1000 | `92b0ccff07336a7417d631b55342bd80d9b65aa4` |
| Diffusers | https://github.com/Entropic-Silence/Diffusers_Hygon | bw1000 | `77df80c3cd62c413a34b9e2fd1089b3b91dd2610` |

集成仓库的 implementation 目录完整收录四个主组件源码，文件校验值位于 vendored-sources.json。InfiniCore 的 submodules 目录完整收录 InfiniRT、InfiniOps、InfiniCCL；tools/h3_dense_attention/prepared 收录 H3Attention 的 FlashAttention、CUTLASS 和内核入口。各嵌入目录与对应独立仓库使用相同文件内容。InfiniLM 的 spdlog、JSON 等上游依赖保留在源码树中。

## 执行过程

执行从 Qwen 条件编码和首帧编码开始，经八卡序列并行完成 DiT 去噪，再由 VAE 解码并输出媒体文件。InfiniLM 组织这些阶段，InfiniCore 执行 GEMM、归一化、RoPE、打包与注意力，NineToothed 生成融合算子及 HIP AOT 产物。Diffusers 中的 MiniMax-H3 模型、调度器和流水线用于参考生成。

## 样例输出

默认首帧为 assets/first_frame.png，提示词 A dog falls asleep；768×768、124 帧、seed 42、steps 50。官方比较结果：尺寸与长度 PASS，平均 SSIM 98.996%。

| 计时项 | PyTorch 参考 | 优化实现 |
| --- | ---: | ---: |
| 模型生成进程总耗时 | 639.842 s | 577.959 s |
| 生成流水线 | 441.477 s | 110.697 s |
| 49 次 DiT 前向合计 | 406.550 s | 79.554 s |

计时来自 2026-11-09 每种实现各一次完整生成。生成流水线覆盖条件编码、去噪和 VAE 解码。视频、日志和比较结果位于 validation。H3_IMAGE、H3_TEXT、H3_OUTPUT 可设置首帧、提示词和输出目录。模型权重遵循原有许可；各组件保留上游许可证。
