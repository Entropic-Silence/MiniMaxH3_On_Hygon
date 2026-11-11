# Z200SM_80 / gfx906 算子复现

源码复用已有融合算子实现，使用已有 `scripts/z100/setup_env_z100.sh` 配置 DTK 和持久化目录。此处 Z200SM_80 为实际设备名；既有 z100 目录表示 gfx906 环境脚本。

依赖目录应包含 `xmake-3.0.5.run`、`pybind11-2.13.6-py3-none-any.whl` 和 Triton 3.8.0 的 CPython 3.11 Linux x86_64 wheel。保留镜像提供的海光 PyTorch。

在登录主机设置 KUBECONFIG 并选择 Running POD，然后进入：

```bash
export KUBECONFIG="${KUBECONFIG:-$(find "$HOME/.kube" -type f -name config -print -quit)}"
kubectl get pods
POD=<运行中的Pod名称>
kubectl exec -it "$POD" -- env WORK_HOME="$HOME" bash
```

代码上传至登录主机 `~/h3`。POD 内执行：

```bash
ulimit -c 0
bash "$WORK_HOME/h3/scripts/z200/bootstrap.sh" "$WORK_HOME/h3-z200"
bash "$WORK_HOME/h3/scripts/z100/build_z100.sh" core
source "$WORK_HOME/h3/scripts/z200/setup_env.sh"
python "$H3_TREE/tools/probe_triton_arch.py"
python "$H3_TREE/tools/bench_gfx906_pointwise.py" --seq 8192 --dtype bf16 --json "$WORK_HOME/h3-z200/ops_bf16.json"
python "$H3_TREE/tools/bench_gfx906_pointwise.py" --seq 8192 --dtype fp16 --json "$WORK_HOME/h3-z200/ops_fp16.json"
```

`bench_gfx906_pointwise.py --tune --ops swiglu,gated_residual,qk_norm_rope` 对已有九齿算子做分块参数测量；`--config <JSON>` 固定指定参数。`--broadcast-modulation` 测试 批量 AdaLN 所用的零步长广播视图。

基准中的 PyTorch 仅用于验证与对照，不是部署实现。BF16 InfiniCore H3 接口与九齿 AOT 的集成另需生成并启用 `--h3-ninetoothed=y`；FP16 是独立算子诊断，不改变模型默认 dtype。

本次验证采用 BF16 算子、T=17/128/8192、seed=0/1，并覆盖调制向量零步长广播布局。完整复现：

```bash
bash "$WORK_HOME/h3/scripts/z200/run_ops.sh" "$WORK_HOME/h3-z200/results"
```

最终构建参数应与报告一致。此脚本只构建和测试算子，不加载模型。精度与性能以 JSON 逐项记录；非逐位但满足明示容差的结果仍保留误差数据。
