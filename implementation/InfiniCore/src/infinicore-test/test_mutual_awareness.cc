#include "test_mutual_awareness.h"

#ifdef ENABLE_MUTUAL_AWARENESS

#include "infinicore/analyzer.hpp"
#include "infinicore/ops/common/dispatcher.hpp"

#include <vector>

namespace infinicore::test {

namespace {

using namespace infinicore::analyzer;

#define MA_ASSERT_TRUE(cond)                                \
    do {                                                    \
        if (!(cond)) {                                      \
            spdlog::error("ASSERT_TRUE failed: {} ({}:{})", \
                          #cond, __FILE__, __LINE__);       \
            return false;                                   \
        }                                                   \
    } while (0)

#define MA_ASSERT_EQ(a, b)                                      \
    do {                                                        \
        if (!((a) == (b))) {                                    \
            spdlog::error("ASSERT_EQ failed: {} == {} ({}:{})", \
                          #a, #b, __FILE__, __LINE__);          \
            return false;                                       \
        }                                                       \
    } while (0)

std::vector<OpTraceEntry> makeWindow(const std::vector<OpType> &types,
                                     uint32_t seq_len = 128) {
    std::vector<OpTraceEntry> window;
    for (auto t : types) {
        OpTraceEntry e;
        e.op_type = t;
        e.ndim = 3;
        e.shape[0] = 1;
        e.shape[1] = 32;
        e.shape[2] = seq_len;
        window.push_back(e);
    }
    return window;
}

bool testOpTraceRing() {
    OpTraceRing ring(4);
    MA_ASSERT_EQ(ring.size(), 0u);
    MA_ASSERT_EQ(ring.capacity(), 4u);

    OpTraceEntry first;
    first.op_type = OpType::GEMM;
    ring.write(first);
    MA_ASSERT_EQ(ring.size(), 1u);
    MA_ASSERT_EQ(ring.totalCount(), 1u);

    for (int i = 0; i < 6; ++i) {
        OpTraceEntry x;
        x.op_type = static_cast<OpType>(i + 1);
        ring.write(x);
    }
    MA_ASSERT_EQ(ring.size(), 4u);
    MA_ASSERT_EQ(ring.totalCount(), 7u);

    auto recent = ring.getRecentEntries(2);
    MA_ASSERT_EQ(recent.size(), 2u);

    ring.clear();
    MA_ASSERT_EQ(ring.size(), 0u);
    return true;
}

bool testOpTypeRegistry() {
    MA_ASSERT_EQ(opTypeFromName("Add"), OpType::ADD);
    MA_ASSERT_EQ(opTypeFromName("FlashAttention"), OpType::FLASH_ATTENTION);
    MA_ASSERT_EQ(opTypeFromName("RMSNorm"), OpType::RMS_NORM);
    MA_ASSERT_EQ(opTypeFromName("Rearrange"), OpType::REARRANGE);
    MA_ASSERT_EQ(opTypeFromName("__nonexistent__"), OpType::UNKNOWN);
    return true;
}

bool testOpTypeClassification() {
    MA_ASSERT_TRUE(isAttentionOp(OpType::FLASH_ATTENTION));
    MA_ASSERT_TRUE(isAttentionOp(OpType::PAGED_ATTENTION));
    MA_ASSERT_TRUE(!isAttentionOp(OpType::GEMM));
    MA_ASSERT_TRUE(isGemmMlpOp(OpType::GEMM));
    MA_ASSERT_TRUE(isGemmMlpOp(OpType::LINEAR));
    MA_ASSERT_TRUE(isActivationOp(OpType::SILU));
    MA_ASSERT_TRUE(isKvCacheOp(OpType::KV_CACHING));
    return true;
}

bool testPhaseDetector() {
    PhaseDetector detector;

    auto attention = makeWindow(
        {OpType::ATTENTION, OpType::FLASH_ATTENTION, OpType::CAUSAL_SOFTMAX,
         OpType::ATTENTION, OpType::RMS_NORM, OpType::ADD},
        /*seq_len=*/16);
    MA_ASSERT_EQ(detector.detect(attention), PhaseType::ATTENTION_DENSE);

    auto gemm = makeWindow({OpType::LINEAR, OpType::SILU, OpType::LINEAR,
                            OpType::GEMM, OpType::GELU, OpType::LINEAR});
    MA_ASSERT_EQ(detector.detect(gemm), PhaseType::GEMM_MLP_DENSE);

    auto kv = makeWindow({OpType::KV_CACHING, OpType::KV_CACHING,
                          OpType::PAGED_CACHING, OpType::KV_CACHING,
                          OpType::ADD});
    MA_ASSERT_EQ(detector.detect(kv), PhaseType::KV_CACHE);

    auto decode = makeWindow(
        {OpType::ATTENTION, OpType::CAUSAL_SOFTMAX, OpType::LINEAR,
         OpType::SILU, OpType::GEMM, OpType::ADD},
        /*seq_len=*/1);
    MA_ASSERT_EQ(detector.detect(decode), PhaseType::DECODE);

    auto prefill = makeWindow(
        {OpType::ATTENTION, OpType::FLASH_ATTENTION, OpType::CAUSAL_SOFTMAX,
         OpType::ATTENTION, OpType::RMS_NORM},
        /*seq_len=*/512);
    MA_ASSERT_EQ(detector.detect(prefill), PhaseType::PREFILL);

    std::vector<OpTraceEntry> empty;
    MA_ASSERT_EQ(detector.detect(empty), PhaseType::UNKNOWN);
    return true;
}

bool testResourceSensor() {
    ResourceSensor sensor;

    MemoryStats high{900, 1000, 0, 0};
    auto high_intent = sensor.sense(0, high);
    MA_ASSERT_EQ(high_intent.local_bottleneck, BottleneckType::MEMORY_BOUND);

    MemoryStats low{100, 1000, 0, 0};
    auto low_intent = sensor.sense(0, low);
    MA_ASSERT_EQ(low_intent.local_bottleneck, BottleneckType::COMPUTE_BOUND);

    DeviceResourceSnapshot snap;
    snap.device_id = 0;
    snap.device_type = Device::Type::NVIDIA;
    snap.has_memory_capacity = true;
    snap.total_bytes = 1000;
    snap.used_bytes = 300;
    snap.free_bytes = 700;
    snap.has_memory_bandwidth_utilization = true;
    snap.memory_bandwidth_utilization = 0.9f;
    auto bw_intent = sensor.sense(snap);
    MA_ASSERT_EQ(bw_intent.local_bottleneck, BottleneckType::BANDWIDTH_BOUND);
    return true;
}

bool testIntentGenerator() {
    IntentGenerator gen;

    DeviceLocalIntent pressure;
    pressure.device_id = 0;
    pressure.memory_usage_ratio = 0.95f;
    pressure.local_bottleneck = BottleneckType::MEMORY_BOUND;
    auto window = makeWindow({OpType::GEMM, OpType::LINEAR});
    auto intent = gen.generate(PhaseType::GEMM_MLP_DENSE, window, {pressure});
    MA_ASSERT_EQ(intent.global.primary_bottleneck, BottleneckType::MEMORY_BOUND);
    MA_ASSERT_EQ(intent.global.goal, OptimizationGoal::MEMORY_SAFE);
    MA_ASSERT_TRUE(intent.global.strategy.prefer_in_place);
    MA_ASSERT_TRUE(intent.global.strategy.prefer_recomputation);

    DeviceLocalIntent fresh;
    fresh.device_id = 0;
    fresh.memory_usage_ratio = 0.3f;
    fresh.local_bottleneck = BottleneckType::COMPUTE_BOUND;
    auto attention = makeWindow({OpType::ATTENTION}, /*seq_len=*/1);
    auto decode_intent = gen.generate(PhaseType::DECODE, attention, {fresh});
    MA_ASSERT_EQ(decode_intent.global.goal, OptimizationGoal::LATENCY_FIRST);
    return true;
}

bool testAnalyzerWithMemoryStats() {
    auto &analyzer = MutualAwarenessAnalyzer::instance();
    auto &trace = getGlobalOpTrace();
    analyzer.setEnabled(true);
    analyzer.clearGraphCache();
    trace.clear();

    size_t dims[] = {1, 32, 512};
    traceOp(OpType::ATTENTION, dims, 3, 0, 0, 0);
    traceOp(OpType::FLASH_ATTENTION, dims, 3, 0, 0, 0);

    MemoryStats stats{950 * 1024 * 1024, 1024 * 1024 * 1024UL, 0, 0};
    auto intent = analyzer.analyze({{0, stats}});
    MA_ASSERT_EQ(intent.global.goal, OptimizationGoal::MEMORY_SAFE);

    trace.clear();
    analyzer.clearGraphCache();
    return true;
}

using ProbeFn = int (*)(int);

int probeDefault(int v) { return v + 1; }
int probeThroughput(int v) { return v + 100; }

bool testDispatcherGoalAware() {
    op::common::OpDispatcher<ProbeFn> dispatcher;

    dispatcher.registerDevice(Device::Type::CPU, &probeDefault);
    dispatcher.registerDevice(Device::Type::CPU, &probeThroughput,
                              OptimizationGoal::THROUGHPUT_FIRST);

    // Goal-specific lookup hits the registered kernel.
    MA_ASSERT_EQ(dispatcher.lookup(Device::Type::CPU,
                                   OptimizationGoal::THROUGHPUT_FIRST)(5),
                 105);

    // Goal without a specific registration falls back to the device default.
    MA_ASSERT_EQ(dispatcher.lookup(Device::Type::CPU,
                                   OptimizationGoal::MEMORY_SAFE)(7),
                 8);

    // Legacy device-only lookup still works.
    MA_ASSERT_EQ(dispatcher.lookup(Device::Type::CPU)(9), 10);
    return true;
}

} // namespace

TestResult MutualAwarenessTest::run() {
    return measureTime("MutualAwarenessAllTests", [this]() {
        bool ok = true;
        struct Case {
            const char *name;
            bool (*fn)();
        };
        Case cases[] = {
            {"OpTraceRing", &testOpTraceRing},
            {"OpTypeRegistry", &testOpTypeRegistry},
            {"OpTypeClassification", &testOpTypeClassification},
            {"PhaseDetector", &testPhaseDetector},
            {"ResourceSensor", &testResourceSensor},
            {"IntentGenerator", &testIntentGenerator},
            {"AnalyzerWithMemoryStats", &testAnalyzerWithMemoryStats},
            {"DispatcherGoalAware", &testDispatcherGoalAware},
        };
        for (auto &c : cases) {
            spdlog::info("[mutual-awareness] running: {}", c.name);
            if (!c.fn()) {
                spdlog::error("[mutual-awareness] {} FAILED", c.name);
                ok = false;
            } else {
                spdlog::info("[mutual-awareness] {} passed", c.name);
            }
        }
        return ok;
    });
}

} // namespace infinicore::test

#endif // ENABLE_MUTUAL_AWARENESS
