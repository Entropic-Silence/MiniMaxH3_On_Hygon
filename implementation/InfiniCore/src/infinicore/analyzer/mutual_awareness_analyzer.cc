#include "infinicore/analyzer/mutual_awareness_analyzer.hpp"

#include "infinirt.h"

namespace infinicore::analyzer {

namespace {

DeviceResourceSnapshot buildSnapshotFromMemoryStats(
    int device_id,
    infinicore::Device::Type device_type,
    const MemoryStats &stats) {
    DeviceResourceSnapshot snapshot;
    snapshot.device_id = device_id;
    snapshot.device_type = device_type;
    snapshot.has_memory_capacity = stats.total_capacity > 0;
    snapshot.free_bytes = stats.total_capacity >= stats.allocated_bytes
                            ? (stats.total_capacity - stats.allocated_bytes)
                            : 0;
    snapshot.total_bytes = stats.total_capacity;
    snapshot.used_bytes = stats.allocated_bytes;
    snapshot.reserved_bytes = stats.total_capacity;
    return snapshot;
}

DeviceResourceSnapshot buildSnapshotFromInfinirt(
    infinicore::Device device,
    const MemoryStats &allocator_stats) {
    DeviceResourceSnapshot snapshot;
    snapshot.device_id = static_cast<int>(device.getIndex());
    snapshot.device_type = device.getType();

    infinirtDeviceResourceSnapshot_t rt_snapshot{};
    auto status = infinirtGetDeviceResourceSnapshot(
        static_cast<infiniDevice_t>(device.getType()),
        static_cast<int>(device.getIndex()),
        &rt_snapshot);

    if (status == INFINI_STATUS_SUCCESS) {
        snapshot.has_memory_capacity = (rt_snapshot.valid_fields & INFINIRT_RESOURCE_FIELD_MEMORY_CAPACITY) != 0;
        snapshot.has_compute_utilization = (rt_snapshot.valid_fields & INFINIRT_RESOURCE_FIELD_COMPUTE_UTILIZATION) != 0;
        snapshot.has_memory_bandwidth_utilization = (rt_snapshot.valid_fields & INFINIRT_RESOURCE_FIELD_MEMORY_BANDWIDTH_UTILIZATION) != 0;
        snapshot.has_kernel_time_ratio = (rt_snapshot.valid_fields & INFINIRT_RESOURCE_FIELD_KERNEL_TIME_RATIO) != 0;
        snapshot.has_communication = (rt_snapshot.valid_fields & INFINIRT_RESOURCE_FIELD_COMMUNICATION) != 0;
        snapshot.kernel_time_estimated = (rt_snapshot.estimated_fields & INFINIRT_RESOURCE_FIELD_KERNEL_TIME_RATIO) != 0;

        snapshot.free_bytes = rt_snapshot.free_bytes;
        snapshot.total_bytes = rt_snapshot.total_bytes;
        snapshot.used_bytes = rt_snapshot.used_bytes;
        snapshot.reserved_bytes = rt_snapshot.reserved_bytes;
        snapshot.compute_utilization = rt_snapshot.compute_utilization;
        snapshot.memory_bandwidth_utilization = rt_snapshot.memory_bandwidth_utilization;
        snapshot.kernel_time_ratio = rt_snapshot.kernel_time_ratio;
        snapshot.communication_time_ratio = rt_snapshot.communication_time_ratio;
        snapshot.communication_bytes = rt_snapshot.communication_bytes;
    }

    // Keep allocator pool information around even when vendor memory info exists.
    if (snapshot.reserved_bytes == 0) {
        snapshot.reserved_bytes = allocator_stats.total_capacity;
    }

    // Fallback path for backends without a resource snapshot implementation.
    if (!snapshot.has_memory_capacity || snapshot.total_bytes == 0) {
        auto fallback = buildSnapshotFromMemoryStats(
            static_cast<int>(device.getIndex()),
            device.getType(),
            allocator_stats);
        snapshot.has_memory_capacity = fallback.has_memory_capacity;
        snapshot.free_bytes = fallback.free_bytes;
        snapshot.total_bytes = fallback.total_bytes;
        snapshot.used_bytes = fallback.used_bytes;
    }

    return snapshot;
}

std::vector<DeviceResourceSnapshot> collectRuntimeResourceSnapshots() {
    std::vector<DeviceResourceSnapshot> device_snapshots;
    // TODO: integrate with ContextImpl allocator stats when available.
    // For now, return an empty snapshot list; the analyzer will use
    // the fallback path that queries infinirt directly.
    return device_snapshots;
}

} // namespace

// ============================================================
// Singleton
// ============================================================

MutualAwarenessAnalyzer &MutualAwarenessAnalyzer::instance() {
    static MutualAwarenessAnalyzer inst;
    return inst;
}

MutualAwarenessAnalyzer::MutualAwarenessAnalyzer()
    : phase_detector_(),
      resource_sensor_(),
      intent_generator_(),
      enabled_(true),
      graph_intent_cached_(false) {
}

// ============================================================
// Main analysis entry points
// ============================================================

OptimizationIntent MutualAwarenessAnalyzer::analyze() {
    if (!enabled_) {
        return OptimizationIntent{};
    }

    // If we have a cached graph intent, return it
    if (graph_intent_cached_) {
        return graph_cached_intent_;
    }

    // Get recent op trace window
    auto &trace = getGlobalOpTrace();
    auto window = trace.getRecentEntries(phase_detector_.config().window_size);

    // Detect phase
    PhaseType phase = phase_detector_.detect(window);

    auto device_snapshots = collectRuntimeResourceSnapshots();
    std::vector<DeviceLocalIntent> device_intents;
    device_intents.reserve(device_snapshots.size());
    for (auto const &snapshot : device_snapshots) {
        device_intents.push_back(resource_sensor_.sense(snapshot));
    }

    // Generate intent
    auto intent = intent_generator_.generate(phase, window, device_intents);

    // Cache result
    {
        std::lock_guard<std::mutex> lock(mutex_);
        last_intent_ = intent;
    }

    return intent;
}

OptimizationIntent MutualAwarenessAnalyzer::analyze(
    const std::vector<std::pair<int, MemoryStats>> &device_stats) {

    if (!enabled_) {
        return OptimizationIntent{};
    }

    // If we have a cached graph intent, return it
    if (graph_intent_cached_) {
        return graph_cached_intent_;
    }

    // Get recent op trace window
    auto &trace = getGlobalOpTrace();
    auto window = trace.getRecentEntries(phase_detector_.config().window_size);

    // Detect phase
    PhaseType phase = phase_detector_.detect(window);

    // Build per-device intents from provided stats
    std::vector<DeviceLocalIntent> device_intents;
    device_intents.reserve(device_stats.size());
    for (auto &[dev_id, stats] : device_stats) {
        device_intents.push_back(resource_sensor_.sense(dev_id, stats));
    }

    // Generate intent
    auto intent = intent_generator_.generate(phase, window, device_intents);

    // Cache result
    {
        std::lock_guard<std::mutex> lock(mutex_);
        last_intent_ = intent;
    }

    return intent;
}

OptimizationIntent MutualAwarenessAnalyzer::analyze(
    const std::vector<DeviceResourceSnapshot> &device_snapshots) {

    if (!enabled_) {
        return OptimizationIntent{};
    }

    if (graph_intent_cached_) {
        return graph_cached_intent_;
    }

    auto &trace = getGlobalOpTrace();
    auto window = trace.getRecentEntries(phase_detector_.config().window_size);

    PhaseType phase = phase_detector_.detect(window);

    std::vector<DeviceLocalIntent> device_intents;
    device_intents.reserve(device_snapshots.size());
    for (auto const &snapshot : device_snapshots) {
        device_intents.push_back(resource_sensor_.sense(snapshot));
    }

    auto intent = intent_generator_.generate(phase, window, device_intents);

    {
        std::lock_guard<std::mutex> lock(mutex_);
        last_intent_ = intent;
    }

    return intent;
}

PhaseType MutualAwarenessAnalyzer::getCurrentPhase() const {
    if (!enabled_) {
        return PhaseType::UNKNOWN;
    }

    return phase_detector_.detectFromTrace(getGlobalOpTrace());
}

OptimizationGoal MutualAwarenessAnalyzer::getCurrentOptimizationGoal() const {
    return const_cast<MutualAwarenessAnalyzer *>(this)->analyze().global.goal;
}

const OptimizationIntent &MutualAwarenessAnalyzer::lastIntent() const {
    return last_intent_;
}

// ============================================================
// Graph recording support
// ============================================================

void MutualAwarenessAnalyzer::onGraphRecordingStop() {
    if (!enabled_) {
        return;
    }

    // Analyze the op sequence recorded during graph capture
    // and cache the result. Graph ops are static, so we only
    // need to analyze once.
    graph_cached_intent_ = analyze();
    graph_intent_cached_ = true;
}

void MutualAwarenessAnalyzer::clearGraphCache() {
    graph_intent_cached_ = false;
    graph_cached_intent_ = OptimizationIntent{};
}

// ============================================================
// C-style API for external framework integration
// ============================================================

OptimizationIntent analyzeCurrentState() {
    return MutualAwarenessAnalyzer::instance().analyze();
}

PhaseType getCurrentPhase() {
    return MutualAwarenessAnalyzer::instance().getCurrentPhase();
}

OptimizationGoal getCurrentOptimizationGoal() {
    return MutualAwarenessAnalyzer::instance().getCurrentOptimizationGoal();
}

void setAnalyzerEnabled(bool enabled) {
    MutualAwarenessAnalyzer::instance().setEnabled(enabled);
}

} // namespace infinicore::analyzer
