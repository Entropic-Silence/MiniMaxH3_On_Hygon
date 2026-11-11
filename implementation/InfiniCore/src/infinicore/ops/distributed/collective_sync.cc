#include "infinicore/ops/distributed/collective_sync.hpp"

#include "infinicore/context/context.hpp"

#include <mutex>

namespace infinicore::op::distributed {

namespace {
thread_local bool synchronous = false;
std::mutex launch_mutex;
thread_local int guard_depth = 0;
thread_local bool guard_owned = false;
} // namespace

FirstForwardLaunchGuard::FirstForwardLaunchGuard() {
    if (!synchronous) {
        return;
    }
    counted_ = true;
    if (guard_depth++ == 0) {
        launch_mutex.lock();
        guard_owned = true;
    }
}

FirstForwardLaunchGuard::~FirstForwardLaunchGuard() {
    if (counted_ && --guard_depth == 0) {
        release_launch_guard();
    }
}

void release_launch_guard() {
    if (guard_owned) {
        guard_owned = false;
        launch_mutex.unlock();
    }
}

void set_synchronous_collectives(bool enabled) {
    synchronous = enabled;
}

bool synchronous_collectives() {
    return synchronous;
}

void finish_collective() {
    if (synchronous) {
        release_launch_guard();
        infinicore::context::syncStream();
    }
}

} // namespace infinicore::op::distributed
