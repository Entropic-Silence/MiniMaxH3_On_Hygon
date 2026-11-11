#pragma once

namespace infinicore::op::distributed {

// Thread-local switch: when enabled, every collective issued by the calling
// thread waits on its stream until the collective has completed. Engines turn
// it on for their first (warm-up) forward so that lazily loaded GPU code
// objects are never uploaded to a device while a collective kernel is still
// spinning there waiting for peer ranks (observed as a whole-process hang on
// 4x K100-AI with HIP deferred code-object loading).
void set_synchronous_collectives(bool enabled);
bool synchronous_collectives();

// Called by collective operators right after enqueueing the collective.
void finish_collective();

// While the calling thread is in synchronous-collectives mode (an engine's
// first forward), operator launches of all such threads are serialized
// process-wide. The first launches lazily load GPU code objects, and on DTK
// 25.04.2 loading a code object while another thread launches a kernel can
// crash the runtime (segfault in hipLaunchKernel at address 0x20, seen in the
// Qwen3-VL and DiT first forwards). Collectives never hold the lock (they
// call release_launch_guard() before entering the communication library): on
// its first use RCCL connects peers inside the enqueue call and blocks until
// every rank has entered it, so a rank holding the lock there deadlocks the
// others (seen as a hang in the first DiT all-to-all). Outside first forwards
// the guard does nothing.
class FirstForwardLaunchGuard {
public:
    FirstForwardLaunchGuard();
    ~FirstForwardLaunchGuard();
    FirstForwardLaunchGuard(const FirstForwardLaunchGuard &) = delete;
    FirstForwardLaunchGuard &operator=(const FirstForwardLaunchGuard &) = delete;

private:
    bool counted_ = false;
};

// Releases the calling thread's launch lock early (before a collective).
void release_launch_guard();

} // namespace infinicore::op::distributed
