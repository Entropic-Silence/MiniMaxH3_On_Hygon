#ifndef __INFINICORE_TEST_MUTUAL_AWARENESS_H__
#define __INFINICORE_TEST_MUTUAL_AWARENESS_H__

#ifdef ENABLE_MUTUAL_AWARENESS

#include "test_runner.h"

namespace infinicore::test {

class MutualAwarenessTest : public TestFramework {
public:
    TestResult run() override;
    std::string getName() const override { return "MutualAwarenessTest"; }
};

} // namespace infinicore::test

#endif // ENABLE_MUTUAL_AWARENESS
#endif // __INFINICORE_TEST_MUTUAL_AWARENESS_H__
