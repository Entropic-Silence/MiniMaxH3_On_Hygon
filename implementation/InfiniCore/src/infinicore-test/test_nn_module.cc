#include "test_nn_module.h"
#include "infinicore/ops.hpp"

namespace infinicore::test {

// Helper function to format shape for logging
inline std::string formatShape(const std::vector<size_t> &shape) {
    std::ostringstream oss;
    oss << "[";
    for (size_t i = 0; i < shape.size(); ++i) {
        if (i > 0) {
            oss << ", ";
        }
        oss << shape[i];
    }
    oss << "]";
    return oss.str();
}

// Test 1: Basic module operations (creation, parameters, state_dict, load_state_dict)
TestResult NNModuleTest::testBasicModuleCreation() {
    return measureTime("BasicModuleOperations", [this]() {
        try {
            spdlog::info("==========================================");
            spdlog::info("Testing Basic Module Operations");
            spdlog::info("==========================================");

            // Test 1a: Module creation and parameter registration
            spdlog::info("Test 1a: Module creation and parameter registration");
            MockLinearModule module(8, 4, infinicore::Device());

            // Verify the module was created successfully
            auto state_dict = module.state_dict();
            if (state_dict.size() != 2) {
                spdlog::error("Expected 2 parameters, got {}", state_dict.size());
                return false;
            }

            // Test weight and bias parameters
            const auto &weight = module.get_weight();
            const auto &bias = module.get_bias();

            // Verify parameter shapes
            if (weight->shape() != std::vector<size_t>({4, 8})) {
                spdlog::error("Weight shape mismatch. Expected {{4, 8}}");
                return false;
            }

            if (bias->shape() != std::vector<size_t>({4})) {
                spdlog::error("Bias shape mismatch. Expected {{4}}");
                return false;
            }

            spdlog::info("✓ Module creation and parameter registration passed");

            // Test 1b: State dictionary functionality
            spdlog::info("Test 1b: State dictionary functionality");

            // Check if both parameters are in state dict
            if (state_dict.find("weight") == state_dict.end()) {
                spdlog::error("'weight' parameter not found in state dict");
                return false;
            }

            if (state_dict.find("bias") == state_dict.end()) {
                spdlog::error("'bias' parameter not found in state dict");
                return false;
            }

            spdlog::debug("State dict contains {} parameters:", state_dict.size());
            for (const auto &[name, tensor] : state_dict) {
                std::ostringstream shape_str;
                shape_str << "[";
                for (size_t i = 0; i < tensor->shape().size(); ++i) {
                    if (i > 0) {
                        shape_str << ", ";
                    }
                    shape_str << tensor->shape()[i];
                }
                shape_str << "]";
                spdlog::debug("  - {} with shape: {}", name, shape_str.str());
            }

            spdlog::info("✓ State dict functionality passed");

            // Test 1c: Load state dict functionality
            spdlog::info("Test 1c: Load state dict functionality");

            // Create new tensors to load
            auto new_weight = infinicore::Tensor::ones({4, 8}, infinicore::DataType::F32, infinicore::Device());
            auto new_bias = infinicore::Tensor::zeros({4}, infinicore::DataType::F32, infinicore::Device());

            // Load using load_parameter_
            module.load_parameter_("weight", new_weight);
            module.load_parameter_("bias", new_bias);

            // Verify the parameters were updated
            auto updated_state_dict = module.state_dict();
            if (!tensorsAllClose(updated_state_dict.at("weight"), new_weight, 1e-6, 1e-6)) {
                spdlog::error("Weight parameter values do not match after load_parameter_");
                return false;
            }
            if (!tensorsAllClose(updated_state_dict.at("bias"), new_bias, 1e-6, 1e-6)) {
                spdlog::error("Bias parameter values do not match after load_parameter_");
                return false;
            }

            // Test load_state_dict
            std::unordered_map<std::string, infinicore::Tensor> new_state_dict;
            new_state_dict.emplace("weight", infinicore::Tensor::ones({4, 8}, infinicore::DataType::F32, infinicore::Device()));
            new_state_dict.emplace("bias", infinicore::Tensor::ones({4}, infinicore::DataType::F32, infinicore::Device()));

            module.load_state_dict(new_state_dict);

            auto final_state_dict = module.state_dict();
            if (final_state_dict.size() != 2) {
                spdlog::error("State dict size mismatch after load_state_dict");
                return false;
            }

            spdlog::info("✓ Load state dict functionality passed");

            spdlog::info("=== All Basic Module Operations Passed ===");
            return true;
        } catch (const std::exception &e) {
            spdlog::error("Exception in testBasicModuleOperations: {}", e.what());
            return false;
        }
    });
}

TestResult NNModuleTest::testTensorParallelParameters() {
    return measureTime("TensorParallelParameters", [this]() {
        try {
            spdlog::info("==========================================");
            spdlog::info("Testing Tensor Parallel Parameters");
            spdlog::info("==========================================");

            auto device = infinicore::context::getDevice();

            spdlog::info("Test Tensor Parallel Parameter");
            // Case 1: Partition along dimension 0 (row-wise partitioning)
            infinicore::nn::Parameter param_dim0({8, 4}, infinicore::DataType::F32, device, 0, 0, 2);
            if (param_dim0->shape() != std::vector<size_t>({4, 4})) {
                spdlog::error("TP dim0: Expected shape [4, 4], got [{}]", formatShape(param_dim0->shape()));
                return false;
            }
            spdlog::info("✓ TP dim0 parameter created with correct partitioned shape");
            // Case 2: Partition along dimension 1 (column-wise partitioning)
            infinicore::nn::Parameter param_dim1({8, 4}, infinicore::DataType::F32, device, 1, 0, 2);
            if (param_dim1->shape() != std::vector<size_t>({8, 2})) {
                spdlog::error("TP dim1: Expected shape [8, 2], got [{}]", formatShape(param_dim1->shape()));
                return false;
            }
            spdlog::info("✓ TP dim1 parameter created with correct partitioned shape");
            spdlog::info("✓ Parameter creation with tensor parallelism passed");

            spdlog::info("Test Tensor Parallel Linear Module");
            auto w_data = std::vector<float>(32 * 64);
            auto b_data = std::vector<float>(32);
            for (size_t i = 0; i < 32; ++i) {
                for (size_t j = 0; j < 64; ++j) {
                    w_data[i * 64 + j] = static_cast<float>(j);
                }
                b_data[i] = static_cast<float>(i);
            }
            {
                spdlog::info("Test tp_size=4 tp_dim=0");
                Size tp_size = 4;
                Size tp_dim = 0;
                std::vector<std::unique_ptr<MockLinearModule>> tp_modules;

                for (Size tp_rank = 0; tp_rank < tp_size; ++tp_rank) {
                    auto module = std::make_unique<MockLinearModule>(64, 32, device, tp_dim, tp_rank, tp_size);
                    tp_modules.push_back(std::move(module));
                }

                // Verify each partition has correct shape
                for (size_t i = 0; i < tp_modules.size(); ++i) {
                    const auto &weight = tp_modules[i]->get_weight();
                    const auto &bias = tp_modules[i]->get_bias();

                    // Weight should be partitioned along output dimension (dim 0)
                    if (weight->shape() != std::vector<size_t>({8, 64})) { // 32/4 = 8
                        spdlog::error("TP rank {}: Weight shape mismatch. Expected [8, 64], got [{}]",
                                      i, formatShape(weight->shape()));
                        return false;
                    }

                    // Bias should be partitioned along output dimension
                    if (bias->shape() != std::vector<size_t>({8})) { // 32/4 = 8
                        spdlog::error("TP rank {}: Bias shape mismatch. Expected [8], got [{}]",
                                      i, formatShape(bias->shape()));
                        return false;
                    }

                    spdlog::debug("TP rank {}: weight shape [{}], bias shape [{}]",
                                  i, formatShape(weight->shape()), formatShape(bias->shape()));

                    tp_modules[i]->load_parameter_from_blob("weight", w_data.data());
                    tp_modules[i]->load_parameter_from_blob("bias", b_data.data());

                    auto weight_loaded = infinicore::Tensor::from_blob(
                                             w_data.data(),
                                             {32, 64},
                                             infinicore::DataType::F32,
                                             infinicore::Device::cpu())
                                             ->narrow({{0, i * 8, 8}})
                                             ->to(device); // Narrow to get the partition
                    auto bias_loaded = infinicore::Tensor::from_blob(
                                           b_data.data(),
                                           {32},
                                           infinicore::DataType::F32,
                                           infinicore::Device::cpu())
                                           ->narrow({{0, i * 8, 8}})
                                           ->to(device); // Narrow to get the partition

                    if (!tensorsAllClose(tp_modules[i]->get_weight(), weight_loaded, 1e-6, 1e-6)) {
                        spdlog::error("TP rank {}: Weight values do not match after load_parameter_from_blob", i);
                        return false;
                    }

                    if (!tensorsAllClose(tp_modules[i]->get_bias(), bias_loaded, 1e-6, 1e-6)) {
                        spdlog::error("TP rank {}: Bias values do not match after load_parameter_from_blob", i);
                        return false;
                    }
                }
            }

            {
                spdlog::info("Test tp_size=4 tp_dim=1");
                Size tp_size = 4;
                Size tp_dim = 1;
                std::vector<std::unique_ptr<MockLinearModule>> tp_modules;

                for (Size tp_rank = 0; tp_rank < tp_size; ++tp_rank) {
                    auto module = std::make_unique<MockLinearModule>(64, 32, device, tp_dim, tp_rank, tp_size);
                    tp_modules.push_back(std::move(module));
                }

                // Verify each partition has correct shape
                for (size_t i = 0; i < tp_modules.size(); ++i) {
                    const auto &weight = tp_modules[i]->get_weight();
                    const auto &bias = tp_modules[i]->get_bias();

                    // Weight should be partitioned along output dimension (dim 0)
                    if (weight->shape() != std::vector<size_t>({32, 16})) { // 64/4 = 16
                        spdlog::error("TP rank {}: Weight shape mismatch. Expected [32, 16], got [{}]",
                                      i, formatShape(weight->shape()));
                        return false;
                    }

                    // Bias should be partitioned along output dimension
                    if (bias->shape() != std::vector<size_t>({32})) { // Bias not partitioned when tp_dim=1
                        spdlog::error("TP rank {}: Bias shape mismatch. Expected [32], got [{}]",
                                      i, formatShape(bias->shape()));
                        return false;
                    }

                    spdlog::debug("TP rank {}: weight shape [{}], bias shape [{}]",
                                  i, formatShape(weight->shape()), formatShape(bias->shape()));
                    ;
                    tp_modules[i]->load_parameter_from_blob("weight", w_data.data());
                    tp_modules[i]->load_parameter_from_blob("bias", b_data.data());

                    auto weight_loaded = infinicore::Tensor::from_blob(
                                             w_data.data(),
                                             {32, 64},
                                             infinicore::DataType::F32,
                                             infinicore::Device::cpu())
                                             ->narrow({{1, i * 16, 16}})
                                             ->to(device); // Narrow to get the partition
                    auto bias_loaded = infinicore::Tensor::from_blob(
                                           b_data.data(),
                                           {32},
                                           infinicore::DataType::F32,
                                           infinicore::Device::cpu())
                                           ->to(device); // Narrow to get the partition
                    if (!tensorsAllClose(tp_modules[i]->get_weight(), weight_loaded, 1e-6, 1e-6)) {
                        spdlog::error("TP rank {}: Weight values do not match after load_parameter_from_blob", i);
                        return false;
                    }
                    if (!tensorsAllClose(tp_modules[i]->get_bias(), bias_loaded, 1e-6, 1e-6)) {
                        spdlog::error("TP rank {}: Bias values do not match after load_parameter_from_blob", i);
                        return false;
                    }
                }
            }

            spdlog::info("=== All Tensor Parallel Parameter Tests Passed ===");
            return true;

        } catch (const std::exception &e) {
            spdlog::error("Exception in testTensorParallelParameters: {}", e.what());
            return false;
        }
    });
}

// Test 2: Advanced load state dict functionality (hierarchical modules)
TestResult NNModuleTest::testLoadStateDict() {
    return measureTime("AdvancedLoadStateDict", [this]() {
        try {
            spdlog::info("==========================================");
            spdlog::info("Testing Advanced load_state_dict with Hierarchical Modules");
            spdlog::info("==========================================");

            // Test: Deep nesting (2-level hierarchy)
            spdlog::info("Test 4: Testing load_state_dict with 2-level deep nesting");

            // Create parent -> child -> grandchild hierarchy using proper module definition
            class DeepGrandchildModule : public infinicore::nn::Module {
            protected:
                INFINICORE_NN_MODULE(MockLinearModule, sublayer);

            public:
                DeepGrandchildModule() {
                    INFINICORE_NN_MODULE_INIT(sublayer, 6, 4, infinicore::Device());
                }
            };

            class DeepChildModule : public infinicore::nn::Module {
            protected:
                INFINICORE_NN_MODULE(MockLinearModule, own_layer);
                INFINICORE_NN_MODULE(DeepGrandchildModule, sublayer);

            public:
                DeepChildModule() {
                    INFINICORE_NN_MODULE_INIT(own_layer, 8, 6, infinicore::Device());
                    INFINICORE_NN_MODULE_INIT(sublayer);
                }
            };

            class DeepParentModule : public infinicore::nn::Module {
            protected:
                INFINICORE_NN_MODULE(MockLinearModule, own_layer);
                INFINICORE_NN_MODULE(DeepChildModule, layer1);

            public:
                DeepParentModule() {
                    INFINICORE_NN_MODULE_INIT(own_layer, 10, 8, infinicore::Device());
                    INFINICORE_NN_MODULE_INIT(layer1);
                }
            };

            DeepParentModule deep_parent;

            // Verify initial state dict includes all 2-level hierarchical parameters
            auto deep_initial_state = deep_parent.state_dict();
            spdlog::debug("Deep hierarchical state dict has {} parameters", deep_initial_state.size());

            // Expected parameters:
            // parent: own_layer.weight, own_layer.bias (2)
            // layer1: layer1.own_layer.weight, layer1.own_layer.bias (2)
            // sublayer: layer1.sublayer.sublayer.weight, layer1.sublayer.sublayer.bias (2)
            // Total: 6 parameters
            if (deep_initial_state.size() < 6) {
                spdlog::error("Deep hierarchy state dict size mismatch. Expected at least 6, got {}",
                              deep_initial_state.size());
                return false;
            }

            // Verify 2-level parameter names exist
            bool has_sublayer_weight = deep_initial_state.find("layer1.sublayer.sublayer.weight") != deep_initial_state.end();
            bool has_sublayer_bias = deep_initial_state.find("layer1.sublayer.sublayer.bias") != deep_initial_state.end();

            if (!has_sublayer_weight || !has_sublayer_bias) {
                spdlog::error("2-level nested parameters missing from state dict");
                return false;
            }
            spdlog::debug("All 2-level hierarchical parameter names verified");

            // Create state dict for 2-level hierarchy with all 1.0 values
            std::unordered_map<std::string, infinicore::Tensor> deep_state_dict;
            deep_state_dict.emplace("own_layer.weight", infinicore::Tensor::ones({8, 10}, infinicore::DataType::F32, infinicore::Device()));
            deep_state_dict.emplace("own_layer.bias", infinicore::Tensor::ones({8}, infinicore::DataType::F32, infinicore::Device()));
            deep_state_dict.emplace("layer1.own_layer.weight", infinicore::Tensor::ones({6, 8}, infinicore::DataType::F32, infinicore::Device()));
            deep_state_dict.emplace("layer1.own_layer.bias", infinicore::Tensor::ones({6}, infinicore::DataType::F32, infinicore::Device()));
            deep_state_dict.emplace("layer1.sublayer.sublayer.weight", infinicore::Tensor::ones({4, 6}, infinicore::DataType::F32, infinicore::Device()));
            deep_state_dict.emplace("layer1.sublayer.sublayer.bias", infinicore::Tensor::ones({4}, infinicore::DataType::F32, infinicore::Device()));

            // Load the deep hierarchical state dict
            deep_parent.load_state_dict(deep_state_dict);
            spdlog::debug("Successfully loaded 2-level deep hierarchical state dict");

            // Verify all parameters were loaded correctly
            auto deep_loaded_state = deep_parent.state_dict();

            // Verify shapes at all levels
            if (deep_loaded_state.at("own_layer.weight")->shape() != std::vector<size_t>({8, 10})) {
                spdlog::error("Deep parent weight shape mismatch");
                return false;
            }
            if (deep_loaded_state.at("layer1.own_layer.weight")->shape() != std::vector<size_t>({6, 8})) {
                spdlog::error("Deep layer1 weight shape mismatch");
                return false;
            }
            if (deep_loaded_state.at("layer1.sublayer.sublayer.weight")->shape() != std::vector<size_t>({4, 6})) {
                spdlog::error("Deep sublayer weight shape mismatch");
                return false;
            }
            spdlog::debug("All 2-level deep parameter shapes verified");

            // Verify actual weight loading correctness by checking that loaded parameters
            // match what we provided in the state dict (use the original tensors)
            spdlog::info("Verifying weight loading correctness by direct comparison");

            // Get the tensors we loaded from the state dict
            auto loaded_parent_weight = deep_loaded_state.at("own_layer.weight");
            auto loaded_layer1_weight = deep_loaded_state.at("layer1.own_layer.weight");
            auto loaded_sublayer_weight = deep_loaded_state.at("layer1.sublayer.sublayer.weight");

            // Compare with the original tensors we put in the state dict
            if (!tensorsAllClose(loaded_parent_weight, deep_state_dict.at("own_layer.weight"), 1e-5, 1e-5)) {
                spdlog::error("Deep parent weight not preserved after loading");
                return false;
            }
            if (!tensorsAllClose(loaded_layer1_weight, deep_state_dict.at("layer1.own_layer.weight"), 1e-5, 1e-5)) {
                spdlog::error("Deep layer1 weight not preserved after loading");
                return false;
            }
            if (!tensorsAllClose(loaded_sublayer_weight, deep_state_dict.at("layer1.sublayer.sublayer.weight"), 1e-5, 1e-5)) {
                spdlog::error("Deep sublayer weight not preserved after loading");
                return false;
            }

            spdlog::info("✓ Weight loading correctness verified - loaded values match input state dict");
            spdlog::info("✓ 2-level deep hierarchy load_state_dict verification passed");

            spdlog::info("=== All Advanced load_state_dict Tests Passed ===");
            return true;
        } catch (const std::exception &e) {
            spdlog::error("Exception in testLoadStateDict: {}", e.what());
            return false;
        }
    });
}

// Test 3: Module hierarchy (demonstrates proper hierarchical construction pattern)
TestResult NNModuleTest::testModuleHierarchy() {
    return measureTime("ModuleHierarchy", [this]() {
        try {
            // Create a hierarchy using proper module definition: root -> layer1 -> layer2
            class Layer2Module : public infinicore::nn::Module {
            protected:
                INFINICORE_NN_MODULE(MockLinearModule, sublayer);

            public:
                Layer2Module() {
                    INFINICORE_NN_MODULE_INIT(sublayer, 8, 4, infinicore::Device());
                }
            };

            class Layer1Module : public infinicore::nn::Module {
            protected:
                INFINICORE_NN_MODULE(MockLinearModule, sublayer);
                INFINICORE_NN_MODULE(Layer2Module, layer2);

            public:
                Layer1Module() {
                    INFINICORE_NN_MODULE_INIT(sublayer, 16, 8, infinicore::Device());
                    INFINICORE_NN_MODULE_INIT(layer2);
                }
            };

            class RootModule : public infinicore::nn::Module {
            protected:
                INFINICORE_NN_MODULE(MockLinearModule, root_layer);
                INFINICORE_NN_MODULE(Layer1Module, layer1);

            public:
                RootModule() {
                    INFINICORE_NN_MODULE_INIT(root_layer, 20, 16, infinicore::Device());
                    INFINICORE_NN_MODULE_INIT(layer1);
                }
            };

            RootModule root_module;

            // Check the complete state dict
            auto root_state_dict = root_module.state_dict();

            // Debug: Print all parameters
            spdlog::debug("Found {} parameters:", root_state_dict.size());
            for (const auto &pair : root_state_dict) {
                spdlog::debug("  - {}", pair.first);
            }

            // Should have: root_layer.weight, root_layer.bias,
            // layer1.sublayer.weight, layer1.sublayer.bias,
            // layer1.layer2.sublayer.weight, layer1.layer2.sublayer.bias
            if (root_state_dict.size() < 6) {
                spdlog::error("Error: Expected at least 6 parameters in hierarchy, got {}", root_state_dict.size());
                return false;
            }

            spdlog::info("Module hierarchy test passed. Root state dict has {} parameters", root_state_dict.size());

            // Print the hierarchy
            std::cout << "Module hierarchy:" << std::endl;
            for (const auto &pair : root_state_dict) {
                std::cout << "  - " << pair.first << std::endl;
            }

            // Additional: Test INFINICORE_NN_MODULE_VEC vector registration
            spdlog::info("Testing INFINICORE_NN_MODULE_VEC (vector of submodules)");
            class VecModule : public infinicore::nn::Module {
            protected:
                INFINICORE_NN_MODULE_VEC(MockLinearModule, layers);

            public:
                VecModule() {
                    INFINICORE_NN_MODULE_VEC_INIT(layers, 3, MockLinearModule, 16, 8, infinicore::Device());
                }
            };

            VecModule vec_mod;
            auto vec_state = vec_mod.state_dict();

            // Expect parameters for layers.0, layers.1, layers.2 (weight and bias for each)
            std::vector<std::string> expected_vec_params = {
                "layers.0.weight", "layers.0.bias",
                "layers.1.weight", "layers.1.bias",
                "layers.2.weight", "layers.2.bias"};

            for (const auto &param : expected_vec_params) {
                if (vec_state.find(param) == vec_state.end()) {
                    spdlog::error("INFINICORE_NN_MODULE_VEC: missing '{}' in state_dict", param);
                    return false;
                }
            }

            spdlog::info("INFINICORE_NN_MODULE_VEC test passed - found all vector layer parameters");

            return true;
        } catch (const std::exception &e) {
            spdlog::error("Exception in testModuleHierarchy: {}", e.what());
            return false;
        }
    });
}

// Test 4: Parameter loading from blob
TestResult NNModuleTest::testParameterLoading() {
    return measureTime("ParameterLoading", [this]() {
        try {
            spdlog::info("==========================================");
            spdlog::info("Testing Parameter loading from blob");
            spdlog::info("==========================================");
            MockLinearModule module(3, 2, infinicore::Device());

            // Create test data
            std::vector<float> weight_data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
            std::vector<float> bias_data = {0.1f, 0.2f};

            // Load parameters from blob data
            module.load_parameter_from_blob("weight", weight_data.data());
            module.load_parameter_from_blob("bias", bias_data.data());

            spdlog::info("Successfully loaded parameters from blob data");

            // Verify parameters exist
            auto state_dict = module.state_dict();
            if (state_dict.find("weight") == state_dict.end() || state_dict.find("bias") == state_dict.end()) {
                spdlog::error("Error: Parameters not found after loading");
                return false;
            }

            MockLinearModule module_row_parallel(3, 2, infinicore::Device(), 0, 1, 2);

            spdlog::info("Parameter loading test passed");
            return true;
        } catch (const std::exception &e) {
            spdlog::error("Exception in testParameterLoading: {}", e.what());
            return false;
        }
    });
}

// Test 6: Embedding module implementation
TestResult NNModuleTest::testModuleEmbedding() {
    return measureTime("ModuleEmbedding", [this]() {
        try {
            spdlog::info("==========================================");
            spdlog::info("Testing Embedding module implementation");
            spdlog::info("==========================================");

            // Test 1: Basic embedding creation
            spdlog::info("Test 1: Basic embedding creation (vocab=100, dim=64)");
            infinicore::nn::Embedding emb1(100, 64);

            auto state1 = emb1.state_dict();
            if (state1.find("weight") == state1.end()) {
                spdlog::error("Embedding weight not found in state dict");
                return false;
            }

            if (state1.at("weight")->shape() != std::vector<size_t>({100, 64})) {
                spdlog::error("Embedding weight shape mismatch. Expected {{100, 64}}");
                return false;
            }

            if (emb1.num_embeddings() != 100) {
                spdlog::error("num_embeddings mismatch. Expected 100, got {}", emb1.num_embeddings());
                return false;
            }

            if (emb1.embedding_dim() != 64) {
                spdlog::error("embedding_dim mismatch. Expected 64, got {}", emb1.embedding_dim());
                return false;
            }

            spdlog::debug("Basic embedding creation passed");

            // Test 2: Embedding with padding_idx
            spdlog::info("Test 2: Embedding with padding_idx=0");
            infinicore::nn::Embedding emb2(50, 32, 0, infinicore::DataType::F32, infinicore::Device());

            if (!emb2.padding_idx().has_value()) {
                spdlog::error("padding_idx should have a value");
                return false;
            }

            if (emb2.padding_idx().value() != 0) {
                spdlog::error("padding_idx mismatch. Expected 0, got {}", emb2.padding_idx().value());
                return false;
            }

            spdlog::debug("Embedding with padding_idx passed");

            // Test 3: Forward pass - single index
            spdlog::info("Test 3: Forward pass with single index");
            std::vector<int64_t> single_data = {5};
            auto indices_single = infinicore::Tensor::from_blob(single_data.data(), {1}, infinicore::DataType::I64, infinicore::Device());
            auto output_single = emb1.forward(indices_single);

            if (output_single->shape() != std::vector<size_t>({1, 64})) {
                spdlog::error("Single index output shape mismatch. Expected {{1, 64}}");
                return false;
            }

            spdlog::debug("Single index forward pass passed. Output shape: {{1, 64}}");

            // Test 4: Forward pass - batch of indices
            spdlog::info("Test 4: Forward pass with batch of indices");
            std::vector<int64_t> batch_data = {0, 5, 10};
            auto indices_batch = infinicore::Tensor::from_blob(batch_data.data(), {3}, infinicore::DataType::I64, infinicore::Device());
            auto output_batch = emb1.forward(indices_batch);

            if (output_batch->shape() != std::vector<size_t>({3, 64})) {
                spdlog::error("Batch output shape mismatch. Expected {{3, 64}}");
                return false;
            }

            spdlog::debug("Batch forward pass passed. Output shape: {{3, 64}}");

            // Test 5: Forward pass - 2D indices (batch_size, seq_len)
            spdlog::info("Test 5: Forward pass with 2D indices [batch, seq_len]");
            std::vector<int64_t> data_2d = {1, 2, 3, 4, 5, 6, 7, 8};
            auto indices_2d = infinicore::Tensor::from_blob(data_2d.data(), {2, 4},
                                                            infinicore::DataType::I64, infinicore::Device());
            auto output_2d = emb1.forward(indices_2d);

            if (output_2d->shape() != std::vector<size_t>({2, 4, 64})) {
                spdlog::error("2D indices output shape mismatch. Expected {{2, 4, 64}}");
                return false;
            }

            spdlog::debug("2D indices forward pass passed. Output shape: {{2, 4, 64}}");

            // Test 6: Embedding lookup consistency
            spdlog::info("Test 6: Testing embedding lookup consistency");
            std::vector<int64_t> idx_data = {7};
            auto idx1 = infinicore::Tensor::from_blob(idx_data.data(), {1}, infinicore::DataType::I64, infinicore::Device());
            auto idx2 = infinicore::Tensor::from_blob(idx_data.data(), {1}, infinicore::DataType::I64, infinicore::Device());

            auto out1 = emb1.forward(idx1);
            auto out2 = emb1.forward(idx2);

            // Same index should give same embedding
            if (!tensorsAllClose(out1, out2, 1e-7, 1e-7)) {
                spdlog::error("Same index should return identical embeddings");
                return false;
            }

            spdlog::debug("Embedding lookup consistency passed");

            // Test 7: load_state_dict
            spdlog::info("Test 7: Testing load_state_dict for Embedding");
            auto new_weight = infinicore::Tensor::ones({100, 64}, infinicore::DataType::F32, infinicore::Device());
            std::unordered_map<std::string, infinicore::Tensor> new_state;
            new_state.emplace("weight", new_weight);

            emb1.load_state_dict(new_state);

            if (!tensorsAllClose(emb1.weight(), new_weight, 1e-7, 1e-7)) {
                spdlog::error("Embedding weight not loaded correctly");
                return false;
            }

            spdlog::debug("load_state_dict for Embedding passed");

            // Test 8: extra_repr
            spdlog::info("Test 8: Testing extra_repr");
            std::string repr1 = emb1.extra_repr();
            std::string repr2 = emb2.extra_repr();

            spdlog::debug("Embedding repr (no padding): {}", repr1);
            spdlog::debug("Embedding repr (with padding): {}", repr2);

            if (repr1.find("num_embeddings=100") == std::string::npos) {
                spdlog::error("extra_repr should contain num_embeddings");
                return false;
            }

            if (repr2.find("padding_idx=0") == std::string::npos) {
                spdlog::error("extra_repr should contain padding_idx when specified");
                return false;
            }

            spdlog::debug("extra_repr test passed");

            spdlog::info("All Embedding module tests passed!");
            return true;

        } catch (const std::exception &e) {
            spdlog::error("Exception in testModuleEmbedding: {}", e.what());
            return false;
        }
    });
}

// Test 7: RMSNorm module implementation
TestResult NNModuleTest::testModuleRMSNorm() {
    return measureTime("ModuleRMSNorm", [this]() {
        try {
            spdlog::info("==========================================");
            spdlog::info("Testing RMSNorm module implementation");
            spdlog::info("==========================================");

            // Test 1: Basic RMSNorm creation
            spdlog::info("Test 1: Basic RMSNorm creation (hidden_size=768)");
            infinicore::nn::RMSNorm norm1(768);

            auto state1 = norm1.state_dict();
            if (state1.find("weight") == state1.end()) {
                spdlog::error("RMSNorm weight not found in state dict");
                return false;
            }

            if (state1.at("weight")->shape() != std::vector<size_t>({768})) {
                spdlog::error("RMSNorm weight shape mismatch. Expected {{768}}");
                return false;
            }

            if (norm1.normalized_shape() != 768) {
                spdlog::error("normalized_shape mismatch. Expected 768, got {}", norm1.normalized_shape());
                return false;
            }

            spdlog::debug("Basic RMSNorm creation passed");

            // Test 2: Forward pass - 2D input [batch, hidden]
            spdlog::info("Test 2: Forward pass with 2D input [batch, hidden]");
            auto input_2d = infinicore::Tensor::ones({4, 768}, infinicore::DataType::F32, infinicore::Device());
            auto output_2d = norm1.forward(input_2d);

            if (output_2d->shape() != std::vector<size_t>({4, 768})) {
                spdlog::error("2D output shape mismatch. Expected {{4, 768}}");
                return false;
            }

            spdlog::debug("2D forward pass passed. Output shape: {{4, 768}}");

            // Test 3: Forward pass - 3D input [batch, seq_len, hidden]
            spdlog::info("Test 3: Forward pass with 3D input [batch, seq_len, hidden]");
            auto input_3d = infinicore::Tensor::ones({2, 10, 768}, infinicore::DataType::F32, infinicore::Device());
            auto output_3d = norm1.forward(input_3d);

            if (output_3d->shape() != std::vector<size_t>({2, 10, 768})) {
                spdlog::error("3D output shape mismatch. Expected {{2, 10, 768}}");
                return false;
            }

            spdlog::debug("3D forward pass passed. Output shape: {{2, 10, 768}}");

            // Test 4: Test normalization properties
            spdlog::info("Test 4: Testing RMSNorm properties");
            auto test_input = infinicore::Tensor::ones({1, 768}, infinicore::DataType::F32, infinicore::Device());
            auto test_output = norm1.forward(test_input);

            // Output should have same shape
            if (test_output->shape() != test_input->shape()) {
                spdlog::error("Output shape doesn't match input shape");
                return false;
            }

            spdlog::debug("RMSNorm properties test passed");

            // Test 5: load_state_dict
            spdlog::info("Test 5: Testing load_state_dict for RMSNorm");
            auto new_weight = infinicore::Tensor::ones({768}, infinicore::DataType::F32, infinicore::Device());
            std::unordered_map<std::string, infinicore::Tensor> new_state;
            new_state.emplace("weight", new_weight);

            norm1.load_state_dict(new_state);

            if (!tensorsAllClose(norm1.weight(), new_weight, 1e-7, 1e-7)) {
                spdlog::error("RMSNorm weight not loaded correctly");
                return false;
            }

            spdlog::debug("load_state_dict for RMSNorm passed");

            // Test 6: extra_repr
            spdlog::info("Test 6: Testing extra_repr");
            std::string repr = norm1.extra_repr();
            spdlog::debug("RMSNorm repr: {}", repr);

            if (repr.find("normalized_shape=768") == std::string::npos) {
                spdlog::error("extra_repr should contain normalized_shape");
                return false;
            }

            if (repr.find("eps=") == std::string::npos) {
                spdlog::error("extra_repr should contain eps");
                return false;
            }

            spdlog::debug("extra_repr test passed");

            // Test 7: Input validation - normalized_shape mismatch (op layer handles this)
            spdlog::info("Test 7: Testing input validation - normalized_shape mismatch");
            auto input_wrong_shape = infinicore::Tensor::ones({4, 512}, infinicore::DataType::F32, infinicore::Device()); // normalized_shape=512, expected 768

            try {
                norm1.forward(input_wrong_shape);
                spdlog::error("Should have thrown exception for normalized_shape mismatch");
                return false;
            } catch (const std::exception &e) {
                spdlog::debug("Correctly caught exception for normalized_shape mismatch (handled by op layer): {}", e.what());
            } catch (...) {
                spdlog::error("Caught unexpected exception type");
                return false;
            }

            spdlog::debug("Normalized_shape mismatch validation test passed");

            // Test 8: Different hidden sizes
            spdlog::info("Test 8: Testing different hidden sizes");
            infinicore::nn::RMSNorm norm_small(128, 1e-5);
            infinicore::nn::RMSNorm norm_large(4096);

            auto input_small = infinicore::Tensor::ones({2, 128}, infinicore::DataType::F32, infinicore::Device());
            auto output_small = norm_small.forward(input_small);

            auto input_large = infinicore::Tensor::ones({2, 4096}, infinicore::DataType::F32, infinicore::Device());
            auto output_large = norm_large.forward(input_large);

            if (output_small->shape() != std::vector<size_t>({2, 128})) {
                spdlog::error("Small RMSNorm output shape mismatch");
                return false;
            }

            if (output_large->shape() != std::vector<size_t>({2, 4096})) {
                spdlog::error("Large RMSNorm output shape mismatch");
                return false;
            }

            spdlog::debug("Different hidden sizes test passed");

            spdlog::info("All RMSNorm module tests passed!");
            return true;

        } catch (const std::exception &e) {
            spdlog::error("Exception in testModuleRMSNorm: {}", e.what());
            return false;
        }
    });
}

// Test 7.5: RoPE module test
TestResult NNModuleTest::testModuleRoPE() {
    return measureTime("ModuleRoPE", [this]() {
        try {
            spdlog::info("==========================================");
            spdlog::info("Testing RoPE module implementation");
            spdlog::info("==========================================");

            // Test 1: Basic RoPE creation
            spdlog::info("Test 1: Basic RoPE creation (head_dim=128, max_seq_len=2048)");
            infinicore::nn::RoPE rope1(128, 2048);

            auto state1 = rope1.state_dict();

            if (rope1.head_dim() != 128) {
                spdlog::error("head_dim mismatch. Expected 128, got {}", rope1.head_dim());
                return false;
            }

            if (rope1.max_seq_len() != 2048) {
                spdlog::error("max_seq_len mismatch. Expected 2048, got {}", rope1.max_seq_len());
                return false;
            }

            spdlog::debug("Basic RoPE creation passed");

            // Test 2: Forward pass - 3D input [seq_len, n_head, head_dim]
            spdlog::info("Test 2: Forward pass with 3D input [seq_len, n_head, head_dim]");
            auto x_3d = infinicore::Tensor::ones({32, 32, 128}, infinicore::DataType::F32, infinicore::Device());

            // Create position tensor [0, 1, 2, ..., 31]
            std::vector<int32_t> pos_data(32);
            for (size_t i = 0; i < 32; i++) {
                pos_data[i] = static_cast<int32_t>(i);
            }
            auto pos = infinicore::Tensor::from_blob(pos_data.data(), {32}, infinicore::DataType::I32, infinicore::Device());

            auto x_out = rope1.forward(x_3d, pos);

            if (x_out->shape() != std::vector<size_t>({32, 32, 128})) {
                spdlog::error("3D output shape mismatch. Expected {{32, 32, 128}}");
                return false;
            }

            spdlog::debug("3D forward pass passed. Output shape: {{32, 32, 128}}");

            // Test 3: Different algorithms
            spdlog::info("Test 3: Testing different algorithms");
            infinicore::nn::RoPE rope_gptj(64, 1024, 10000.0, infinicore::nn::RoPE::Algo::GPT_J);
            infinicore::nn::RoPE rope_gptneox(64, 1024, 10000.0, infinicore::nn::RoPE::Algo::GPT_NEOX);

            if (rope_gptj.algo() != infinicore::nn::RoPE::Algo::GPT_J) {
                spdlog::error("GPT_J algorithm not set correctly");
                return false;
            }

            if (rope_gptneox.algo() != infinicore::nn::RoPE::Algo::GPT_NEOX) {
                spdlog::error("GPT_NEOX algorithm not set correctly");
                return false;
            }

            auto x_test = infinicore::Tensor::ones({10, 32, 64}, infinicore::DataType::F32, infinicore::Device());

            std::vector<int32_t> pos_test_data(10);
            for (size_t i = 0; i < 10; i++) {
                pos_test_data[i] = static_cast<int32_t>(i);
            }
            auto pos_test = infinicore::Tensor::from_blob(pos_test_data.data(), {10}, infinicore::DataType::I32, infinicore::Device());

            auto x_gptj = rope_gptj.forward(x_test, pos_test);
            auto x_neox = rope_gptneox.forward(x_test, pos_test);

            if (x_gptj->shape() != x_test->shape()) {
                spdlog::error("GPT_J forward pass shape mismatch");
                return false;
            }

            if (x_neox->shape() != x_test->shape()) {
                spdlog::error("GPT_NEOX forward pass shape mismatch");
                return false;
            }

            spdlog::debug("Different algorithms test passed");

            // Test 4: Different theta values
            spdlog::info("Test 4: Testing different theta values");
            infinicore::nn::RoPE rope_theta1(128, 2048, 1e5);
            infinicore::nn::RoPE rope_theta2(128, 2048, 1e4);

            if (rope_theta1.theta() != 1e5) {
                spdlog::error("theta mismatch. Expected 1e5, got {}", rope_theta1.theta());
                return false;
            }

            if (rope_theta2.theta() != 1e4) {
                spdlog::error("theta mismatch. Expected 1e4, got {}", rope_theta2.theta());
                return false;
            }

            spdlog::debug("Different theta values test passed");

            // Test 5: load_state_dict
            std::unordered_map<std::string, infinicore::Tensor> new_state;
            rope1.load_state_dict(new_state);
            spdlog::debug("load_state_dict for RoPE passed (no parameters to load)");

            // Test 6: extra_repr
            spdlog::info("Test 6: Testing extra_repr");
            std::string repr = rope1.extra_repr();
            spdlog::debug("RoPE repr: {}", repr);

            if (repr.find("head_dim=128") == std::string::npos) {
                spdlog::error("extra_repr should contain head_dim");
                return false;
            }

            if (repr.find("max_seq_len=2048") == std::string::npos) {
                spdlog::error("extra_repr should contain max_seq_len");
                return false;
            }

            if (repr.find("theta=") == std::string::npos) {
                spdlog::error("extra_repr should contain theta");
                return false;
            }

            spdlog::debug("extra_repr test passed");

            // Test 7: Different head dimensions
            spdlog::info("Test 7: Testing different head dimensions");
            infinicore::nn::RoPE rope_small(64, 1024);
            infinicore::nn::RoPE rope_large(256, 4096);

            auto x_small = infinicore::Tensor::ones({10, 32, 64}, infinicore::DataType::F32, infinicore::Device());

            std::vector<int32_t> pos_small_data(10);
            for (size_t i = 0; i < 10; i++) {
                pos_small_data[i] = static_cast<int32_t>(i);
            }
            auto pos_small = infinicore::Tensor::from_blob(pos_small_data.data(), {10}, infinicore::DataType::I32, infinicore::Device());

            auto x_small_out = rope_small.forward(x_small, pos_small);

            if (x_small_out->shape() != std::vector<size_t>({10, 32, 64})) {
                spdlog::error("Small RoPE output shape mismatch");
                return false;
            }

            auto x_large = infinicore::Tensor::ones({20, 32, 256}, infinicore::DataType::F32, infinicore::Device());

            std::vector<int32_t> pos_large_data(20);
            for (size_t i = 0; i < 20; i++) {
                pos_large_data[i] = static_cast<int32_t>(i);
            }
            auto pos_large = infinicore::Tensor::from_blob(pos_large_data.data(), {20}, infinicore::DataType::I32, infinicore::Device());

            auto x_large_out = rope_large.forward(x_large, pos_large);

            if (x_large_out->shape() != std::vector<size_t>({20, 32, 256})) {
                spdlog::error("Large RoPE output shape mismatch");
                return false;
            }

            spdlog::debug("Different head dimensions test passed");

            // Test 8: Invalid head_dim (odd number)
            spdlog::info("Test 8: Testing invalid head_dim (odd number)");
            try {
                infinicore::nn::RoPE rope_invalid(127, 2048);
                spdlog::error("Should have thrown exception for odd head_dim");
                return false;
            } catch (const std::invalid_argument &e) {
                spdlog::debug("Correctly caught exception for odd head_dim: {}", e.what());
            } catch (...) {
                spdlog::error("Caught unexpected exception type");
                return false;
            }

            spdlog::debug("Invalid head_dim test passed");

            // Test 9: Input validation - empty tensor (op layer handles this)
            spdlog::info("Test 9: Testing input validation - empty tensor");
            auto x_empty = infinicore::Tensor::ones({}, infinicore::DataType::F32, infinicore::Device());
            std::vector<int32_t> pos_empty_data(1);
            pos_empty_data[0] = 0;
            auto pos_empty = infinicore::Tensor::from_blob(pos_empty_data.data(), {1}, infinicore::DataType::I32, infinicore::Device());

            try {
                rope1.forward(x_empty, pos_empty);
                spdlog::error("Should have thrown exception for empty input tensor");
                return false;
            } catch (const std::exception &e) {
                spdlog::debug("Correctly caught exception for empty input (handled by op layer): {}", e.what());
            } catch (...) {
                spdlog::error("Caught unexpected exception type");
                return false;
            }

            spdlog::debug("Empty tensor validation test passed");

            // Test 10: Input validation - head_dim mismatch (op layer handles this)
            spdlog::info("Test 10: Testing input validation - head_dim mismatch");
            auto x_wrong_dim = infinicore::Tensor::ones({32, 32, 64}, infinicore::DataType::F32, infinicore::Device()); // head_dim=64, expected 128
            std::vector<int32_t> pos_wrong_data(32);
            for (size_t i = 0; i < 32; i++) {
                pos_wrong_data[i] = static_cast<int32_t>(i);
            }
            auto pos_wrong = infinicore::Tensor::from_blob(pos_wrong_data.data(), {32}, infinicore::DataType::I32, infinicore::Device());

            try {
                rope1.forward(x_wrong_dim, pos_wrong);
                spdlog::error("Should have thrown exception for head_dim mismatch");
                return false;
            } catch (const std::exception &e) {
                spdlog::debug("Correctly caught exception for head_dim mismatch (handled by op layer): {}", e.what());
            } catch (...) {
                spdlog::error("Caught unexpected exception type");
                return false;
            }

            spdlog::debug("Head_dim mismatch validation test passed");

            // Test 11: Different input shapes (from reference test cases)
            spdlog::info("Test 11: Testing different input shapes");

            // Test shape (1, 32, 128) - single sequence
            auto x_single = infinicore::Tensor::ones({1, 32, 128}, infinicore::DataType::F32, infinicore::Device());
            std::vector<int32_t> pos_single_data(1);
            pos_single_data[0] = 0;
            auto pos_single = infinicore::Tensor::from_blob(pos_single_data.data(), {1}, infinicore::DataType::I32, infinicore::Device());
            auto x_single_out = rope1.forward(x_single, pos_single);
            if (x_single_out->shape() != std::vector<size_t>({1, 32, 128})) {
                spdlog::error("Single sequence output shape mismatch");
                return false;
            }

            // Test shape (10, 32, 64) - different head_dim
            auto rope_64 = infinicore::nn::RoPE(64, 1024);
            auto x_64 = infinicore::Tensor::ones({10, 32, 64}, infinicore::DataType::F32, infinicore::Device());
            std::vector<int32_t> pos_64_data(10);
            for (size_t i = 0; i < 10; i++) {
                pos_64_data[i] = static_cast<int32_t>(i);
            }
            auto pos_64 = infinicore::Tensor::from_blob(pos_64_data.data(), {10}, infinicore::DataType::I32, infinicore::Device());
            auto x_64_out = rope_64.forward(x_64, pos_64);
            if (x_64_out->shape() != std::vector<size_t>({10, 32, 64})) {
                spdlog::error("Shape (10, 32, 64) output mismatch");
                return false;
            }

            // Test shape (4, 1, 32) - single head
            auto rope_32 = infinicore::nn::RoPE(32, 1024);
            auto x_1head = infinicore::Tensor::ones({4, 1, 32}, infinicore::DataType::F32, infinicore::Device());
            std::vector<int32_t> pos_1head_data(4);
            for (size_t i = 0; i < 4; i++) {
                pos_1head_data[i] = static_cast<int32_t>(i);
            }
            auto pos_1head = infinicore::Tensor::from_blob(pos_1head_data.data(), {4}, infinicore::DataType::I32, infinicore::Device());
            auto x_1head_out = rope_32.forward(x_1head, pos_1head);
            if (x_1head_out->shape() != std::vector<size_t>({4, 1, 32})) {
                spdlog::error("Shape (4, 1, 32) output mismatch");
                return false;
            }

            spdlog::debug("Different input shapes test passed");

            // Test 12: Position tensor validation
            spdlog::info("Test 12: Testing position tensor edge cases");

            // Test with seq_len less than max_seq_len
            auto x_short = infinicore::Tensor::ones({10, 32, 128}, infinicore::DataType::F32, infinicore::Device());
            std::vector<int32_t> pos_short_data(10);
            for (size_t i = 0; i < 10; i++) {
                pos_short_data[i] = static_cast<int32_t>(i);
            }
            auto pos_short = infinicore::Tensor::from_blob(pos_short_data.data(), {10}, infinicore::DataType::I32, infinicore::Device());
            auto x_short_out = rope1.forward(x_short, pos_short);
            if (x_short_out->shape() != std::vector<size_t>({10, 32, 128})) {
                spdlog::error("Short sequence output shape mismatch");
                return false;
            }

            spdlog::debug("Position tensor edge cases test passed");

            // Test 13: Test that outputs are on the same device as inputs
            spdlog::info("Test 13: Testing device consistency");
            auto device = x_3d->device();
            if (x_out->device() != device) {
                spdlog::error("Output tensor not on the same device as input");
                return false;
            }
            spdlog::debug("Device consistency test passed");

            spdlog::info("All RoPE module tests passed!");
            return true;

        } catch (const std::exception &e) {
            spdlog::error("Exception in testModuleRoPE: {}", e.what());
            return false;
        }
    });
}

// Test 8: Dtype assertion test
TestResult NNModuleTest::testDtypeAssertion() {
    return measureTime("DtypeAssertionTest", [this]() {
        try {
            spdlog::info("==========================================");
            spdlog::info("Testing dtype assertions when loading parameters");
            spdlog::info("==========================================");

            // Test 1: Successful load with matching dtype
            spdlog::info("Test 1: Successful load with matching dtype (F32)");
            MockLinearModule linear1(8, 4, infinicore::Device());
            auto matching_weight = infinicore::Tensor::ones({4, 8}, infinicore::DataType::F32, infinicore::Device());
            auto matching_bias = infinicore::Tensor::ones({4}, infinicore::DataType::F32, infinicore::Device());

            std::unordered_map<std::string, infinicore::Tensor> matching_state;
            matching_state.emplace("weight", matching_weight);
            matching_state.emplace("bias", matching_bias);

            // This should succeed without throwing
            linear1.load_state_dict(matching_state);
            spdlog::debug("✓ Matching dtype load succeeded");

            // Test 2: Failed load with mismatched dtype (load_parameter_)
            spdlog::info("Test 2: Failed load_parameter_ with mismatched dtype");
            MockLinearModule linear2(8, 4, infinicore::Device());
            auto mismatched_weight = infinicore::Tensor::ones({4, 8}, infinicore::DataType::BF16, infinicore::Device());

            bool exception_thrown = false;
            try {
                linear2.load_parameter_("weight", mismatched_weight);
            } catch (const std::runtime_error &e) {
                exception_thrown = true;
                std::string error_msg = e.what();
                if (error_msg.find("dtype mismatch") == std::string::npos) {
                    spdlog::error("Exception message doesn't contain 'dtype mismatch'");
                    return false;
                }
                spdlog::debug("✓ Mismatched dtype exception caught: {}", error_msg);
            }

            if (!exception_thrown) {
                spdlog::error("Expected exception for dtype mismatch in load_parameter_");
                return false;
            }

            // Test 3: Failed load with mismatched dtype (load_state_dict)
            spdlog::info("Test 3: Failed load_state_dict with mismatched dtype");
            infinicore::nn::Embedding embedding1(100, 64);
            auto mismatched_embed_weight = infinicore::Tensor::ones({100, 64}, infinicore::DataType::BF16, infinicore::Device());

            std::unordered_map<std::string, infinicore::Tensor> mismatched_state;
            mismatched_state.emplace("weight", mismatched_embed_weight);

            exception_thrown = false;
            try {
                embedding1.load_state_dict(mismatched_state);
            } catch (const std::runtime_error &e) {
                exception_thrown = true;
                std::string error_msg = e.what();
                if (error_msg.find("dtype mismatch") == std::string::npos) {
                    spdlog::error("Exception message doesn't contain 'dtype mismatch'");
                    return false;
                }
                if (error_msg.find("weight") == std::string::npos) {
                    spdlog::error("Exception message doesn't contain parameter name 'weight'");
                    return false;
                }
                spdlog::debug("✓ Mismatched dtype exception caught: {}", error_msg);
            }

            if (!exception_thrown) {
                spdlog::error("Expected exception for dtype mismatch in load_state_dict");
                return false;
            }

            // Test 4: Failed load with mismatched dtype (RMSNorm)
            spdlog::info("Test 4: Failed load_state_dict with mismatched dtype (RMSNorm)");
            infinicore::nn::RMSNorm norm1(768);
            auto mismatched_norm_weight = infinicore::Tensor::ones({768}, infinicore::DataType::BF16, infinicore::Device());

            std::unordered_map<std::string, infinicore::Tensor> mismatched_norm_state;
            mismatched_norm_state.emplace("weight", mismatched_norm_weight);

            exception_thrown = false;
            try {
                norm1.load_state_dict(mismatched_norm_state);
            } catch (const std::runtime_error &e) {
                exception_thrown = true;
                std::string error_msg = e.what();
                if (error_msg.find("dtype mismatch") == std::string::npos) {
                    spdlog::error("Exception message doesn't contain 'dtype mismatch'");
                    return false;
                }
                spdlog::debug("✓ Mismatched dtype exception caught for RMSNorm: {}", error_msg);
            }

            if (!exception_thrown) {
                spdlog::error("Expected exception for dtype mismatch in RMSNorm load_state_dict");
                return false;
            }

            // Test 5: Successful load with different module dtypes
            spdlog::info("Test 5: Successful load with BF16 dtype (module created with BF16)");
            MockLinearModule linear3(8, 4, infinicore::Device(), 0, 0, 1, infinicore::DataType::BF16);
            auto bf16_weight = infinicore::Tensor::ones({4, 8}, infinicore::DataType::BF16, infinicore::Device());
            auto bf16_bias = infinicore::Tensor::ones({4}, infinicore::DataType::BF16, infinicore::Device());

            std::unordered_map<std::string, infinicore::Tensor> bf16_state;
            bf16_state.emplace("weight", bf16_weight);
            bf16_state.emplace("bias", bf16_bias);

            // This should succeed
            linear3.load_state_dict(bf16_state);
            spdlog::debug("✓ BF16 dtype load succeeded");

            spdlog::info("All dtype assertion tests passed!");
            return true;

        } catch (const std::exception &e) {
            spdlog::error("Exception in testDtypeAssertion: {}", e.what());
            return false;
        }
    });
}

// Main test runner
TestResult NNModuleTest::run() {
    std::vector<TestResult> results;

    std::cout << "==============================================\n"
              << "InfiniCore nn::Module Test Suite\n"
              << "==============================================" << std::endl;

    results.push_back(testBasicModuleCreation());      // Merged: creation + parameters + state_dict + load
    results.push_back(testTensorParallelParameters()); // Tensor-parallel parameters
    results.push_back(testLoadStateDict());            // Advanced: hierarchical modules
    results.push_back(testModuleHierarchy());          // Demonstrates hierarchical construction
    results.push_back(testParameterLoading());         // Blob loading
    results.push_back(testModuleEmbedding());          // Embedding module test
    results.push_back(testModuleRMSNorm());            // RMSNorm module test
    results.push_back(testModuleRoPE());               // RoPE module test
    results.push_back(testDtypeAssertion());           // Dtype assertion test

    // Check if all tests passed
    bool all_passed = true;
    for (const auto &result : results) {
        if (!result.passed) {
            all_passed = false;
            break;
        }
    }

    return TestResult("NNModuleTest", all_passed,
                      all_passed ? "" : "Some nn::module tests failed");
}

} // namespace infinicore::test
