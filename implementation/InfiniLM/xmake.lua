if os.getenv("H3_OFFLINE_BUILD") == "1" then
    local pybind_include = os.getenv("H3_PYBIND11_INCLUDE")
    if not pybind_include or pybind_include == "" then
        raise("H3_OFFLINE_BUILD requires H3_PYBIND11_INCLUDE from the installed pybind11 wheel")
    end
    local python_include = os.getenv("H3_PYTHON_INCLUDE")
    if not python_include or python_include == "" then
        raise("H3_OFFLINE_BUILD requires H3_PYTHON_INCLUDE from Python sysconfig")
    end
    add_includedirs(pybind_include, python_include)
else
    add_requires("pybind11")
end

set_toolchains("gcc")

option("cxx11-abi")
    set_default(nil)
    set_showmenu(true)
    set_description("Set _GLIBCXX_USE_CXX11_ABI to match the installed InfiniCore")
    set_values("0", "1")
option_end()

local cxx11_abi = os.getenv("INFINILM_CXX11_ABI") or get_config("cxx11-abi")
if cxx11_abi and cxx11_abi ~= "" then
    if cxx11_abi ~= "0" and cxx11_abi ~= "1" then
        raise("INFINILM_CXX11_ABI must be 0 or 1")
    end
    add_defines("_GLIBCXX_USE_CXX11_ABI=" .. cxx11_abi)
end

-- Add spdlog from third_party directory
add_includedirs("third_party/spdlog/include")
add_includedirs("third_party/json/single_include/")

target("_infinilm")
    if os.getenv("H3_OFFLINE_BUILD") ~= "1" then add_packages("pybind11") end
    set_default(false)
    -- The offline driver supplies headers and EXT_SUFFIX from one interpreter.
    -- Xmake's independent python3 probe can select the system Python instead.
    local h3_python_suffix = os.getenv("H3_PYTHON_EXT_SUFFIX")
    add_rules("python.module", {soabi = not h3_python_suffix})
    if h3_python_suffix then
        on_config(function (target)
            target:set("extension", h3_python_suffix)
        end)
    end
    set_languages("cxx17")
    set_kind("shared")

    local INFINI_ROOT = os.getenv("INFINI_ROOT") or (os.getenv(is_host("windows") and "HOMEPATH" or "HOME") .. "/.infini")

    -- add_includedirs("csrc", { public = false })
    -- add_includedirs("csrc/pybind11", { public = false })
    add_includedirs(INFINI_ROOT.."/include", { public = true })
    -- spdlog is already included globally via add_includedirs at the top

    add_linkdirs(INFINI_ROOT.."/lib")
    add_links("infinicore_cpp_api", "infiniop", "infinirt", "infiniccl")

    -- Add C++ sources
    add_files("csrc/**.cpp")
    add_files("csrc/**.cc")

    set_installdir("python/infinilm")
target_end()
