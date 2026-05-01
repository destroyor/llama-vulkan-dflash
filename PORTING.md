# Vulkan KV Cache 量化移植操作记录

## 概述

本文档记录了将 **TQ3_0**（3-bit Lloyd-Max codebook）和 **TURBO4_0**（4-bit PolarQuant + FWHT）两种 KV 缓存量化类型移植到 Vulkan 后端的完整操作过程，目标硬件为 AMD 7900 XTX (gfx1100) 和 780M (gfx1103)。

---

## 1. 修改文件清单

### 1.1 ggml 核心层（CPU 端）

| 文件 | 修改内容 | 说明 |
|------|---------|------|
| `ggml/include/ggml.h` | 添加 `GGML_TYPE_TQ3_0 = 46` 枚举 | TQ3_0 类型 ID |
| `ggml/src/ggml-common.h` | 添加 `block_tq3_0` 结构体 | 14 bytes/block: 2B d + 12B qs |
| `ggml/src/ggml-quants.c` | 添加 `quantize_row_tq3_0_ref` / `dequantize_row_tq3_0` | CPU 端量化/反量化 |
| `ggml/src/ggml.c` | 添加 TQ3_0 type_traits + quantize_chunk case | 核心类型注册 |
| `ggml/src/ggml-cpu/ggml-cpu.c` | 添加 TQ3_0 type_traits | vec_dot_type = Q8_0 |
| `ggml/src/ggml-cpu/quants.c/h` | 添加 `ggml_vec_dot_tq3_0_q8_0` | CPU vec_dot 实现 |
| `common/arg.cpp` | 添加 `GGML_TYPE_TQ3_0` 到 kv_cache_types | 命令行参数支持 |

**注意**: TURBO4_0 的 CPU 端代码已存在于 buun-llama-cpp 仓库中（`block_turbo4_0`、`quantize_row_turbo4_0_ref` 等），无需额外添加。

### 1.2 Vulkan 着色器

| 文件 | 修改内容 | 关键差异 |
|------|---------|---------|
| `vulkan-shaders/types.glsl` | 添加 `block_tq3_0` + `block_turbo4_0` 及 packed16 版本 | TQ3_0: QUANT_K=32; TURBO4_0: QUANT_K=128 |
| `vulkan-shaders/dequant_funcs.glsl` | 添加 TQ3_0 + TURBO4_0 的 `dequantize()` / `dequantize4()` | TQ3_0: 3-bit 位提取; TURBO4_0: 4-bit nibble 提取 |
| `vulkan-shaders/dequant_funcs_cm2.glsl` | 添加 TQ3_0 + TURBO4_0 的 CoopMat2 反量化函数 | 使用 packed16 + unpack8 格式 |
| `vulkan-shaders/flash_attn_base.glsl` | 添加 TQ3_0 + TURBO4_0 的 FA `dequantize4()` | BLOCK_BYTE_SIZE: TQ3_0=14, TURBO4_0=66 |
| `vulkan-shaders/copy_to_quant.comp` | 添加 TQ3_0 + TURBO4_0 的 GPU 端量化 | TURBO4_0 含完整 FWHT 旋转 |
| `vulkan-shaders/dequant_tq3_0.comp` | **新建** TQ3_0 独立反量化着色器 | 3-bit 索引提取 + 8 质心查表 |
| `vulkan-shaders/dequant_turbo4_0.comp` | **新建** TURBO4_0 独立反量化着色器 | 4-bit nibble + 16 质心查表 |
| `vulkan-shaders/vulkan-shaders-gen.cpp` | 注册 tq3_0 + turbo4_0 到各生成路径 | type_names, FA scalar/cm1, cpy, set_rows |

### 1.3 Vulkan 后端注册

| 文件 | 修改内容 |
|------|---------|
| `ggml-vulkan.cpp` | 添加 TQ3_0 + TURBO4_0 的所有 pipeline 注册和 switch case |

---

## 2. 着色器移植关键差异与适配点

### 2.1 TQ3_0（从 TurboQuant-Vulkan 移植）

**来源**: `/home/des/项目/llama/tq3_vulkan_src/` (tsuyu122/TurboQuant-Vulkan)

| 适配点 | 原始实现 | 适配后 |
|--------|---------|--------|
| 数据结构 | `block_tq3_0` (ggml_half d + uint8_t qs[12]) | 相同，添加 packed16 版本 |
| 质心表 | 8 个 float 常量 | 适配为 float / float16_t / FLOAT_TYPE 三种版本 |
| 3-bit 索引提取 | switch-case 方式 | dequant_funcs.glsl 用 switch-case; cm2/FA 用 bits24 位移方式 |
| Flash Attention | 使用 binding_idx 区分 K/V | 与 buun-llama-cpp 的 Q4_0/Q8_0 模式一致 |
| GPU 量化 | 二分搜索边界 | 相同 |
| DATA_A_QUANT_LEGACY | 使用 | 设置，因为只有单个 fp16 缩放因子 |
| QUANT_AUXF | = 1 | 单个辅助浮点数 |

**关键决策**: TQ3_0 **不支持** q8_1 integer dot product 路径。原因：
- TQ3_0 使用 codebook 查表方式，不是简单的 d-scale 格式
- `mul_mat_vecq_funcs.glsl` 缺少 TQ3_0 的 `repack`/`get_dm`/`mul_q8_1` 实现
- 不添加到 `is_legacy_quant` 函数

### 2.2 TURBO4_0（从 buun-llama-cpp CUDA 移植）

**来源**: buun-llama-cpp 仓库自身的 CUDA/HIP 实现

| 适配点 | CUDA 实现 | Vulkan 适配 |
|--------|----------|-------------|
| 块大小 | QK_TURBO4 = 128 | QUANT_K_TURBO4_0 = 128 |
| 4-bit 索引 | nibble 打包 (2/byte) | 相同：`(idx1 << 4) \| idx0` |
| 质心表 | 16 个 `__constant__` float | 16 个 float/float16_t/FLOAT_TYPE 常量 |
| FWHT 旋转 | `turbo_fwht_128_cuda` (7级蝶形) | `turbo4_fwht_128` (相同算法) |
| 符号数组 | `d_turbo_wht_signs1/2[128]` | `turbo4_wht_signs1/2[128]` 常量数组 |
| 范数校正 | `corrected * d_turbo4_alpha` | `corrected`（alpha=1.0，直接烘焙进 norm） |
| 量化搜索 | `turbo_find_nearest_4bit` (二叉树) | `turbo4_find_nearest` (相同逻辑) |
| writeonly 限制 | CUDA 无此限制 | 范数校正需用本地变量，不能从 data_q 回读 |

**关键决策**:
1. TURBO4_0 **不支持** q8_1 integer dot product 路径（同 TQ3_0）
2. CPY pipeline 的 workgroup 大小为 **{128, 1, 1}**（匹配 QUANT_K=128），而非 TQ3_0 的 {32, 1, 1}
3. Flash Attention 中 TURBO4_0 的 KQ 点积在**旋转空间**中进行，不需要逆旋转
4. V 反量化也**不需要逆旋转**（逆旋转在 FA 外部处理）

### 2.3 两种类型对比

| 特性 | TQ3_0 | TURBO4_0 |
|------|-------|----------|
| 块大小 | 32 | 128 |
| 每元素位数 | 3.5 bpv | 4.125 bpv |
| 每块字节数 | 14 | 66 |
| 索引编码 | 3-bit 紧凑 (24-bit/8值) | 4-bit nibble (2/byte) |
| 质心数 | 8 | 16 |
| 旋转 | 无 | FWHT + sign arrays |
| 范数 | 直接 d = vmax/max_centroid | 校正 norm = original/recon |
| FA KQ 点积 | 标准反量化 + float 点积 | codebook lookup + float 点积 |
| is_legacy_quant | 否 | 否 |
| CPU vec_dot | 有 (tq3_0_q8_0) | 无 (NULL) |

---

## 3. 编译命令

```bash
# 配置
cd /home/des/项目/llama/buun-llama-cpp
cmake -B build-vulkan -DGGML_VULKAN=ON -DCMAKE_BUILD_TYPE=Release

# 编译（首次或着色器变更后需要重新生成）
cd build-vulkan
cmake --build . --target llama-server -j$(nproc)

# 如果着色器生成有问题，强制重新生成：
rm -f ggml/src/ggml-vulkan/vulkan-shaders-gen-prefix/src/vulkan-shaders-gen-build/CMakeFiles/vulkan-shaders-gen.dir/vulkan-shaders-gen.cpp.o
rm -f ggml/src/ggml-vulkan/vulkan-shaders-gen-prefix/src/vulkan-shaders-gen-stamp/vulkan-shaders-gen-build
find ggml/src/ggml-vulkan -name "*.comp.cpp" -delete
rm -f ggml/src/ggml-vulkan/ggml-vulkan-shaders.hpp
cmake --build . --target llama-server -j$(nproc)
```

---

## 4. 测试命令

```bash
# TQ3_0 KV 缓存测试
./build-vulkan/bin/llama-server -m /path/to/model.gguf -ngl 99 -ctk tq3_0 -ctv tq3_0

# TURBO4 KV 缓存测试
./build-vulkan/bin/llama-server -m /path/to/model.gguf -ngl 99 -ctk turbo4 -ctv turbo4

# Q8_0 KV 缓存（基线对比）
./build-vulkan/bin/llama-server -m /path/to/model.gguf -ngl 99 -ctk q8_0 -ctv q8_0

# 指定 Vulkan 设备
GGML_VK_VISIBLE_DEVICES=0 ./build-vulkan/bin/llama-server ...  # 7900 XTX
GGML_VK_VISIBLE_DEVICES=1 ./build-vulkan/bin/llama-server ...  # 780M
```

---

## 5. 升级指南：随上游 llama.cpp 更新重新应用修改

当 buun-llama-cpp 上游更新时，需要重新应用以下修改。按文件分组说明：

### 5.1 ggml 核心层修改

这些修改在 TQ3_0 类型添加时已完成。如果上游 llama.cpp 添加了新的类型 ID，需要调整 `GGML_TYPE_TQ3_0` 的值。

**检查方法**: `grep GGML_TYPE_TQ3_0 ggml/include/ggml.h`

### 5.2 Vulkan 着色器修改

每个着色器文件的修改都是**追加式**的（在现有类型的 `#endif` 之后添加新类型的 `#if defined(DATA_A_XXX)` 块），因此冲突概率较低。

**升级步骤**:
1. 合并上游的着色器文件变更
2. 检查 TQ3_0/TURBO4_0 的 `#if defined` 块是否仍在正确位置
3. 如果上游添加了新类型，确保 TQ3_0/TURBO4_0 块在新类型之前
4. 检查 `get_dm` 条件链是否仍包含 `DATA_A_TQ3_0` 和 `DATA_A_TURBO4_0`

**关键文件检查清单**:
```bash
# 验证 TQ3_0 和 TURBO4_0 在各文件中存在
grep -c "DATA_A_TQ3_0" ggml/src/ggml-vulkan/vulkan-shaders/types.glsl  # 应 >= 1
grep -c "DATA_A_TURBO4_0" ggml/src/ggml-vulkan/vulkan-shaders/types.glsl  # 应 >= 1
grep -c "tq3_centroids" ggml/src/ggml-vulkan/vulkan-shaders/dequant_funcs.glsl  # 应 >= 1
grep -c "turbo4_centroids" ggml/src/ggml-vulkan/vulkan-shaders/dequant_funcs.glsl  # 应 >= 1
grep -c "tq3_0\|turbo4_0" ggml/src/ggml-vulkan/vulkan-shaders/vulkan-shaders-gen.cpp  # 应 >= 8
```

### 5.3 vulkan-shaders-gen.cpp 修改

需要检查的修改点：
1. `type_names` 向量包含 `"tq3_0"` 和 `"turbo4_0"`
2. FA scalar 路径条件包含 `tq3_0` 和 `turbo4_0`
3. FA CoopMat1 路径条件包含 `tq3_0` 和 `turbo4_0`
4. cpy 类型列表包含 `"tq3_0"` 和 `"turbo4_0"`
5. set_rows 类型列表包含 `"tq3_0"` 和 `"turbo4_0"`
6. `is_legacy_quant` **不**包含 `tq3_0` 或 `turbo4_0`

### 5.4 ggml-vulkan.cpp 修改

这是最容易产生冲突的文件，因为修改点非常多（~80 处）。

**升级策略**: 使用 `grep GGML_TYPE_TQ3_0\|GGML_TYPE_TURBO4_0` 找到所有修改点，然后逐一检查上游是否修改了周围的代码。

**关键修改模式**（每种类型重复以下模式）:
```cpp
// 1. CREATE_FA (4处/类型)
CREATE_FA(GGML_TYPE_TQ3_0, tq3_0, FA_SCALAR, )
CREATE_FA(GGML_TYPE_TQ3_0, tq3_0, FA_SCALAR, _fp32)
CREATE_FA(GGML_TYPE_TQ3_0, tq3_0, FA_COOPMAT1, _cm1)
CREATE_FA(GGML_TYPE_TQ3_0, tq3_0, FA_COOPMAT2, _cm2)

// 2. Pipeline 创建 (约 20 处/类型)
ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_TQ3_0], ...)
ggml_vk_create_pipeline(device, device->pipeline_get_rows[GGML_TYPE_TQ3_0], ...)
// ... 等等

// 3. Switch case (约 10 处/类型)
case GGML_TYPE_TQ3_0:
```

**自动化检查脚本**:
```bash
#!/bin/bash
# 检查 TQ3_0 和 TURBO4_0 在 ggml-vulkan.cpp 中的注册完整性
FILE="ggml/src/ggml-vulkan/ggml-vulkan.cpp"
echo "=== TQ3_0 引用数 ==="
grep -c "GGML_TYPE_TQ3_0" "$FILE"
echo "=== TURBO4_0 引用数 ==="
grep -c "GGML_TYPE_TURBO4_0" "$FILE"
echo "=== 检查 q8_1 路径（应为 0）==="
grep "GGML_TYPE_TQ3_0.*q8_1\|q8_1.*GGML_TYPE_TQ3_0" "$FILE" | wc -l
grep "GGML_TYPE_TURBO4_0.*q8_1\|q8_1.*GGML_TYPE_TURBO4_0" "$FILE" | wc -l
```

### 5.5 编译验证

升级后必须执行完整编译验证：
```bash
cd build-vulkan
# 强制重新生成着色器
rm -f ggml/src/ggml-vulkan/vulkan-shaders-gen-prefix/src/vulkan-shaders-gen-build/CMakeFiles/vulkan-shaders-gen.dir/vulkan-shaders-gen.cpp.o
find ggml/src/ggml-vulkan -name "*.comp.cpp" -delete
rm -f ggml/src/ggml-vulkan/ggml-vulkan-shaders.hpp
cmake --build . --target llama-server -j$(nproc)
```

如果出现链接错误（如 `undefined reference to xxx_tq3_0_xxx`），说明着色器未正确生成，需要检查 vulkan-shaders-gen.cpp 的注册。

如果出现 GLSL 编译错误（如 `cannot compile xxx_tq3_0`），说明着色器代码有问题，需要检查对应的 .comp/.glsl 文件。

---

## 6. 已知限制

1. **TQ3_0 无 CPU vec_dot 优化**: x86 架构仅使用 generic 实现，无 AVX/SSE 向量化
2. **TURBO4_0 无 CPU vec_dot**: `vec_dot = NULL`，CPU 路径不支持 turbo4 矩阵乘法
3. **TURBO4_0 FA 中无逆旋转**: V 反量化不做逆旋转，逆旋转在 FA 外部处理（与 CUDA 实现一致）
4. **不支持 q8_1 路径**: 两种类型都不支持 integer dot product 优化路径
5. **FWHT 量化性能**: copy_to_quant.comp 中的 FWHT 是单线程实现，GPU 量化可能较慢

---

## 7. 常量参考

### TQ3_0 质心（8个）
```
-2.1519454, -1.3439092, -0.7560052, -0.2450942,
 0.2450942,  0.7560052,  1.3439092,  2.1519454
```

### TQ3_0 量化边界（7个）
```
-1.7479273, -1.0499572, -0.5005497, 0.0, 0.5005497, 1.0499572, 1.7479273
```

### TURBO4_0 质心（16个）
```
-0.241556, -0.182907, -0.143047, -0.111065,
-0.083317, -0.058069, -0.034311, -0.011353,
 0.011353,  0.034311,  0.058069,  0.083317,
 0.111065,  0.143047,  0.182907,  0.241556
```

### TURBO4_0 中点（15个）
```
-0.212232, -0.162977, -0.127056, -0.097191, -0.070693,
-0.046190, -0.022832,  0.000000,  0.022832,  0.046190,
 0.070693,  0.097191,  0.127056,  0.162977,  0.212232
```

### TURBO4_0 FWHT 常量
- `inv_sqrt_128 = 0.08838834764831845` (= 1/√128)
- `turbo4_alpha = 1.0` (KLD 最优，默认值)
- `turbo4_wht_signs1[128]` 和 `turbo4_wht_signs2[128]` 见 copy_to_quant.comp

---

## 8. DFlash Vulkan 移植操作记录

### 8.1 概述

DFlash（DeltaNet Flash）是 buun-llama-cpp 中的推测解码机制，原仅支持 CUDA/HIP 后端。本次移植将 DFlash 的 4 个核心组件移植到 Vulkan 后端，使 AMD 7900 XTX + 780M 可通过 Vulkan 使用 DFlash 推测解码。

### 8.2 移植组件清单

| 组件 | 原始 CUDA 实现 | Vulkan 实现 | 说明 |
|------|---------------|------------|------|
| GATED_DELTA_NET_TREE | `gated_delta_net.cu` | `gated_delta_net_tree.comp` | 树形推测解码的 DeltaNet 状态更新 |
| Argmax 扩展 | `argmax.cu` | `argmax_ext.comp` | Gumbel-max 温度采样 + log-prob 输出 |
| Top-K 扩展 | `argmax.cu` | `topk_ext.comp` | GPU top-K 采样 (K≤32) |
| Cross-Ring | `cross-ring-interleave.cu` | `cross_ring_interleave.comp` + C++ 管理 | GPU 交叉注意力环形缓冲区 |
| get_proc_address | N/A | `ggml_backend_vk_get_proc_address` | Vulkan 后端函数导出 |

### 8.3 新建文件清单

| 文件 | 说明 |
|------|------|
| `vulkan-shaders/gated_delta_net_tree.comp` | 树形 GDN 着色器，9 个 binding (Q/K/V/G/Beta/State/Parent/Inter/Dst) |
| `vulkan-shaders/argmax_ext.comp` | 扩展 argmax，支持 inv_temp + Gumbel 噪声 + log-prob |
| `vulkan-shaders/topk_ext.comp` | GPU top-K，最小堆 + subgroup shuffle 合并 |
| `vulkan-shaders/cross_ring_interleave.comp` | Cross-ring 交错输出着色器 |

### 8.4 修改文件清单

| 文件 | 修改内容 |
|------|---------|
| `vulkan-shaders/vulkan-shaders-gen.cpp` | 注册 6 个新着色器变体 (tree×3 + argmax_ext + topk_ext + cross_ring) |
| `ggml-vulkan.cpp` | 添加 pipeline 声明/创建/dispatch/switch case/get_proc_address/cross-ring 管理 |
| `llama-context.cpp` | 修改 `init_cross_ring_gpu` 支持任意 GPU 后端（不再硬编码 CUDA） |

### 8.5 关键适配点

#### 8.5.1 GATED_DELTA_NET_TREE

- 基于 `gated_delta_net.comp` 扩展，增加 2 个 binding:
  - binding 6: `ParentBuf` (int[], 树结构父节点 ID)
  - binding 7: `InterBuf` (float16_t[], 中间状态持久化)
- 状态回溯逻辑:
  - `parent_t == -1` (GDN_TREE_ROOT_PARENT): 重新加载初始状态
  - `parent_t != t-1`: 从 `data_inter` 加载父节点中间状态 (F16→F32)
  - `parent_t == t-1`: 顺序执行，寄存器中状态正确
- 每个 token 处理完后将当前状态存入 `data_inter` (F32→F16)
- Pipeline 变体: 3 (S_V: 32/64/128) × 2 (KDA: 0/1) = 6 个

#### 8.5.2 Argmax 扩展

- 复用 `generic_head.glsl` 的 `p` push constants:
  - `p.KX` = vocab_size, `p.KY` = nrows
  - `p.param1` = inv_temp (1/temperature)
  - `p.param2` = seed_lo (float 形式存储 uint32)
  - `p.param3` = seed_hi (float 形式存储 uint32)
  - `p.param4` = K (float 形式存储 int32)
- Gumbel 噪声: Philox PRNG + -log(-log(u)) 变换
- 在线 softmax 计算 log-probability
- 输出: `data_d[row]` = argmax index, `data_d[KY + row]` = log_prob (float bits as int)

#### 8.5.3 Top-K 扩展

- 每线程维护 K 元素最小堆（寄存器，K≤32）
- Subgroup shuffle 合并: 5 轮 XOR (16/8/4/2/1)
- 多 subgroup 通过 shared memory 合并
- 插入排序降序输出
- Log-probability 输出格式同 argmax_ext

#### 8.5.4 Cross-Ring Vulkan 实现

**与 CUDA 版本的关键差异**:

| 方面 | CUDA | Vulkan |
|------|------|--------|
| 缓冲区管理 | `cudaMalloc` / `cudaFree` | `vk::Device::createBuffer` + `allocateMemory` |
| H2D 传输 | `cudaMemcpyAsync(H2D)` | 自有 host-visible staging buffer + `vkCmdCopyBuffer` |
| 交错输出 | compute kernel `k_cross_ring_interleave` | 多个 `vkCmdCopyBuffer` (逐 (token, layer) 对拷贝) |
| D2D 拷贝 | `cudaMemcpyAsync(D2D)` | `vkCmdCopyBuffer` |
| 同步 | `cudaStreamSynchronize` | `queue.waitIdle()` |
| 函数导出 | `ggml-cuda.cu` 直接导出 | `ggml_backend_vk_get_proc_address` |

**Vulkan 特有设计**:
- 使用 3 个 GPU 缓冲区:
  1. `ring_buffer`: device-local, 所有层的环形数据 (n_layers × ring_size × n_embd × 4B)
  2. `staging_buffer`: device-local, 交错输出缓冲区
  3. `host_staging_buffer`: host-visible, H2D 上传暂存区
- Interleave 使用 vkCmdCopyBuffer 而非 compute shader（简化实现，避免 descriptor set 管理）
- 通过 `ggml_backend_vk_get_proc_address` 导出与 CUDA 同名的 5 个函数

#### 8.5.5 llama-context.cpp 修改

原 `init_cross_ring_gpu` 硬编码查找 CUDA 后端:
```cpp
// 旧: ggml_backend_reg_t cuda_reg = nullptr;
// 新: ggml_backend_reg_t gpu_reg = nullptr;
```

修改后查找任意 GPU 后端（CUDA 或 Vulkan），通过 `ggml_backend_reg_get_proc_address` 获取函数指针。两个后端导出相同的函数名，因此调用代码无需修改。

### 8.6 编译验证

```bash
cd /home/des/项目/llama/buun-llama-cpp
cmake -B build-vulkan -DGGML_VULKAN=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build-vulkan --target llama-server -j$(nproc)
```

### 8.7 升级指南

DFlash Vulkan 移植的升级要点:

1. **着色器文件**: `gated_delta_net_tree.comp` 基于 `gated_delta_net.comp` 扩展，如果上游修改了 GDN 着色器，需要同步修改 tree 版本
2. **vulkan-shaders-gen.cpp**: 检查 6 个新 `string_to_spv` 调用是否仍在正确位置
3. **ggml-vulkan.cpp**: 检查以下新增项:
   - `pipeline_gated_delta_net_tree[3][2]` 声明
   - `pipeline_argmax_ext_f32` / `pipeline_topk_ext_f32` / `pipeline_cross_ring_interleave_f32` 声明
   - `vk_op_cross_ring_push_constants` 结构体
   - `ggml_vk_gated_delta_net_tree` / `ggml_vk_argmax_ext` / `ggml_vk_topk_ext` 函数
   - `GGML_OP_GATED_DELTA_NET_TREE` 的所有 switch case
   - `dflash_cross_ring_vk_*` 函数族
   - `ggml_backend_vk_get_proc_address` 函数
4. **llama-context.cpp**: 检查 `init_cross_ring_gpu` 是否仍使用 `gpu_reg` 而非 `cuda_reg`

### 8.8 已知限制

1. **Cross-ring interleave 使用 vkCmdCopyBuffer**: 每个 (token, layer) 对一次拷贝，当 cross_len × n_layers 很大时可能有性能问题
2. **Top-K 的 subgroup shuffle 假设 subgroup 大小为 32**: 如果设备 subgroup 大小不同，需要调整
3. **Cross-ring 函数使用第一个 Vulkan 设备**: 多 GPU 场景下可能需要设备选择
4. **DFlash tape GPU**: 使用 ggml 后端分配，已自动适配 Vulkan，无需额外修改
