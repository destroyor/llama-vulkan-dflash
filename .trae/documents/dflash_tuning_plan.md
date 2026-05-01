# DFlash 调优测试计划

## 基础配置
- **模型**: Qwen3.6-27B-Q4_K_M.gguf
- **草稿模型**: dflash-draft-3.6-q8_0.gguf
- **GPU**: AMD RX 7900 XTX
- **上下文**: 65536 (64K)
- **端口**: 8088
- **API Key**: sk-local-123

---

## Phase 1: KV Cache 类型测试

| 测试 ID | KV Cache | 描述 | 压缩率 | 预期质量 |
| ------- | -------- | ---- | ------- | ------- |
| KV1 | `turbo4` | 4.25 bpv，近乎无损 | ~3.8x | 最佳 |
| KV2 | `turbo3_tcq` | 3.25 bpv，TCQ 3-bit | ~5x | 优秀 |
| KV3 | `turbo2_tcq` | 2.25 bpv，TCQ 2-bit | ~7x | 好 |
| KV4 | 非对称 | `-ctk turbo3_tcq -ctv turbo2_tcq` | ~6x | 优秀 |

---

## Phase 2: DFlash 参数调优

| 测试 ID | `--draft-max` | `--draft-p-min` | 说明 |
| ------- | ------------ | --------------- | ---- |
| DF1 | 16 | 0.75 | 默认值（基准） |
| DF2 | 12 | 0.6 | 平衡设置 |
| DF3 | 8 | 0.5 | 激进接受率 |
| DF4 | 20 | 0.75 | 更多草稿，保守 |

---

## Phase 3: 其他 Speculative 模式

| 测试 ID | `--spec-type` | 说明 |
| ------- | ------------- | ---- |
| SP1 | `dflash` | 默认 (当前) |
| SP2 | `copyspec` | CopySpec 复制模式 |
| SP3 | `ngram-map-k` | Ngram 模式 |
| SP4 | `suffix` | 后缀预测 |

---

## 统一的基础启动命令

```bash
HIP_VISIBLE_DEVICES=0 ./build/bin/llama-server \
  -m /home/des/models/Qwen3.6-27B-Q4_K_M.gguf \
  -md /home/des/models/dflash-draft-3.6-q8_0.gguf \
  --host 0.0.0.0 \
  --port 8088 \
  --api-key sk-local-123 \
  --no-mmap \
  -c 65536 \
  -fa on \
  -ngl 99 \
  --no-warmup \
  --reasoning off \
  --spec-type dflash \
  [KVCache 变量] \
  [DFlash 参数变量]
```

---

## 测试指标

1. **草稿接受率** (Draft acceptance rate)
2. **速度** (Tokens per second)
3. **显存占用** (VRAM usage)
4. **主观质量** (生成质量评估)

---

## 推荐的 KV Cache 层级（根据 README）

| 层级 | 配置 | 用例 |
| ---- | ---- | ---- |
| 🏆 最佳 | `-ctk turbo4 -ctv turbo4` | 日常使用，近乎无损 |
| 🥈 平衡 | `-ctk turbo3_tcq -ctv turbo3_tcq` | 更多上下文，质量很好 |
| 🥉 最大 | `-ctk turbo3_tcq -ctv turbo2_tcq` | 极长上下文 |
| 🎯 激进 | `-ctk turbo2_tcq -ctv turbo2_tcq` | 最大压缩 |
