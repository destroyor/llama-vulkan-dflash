# DFlash 草稿模型接受率调优计划

## 目标
测试并调优与草稿模型接受率相关的参数，找到最佳配置。

---

## 基础配置

| 参数 | 值 |
|-----|----|
| 模型 | /home/des/models/Qwen3.6-27B-Q4_K_M.gguf |
| 草稿模型 | /home/des/models/dflash-draft-3.6-q8_0.gguf |
| 主模型 KV cache | turbo3_tcq |
| 上下文大小 | 131072 (128K) |
| Flash Attention | on |
| GPU layers | 99 |
| Speculative type | dflash |
| 监听地址 | 0.0.0.0 |
| 端口 | 8088 |
| API Key | 无 |

---

## 测试矩阵

### 第一阶段：--draft-max 调优
测试不同草稿token数量的接受率

| 测试 | --draft-max | --draft-p-min |
|-----|-------------|---------------|
| 1 | 8 | 0.75 |
| 2 | 10 | 0.75 |
| 3 | 12 | 0.75 |
| 4 | 14 | 0.75 |
| 5 | 16 | 0.75 |

### 第二阶段：--draft-p-min 调优
测试不同接受概率阈值

| 测试 | --draft-max | --draft-p-min |
|-----|-------------|---------------|
| 6 | 12 | 0.50 |
| 7 | 12 | 0.55 |
| 8 | 12 | 0.60 |
| 9 | 12 | 0.65 |
| 10 | 12 | 0.70 |
| 11 | 12 | 0.75 |

### 第三阶段：组合调优
在最佳 max 和 p-min 组合上微调

| 测试 | --draft-max | --draft-p-min |
|-----|-------------|---------------|
| 12 | (最佳-2) | (最佳-0.05) |
| 13 | (最佳) | (最佳) |
| 14 | (最佳+2) | (最佳+0.05) |

---

## 测试方法

1. 对每个配置启动服务器
2. 发送多个测试请求
3. 观察并记录 `draft acceptance rate`
4. 记录每秒 token 速度 (tps)

---

## 启动命令模板

```bash
HIP_VISIBLE_DEVICES=0 /home/des/项目/buun-llama-cpp/build/bin/llama-server \
  -m /home/des/models/Qwen3.6-27B-Q4_K_M.gguf \
  -md /home/des/models/dflash-draft-3.6-q8_0.gguf \
  --host 0.0.0.0 \
  --port 8088 \
  --no-mmap \
  -ctk turbo3_tcq \
  -ctv turbo3_tcq \
  -c 131072 \
  -fa on \
  -ngl 99 \
  --no-warmup \
  --reasoning off \
  --spec-type dflash \
  --draft-max X \
  --draft-p-min Y
```

---

## 评估指标

- **接受率** (acceptance rate): 越高越好
- **生成速度** (tps): 越高越好
- **用户体验**: 平衡接受率和速度

---

## 预期结果

最佳配置应该在接受率 ~0.3-0.4 之间，同时保持良好的生成速度。
