# ComfyUI-Breeze-TTS-2

[English](README.md) | **简体中文**

> 适用于 ComfyUI 的 [Breeze TTS 2](https://huggingface.co/BreezeBlue/Breeze-TTS-2) 自定义节点：在本地 GPU 上完成**语音克隆、声音设计、声音风格指令**，支持中英双语、内联语气事件、INT8 ConvRot 量化，并完整集成 ComfyUI / AIMDO 动态显存管理。

---

## 一、功能特性

- **语音克隆（Voice Clone）**：给一段干净的参考音频 + 精确转写文本，即可克隆该说话人（CFG 1.0）。
- **声音设计（Voice Design）**：无需参考音频，用自然语言描述（年龄、音域、口音、性格）直接"造"一个声音（CFG 4）。
- **声音指令（Voice Direction）**：在克隆的基础上，用指令控制语气、情绪、语速、表达方式（CFG 4）；`stitch_reference` 可把原始参考片段拼到生成结果前/后。
- **中英双语混说**：同一句话里中英文混合；输出 24 kHz 单声道。
- **内联语气事件**：文本里直接写 `(laugh)`/`[笑]`、`(cough)`/`[咳嗽]`、`(sigh)`/`[叹气]` 等，自动合成对应语气。
- **INT8 ConvRot 量化**：主干 + 文本编码器 INT8，深度解码器保持 bf16，显存降低约 27%、速度几乎不变。
- **动态显存（AIMDO DynamicVRAM）**：对可转换权重分页，与其他模型共存时压低实时显存压力；`decode_mode=cuda_graphs` 时深度解码捕获为 CUDA graph，比 eager 快 3–4 倍。
- **MOSS 转写 + 说话人分离（本仓库增强）**：基于 [MOSS-Transcribe-Diarize 0.9B](https://github.com/OpenMOSS/MOSS-Transcribe-Diarize) 端到端完成长音频转写与说话人分离（[S01]/[S02] 标签 + 时间戳），并可直接截取指定说话人的干净片段用于克隆。

---

## 二、节点一览（分类 `Breeze TTS 2`）

| 节点 | 用途 |
| --- | --- |
| **Breeze TTS 2 Load Model** | 加载检查点，支持 `dtype`（auto/bf16/fp32）、`device`（auto/cuda/cpu）、`attention`（auto/eager/sdpa/flash_attention/sageattention）、`decode_mode`（eager/cuda_graphs）以及 `download_if_missing`。 |
| **Breeze TTS 2 Voice Clone** | 通过干净的参考音频 + 精确转写文本克隆说话人（CFG 1.0）。 |
| **Breeze TTS 2 Voice Design** | 通过自然语言描述创建声音，无需参考音频（CFG 4）。 |
| **Breeze TTS 2 Voice Direction** | 克隆参考声音，并用指令控制语气、情绪、语速和表达方式（CFG 4）。`stitch_reference` 可将原始参考片段拼接到生成语音之前或之后输出。 |
| **Breeze TTS 2 Whisper Transcribe** | Whisper 辅助节点，生成克隆所需的精确转写文本。 |
| **Breeze TTS 2 Speaker** | 多说话人节点中的一个具名角色。从 input 目录选择参考音频（下拉框支持浏览/上传，也可连线音频输入）进行克隆；或选择 `none` 并填写指令来设计声音。 |
| **Breeze TTS 2 Multi-Speaker** | 一次性生成整段对话脚本 —— 支持纯文本 `名字:` 行或直接粘贴 JSON —— 最多接入 8 个说话人，可自由混合克隆声音与设计声音。 |
| **Breeze TTS 2 MOSS 转写分离** | 基于 MOSS-Transcribe-Diarize 0.9B，端到端完成转写 + 说话人分离，输出转录文本、结构化分段、以及可直接喂给 Multi-Speaker 的剧本（见第六节）。 |
| **Breeze TTS 2 MOSS 说话人提取** | 按说话人从原始音频截取干净片段（依据 MOSS 时间戳），产出克隆参考音频与该说话人文字稿。 |

> 节点均自带中文界面（ComfyUI 原生 i18n 汉化补丁，原作者更新后只需重加该段补丁）。

---

## 三、安装（简单教程）

1. **放入 custom_nodes**
   ```bash
   cd ComfyUI/custom_nodes
   git clone https://github.com/ZMZ913141/ComfyUI-Breeze-TTS-2.git
   ```
   （或手动把仓库文件夹放到 `ComfyUI/custom_nodes/ComfyUI-Breeze-TTS-2`）

2. **安装 Python 依赖**（在你的 ComfyUI venv 里执行）
   ```bash
   # Windows 示例（用 ComfyUI 自带的 venv）
   E:\ComfyUI\.venv\Scripts\python.exe -m pip install -r ComfyUI/custom_nodes/ComfyUI-Breeze-TTS-2/requirements.txt
   ```
   > `install.py` 会在 ComfyUI 启动时自动运行，但它**不会**动 torch / torchaudio / transformers / numpy——这些请自行保证版本正确（见第五节）。

3. **确保 transformers 版本**（关键，见第五节）
   ```bash
   E:\ComfyUI\.venv\Scripts\python.exe -m pip install -U "transformers>=5.13.0"
   ```

4. **下载模型权重**（见第四节），放入对应路径。

5. **重启 ComfyUI**，在节点列表里应能看到 `Breeze TTS 2` 分类；如果没有，看控制台是否有节点注册报错。

---

## 四、模型下载与路径

### 4.1 Breeze-TTS-2 权重（必需）
来源镜像：[drbaph/Breeze-TTS-2-comfyui](https://huggingface.co/drbaph/Breeze-TTS-2-comfyui)

- **方式 A（推荐，自动）**：在 `Load Model` 节点勾选 `download_if_missing`，缺失文件会在加载时自动下载。
- **方式 B（手动）**：把镜像里的**全部文件**下载到：
  ```
  ComfyUI/models/breezetts2/drbaph_Breeze-TTS-2-comfyui/
  ```
  目录结构（自动下载与此一致）：
  ```
  📂 ComfyUI/models/breezetts2/drbaph_Breeze-TTS-2-comfyui/
  ├── Breeze-TTS-2-bf16.safetensors              （6.49 GiB，官方分片逐位一致合并，速度最快）
  ├── Breeze-TTS-2-int8-hybrid.safetensors       （4.53 GiB，推荐默认）
  ├── Breeze-TTS-2-int8-convrot.safetensors      （4.22 GiB，全部线性层量化，最小）
  ├── Breeze-TTS-2-int8-text-encoder.safetensors （5.84 GiB，解码路径 bf16）
  ├── config.json / generation_config.json
  ├── tokenizer.json / tokenizer_config.json / special_tokens_map.json
  └── 📂 audio_tokenizer/                        （Qwen3-TTS 12 Hz 编解码器，0.64 GiB，所有构建共用）
  ```
  > 只需下载你想用的那个 `.safetensors` 构建；`config.json`、分词器文件与 `audio_tokenizer/` 所有构建共用。

### 4.2 MOSS-Transcribe-Diarize 模型（转写分离节点用）
来源：[OpenMOSS-Team/MOSS-Transcribe-Diarize](https://huggingface.co/OpenMOSS-Team/MOSS-Transcribe-Diarize)

`Breeze TTS 2 MOSS 转写分离` 节点首次运行时会**自动从 HuggingFace 下载**该模型（约 2 GB，走 `transformers` + `trust_remote_code`，缓存在 HF 默认缓存目录）。你也可以提前手动下载到任意本地路径，再把节点 `model` 参数填成该路径。无需单独准备人声分离 / 聚类 / VAD 等模型——MOSS 在单一端到端模型里完成转写与说话人分离。

**下载镜像源**（节点新增 `hf_mirror` 下拉，默认 `auto`）：

- `auto`（默认）：依次尝试 **环境变量 `HF_ENDPOINT` → 国内镜像 [hf-mirror.com](https://hf-mirror.com) → 官方 HuggingFace**，总有一个能下；
- `hf-mirror.com`：强制走国内镜像（网络不通官方源时选这个）；
- `huggingface.co`：强制官方源。

> 也可直接设置环境变量 `HF_ENDPOINT=https://hf-mirror.com` 全局生效（优先级最高）。

```bash
# 可选：提前用镜像下载
pip install -U "huggingface_hub[cli]"
HF_ENDPOINT=https://hf-mirror.com hf download OpenMOSS-Team/MOSS-Transcribe-Diarize
```

---

## 五、关于 Transformers 5.x（技术突破与版本要求）

**项目硬性依赖：`transformers >= 5.13.0`（见 `pyproject.toml` 的 `[tool.breeze-tts-2]` 与 `requirements.txt`）。请勿降级。**

### 为什么锁 5.x——它带来了什么突破

- **主干用原生 transformers Qwen3 层**：解码器直接复用 transformers 的 `modeling_qwen3`，与上游版本演进保持同步，并能被 transformers 的 FA2 / SDPA 注意力路径加速。
- **文本编码器用 transformers 5.x 自带的"原生 T5Gemma2 编码器"**：这是 5.x 才内置的组件，取代了旧版需要随附的兼容实现（`vendor/t5gemma2_compat.py`，仅在 <5.x 时回退使用）。原生实现更稳定、可直接走 flash attention。
- **自定义注册的 FA2 路径**：上游 transformers 因 softcapping 顾虑禁用了 T5Gemma2 的 flash attention，但本检查点并不涉及该机制；本项目把 T5Gemma2 路由到 transformers 自己的 FA2 实现，从而启用 flash attention 加速。
- **随附的 Qwen3-TTS 12 Hz 编解码器已为 transformers 5.x 打补丁**：用 `merge_with_config_defaults` 取代在 5.x 被移除的 `check_model_inputs` 装饰器，避免 codec 在 5.x 下加载 / 解码崩溃。

### 要安装哪个版本？

```bash
# 在 ComfyUI 的 venv 里执行
python -m pip install -U "transformers>=5.13.0"
# 已验证可用：transformers 5.13.0
```

> 实测环境：`transformers 5.13.0` + `torch 2.13.0+cu130` + `numpy 2.5.3`。若你的 ComfyUI 自带更高版本的 transformers 5.x，向下兼容；但**不要低于 5.13.0**（低版本会因被移除的 API 导致 codec / 文本编码器报错）。

---

## 六、MOSS 转写 + 说话人分离（本仓库增强）

基于 [MOSS-Transcribe-Diarize 0.9B](https://github.com/OpenMOSS/MOSS-Transcribe-Diarize) 的端到端模型，一次性完成**长音频转写 + 说话人分离 + 时间戳**，取代原先「参考音频清洗 / 多人说话人分离 / 说话人选择」三件套（旧方案依赖 wespeaker 声纹 + VAD + 聚类 + audio_separator，已弃用）。

**节点链路：**
```
Load Audio → Breeze TTS 2 MOSS 转写分离 → ( transcript / segments / script )
                                    ├─ segments → Breeze TTS 2 MOSS 说话人提取 → AUDIO（某说话人干净片段）
                                    └─ script   → Breeze TTS 2 Multi-Speaker（直接生成多人对话 TTS）
```

- **`Breeze TTS 2 MOSS 转写分离`**：输入音频，输出三样东西
  - `transcript`：带 `[S01]`/`[S02]` 与时间戳的原始转录文本；
  - `segments`：结构化分段列表（每段含 `start`/`end`/`speaker`/`text`），可接「说话人提取」节点；
  - `script`：`S01: 文本` 形式的剧本，可直接喂给「多说话人对话」节点做 TTS。
- **`Breeze TTS 2 MOSS 说话人提取`**：按说话人下标（`speaker_index`，0=S01 …）从**原始音频**按时间戳截取该说话人的所有片段并拼接，产出干净的克隆参考音频 + 该说话人文字稿。

**为什么用 MOSS 而不是旧方案：** MOSS 把 ASR 与 diarization 联合建模，说话人标签与文本、时间戳天然对齐，错误更少、对长音频与重叠语音更鲁棒，且不再需要单独下载人声分离 / 聚类 / VAD 等模型。

**常用参数：** `model`（HF 仓库或本地路径）、`device`（auto/cuda:0/cpu）、`dtype`（auto/bf16/fp32）、`max_new_tokens`（长音频调大以免截断，如 65536）。

> 首次运行会自动下载模型（约 2 GB，见 4.2 节）。若缺少依赖，节点会给出明确报错：`pip install moss-transcribe-diarize`，并确保 `transformers >= 5.13`。

---

## 七、多说话人对话（Multi-Speaker）

将最多 8 个 **Speaker** 节点接入 **Multi-Speaker** 节点的 `speaker_1…8` 输入，然后编写（或粘贴）脚本。说话人名字匹配对大小写、空格和标点宽容（`Ali G` 可匹配名为 `alig` 的说话人），且所有名字都会在生成开始前校验 —— 脚本写错会立即报错，不会白白消耗显卡时间。

**方式一 —— 纯文本。** 每行一个 `名字: 台词`。没有名字的行会并入上一位说话人；`[名字]:` 和 markdown 的 `**名字:**` 写法同样支持：

```
Ada: (sigh) 这次又是谁把我的船撞上码头了？
Bob: 严格来说，船长，是码头撞上了我们。
Ada: 码头可不是这样运作的，Bob。
Bob: 恕我直言，潮汐表上写得清清楚楚。
```

**方式二 —— 粘贴 JSON。** 适合 LLM 生成的脚本。只要文本以 `[` 或 `{` 开头，就会按 JSON 解析：

```json
[
  {"speaker": "Ada", "text": "(sigh) 这次又是谁把我的船撞上码头了？"},
  {"speaker": "Bob", "text": "严格来说，船长，是码头撞上了我们。"},
  {"speaker": "Ada", "text": "码头可不是这样运作的，Bob。"}
]
```

同时还接受：单个 `{"speaker": ..., "text": ...}` 对象、`{"script": [...]}` 这类包装，以及键名别名 `speaker`/`name`/`character`/`role` 和 `text`/`line`/`content`/`message`。内联语气事件在两种格式中完全相同。`text` 控件可转换为输入（右键菜单），从而接入任意文本生成节点。

推理开始前，控制台会先打印摘要 ——

```
[BreezeTTS2] Multi-speaker script: 4 speaker(s), 16 turn(s), ~75s of speech total | cast: Ada (clone), Bob (clone), Carol (design), Dave (design)
```

—— 然后每个回合显示一条进度条（`[3/16] Bob (~6s)`）。参考音频每位说话人只编码一次；每位说话人在所有回合中使用固定的种子偏移（设计的声音因此保持一致）；`pause_between_speakers` 控制回合间的停顿。参考音频建议不超过约 20 秒（硬性上限 60 秒）。想让 LLM 按此格式编写脚本，把 [SKILL.md](SKILL.md) 交给它即可。

---

## 八、检查点

启用 `download_if_missing` 时，加载器会下载到 `ComfyUI/models/breezetts2/`：

| 选项 | 文件 | 说明 |
| --- | --- | --- |
| int8 hybrid **（默认）** | `Breeze-TTS-2-int8-hybrid.safetensors` | 主干 + 文本编码器 INT8 ConvRot，深度解码器保持 bf16。速度持平 bf16，显存降低 27%。 |
| bf16 combined | `Breeze-TTS-2-bf16.safetensors` | 官方分片权重合并为单文件（逐位一致）。速度最快。 |
| INT8 all linears | `Breeze-TTS-2-int8-convrot.safetensors` | 最小构建；解码循环更慢（见下文）。 |
| INT8 text encoder only | `Breeze-TTS-2-int8-text-encoder.safetensors` | 解码路径完全 bf16；在不影响质量的前提下最小幅度削减显存。 |

所有镜像均位于 [drbaph/Breeze-TTS-2-comfyui](https://huggingface.co/drbaph/Breeze-TTS-2-comfyui)。

权重按来源仓库存放在 `ComfyUI/models/breezetts2/` 下（优先主目录；extra_model_paths.yaml 条目与符号链接同样会被搜索）。开启 `download_if_missing` 时缺失文件会自动下载，否则报错信息会指出预期目录。

预计显存占用约 **5.3–7.5 GiB**（取决于构建与 dtype）；AIMDO DynamicVRAM 会对可转换的权重进行分页，以便在与其他模型并存时保持较低的实时显存压力。

---

## 九、基准测试（RTX 5090，ComfyUI 加载器，sdpa）

| 构建 | 平均 RTF | 峰值显存 | Whisper WER（对比提示文本） |
| --- | --- | --- | --- |
| bf16 | 5.68 | 7.51 GiB | 基准 |
| int8-convrot（全部线性层） | 9.08 | 5.21 GiB | 与 bf16 一致 |
| **int8-hybrid（深度解码器 bf16）** | **5.64** | **5.53 GiB** | 与 bf16 一致 |
| int8-text-encoder only | 5.34 | 6.82 GiB | 与 bf16 一致 |

全量 INT8 变慢的原因：深度解码器在每个 12.5 Hz 帧上执行 15 次窄小的单 token GEMM（≈每帧 1260 次线性层调用）。当 M=1–2 时，int8 的每次调用开销（约 0.1 ms 的量化 + 调度 + 内存分配）远超 GEMM 节省的时间，因此这些层反而输给纯 bf16 `F.linear`（约 0.03 ms）。主干每帧仅贡献约 196 次调用，文本编码器只在 prefill 运行一次，量化它们几乎无成本——这正是 hybrid 构建的做法。

---

## 十、模型架构（实际运行的部分）

- **文本编码器**：T5Gemma2（26 层，隐藏维度 1152）+ 线性投影到主干空间（transformers 5.x 原生实现）。
- **主干**：Qwen3 风格解码器（28 层，隐藏维度 2048），每个 12.5 Hz 帧预测第一个 codebook token，音频 token 嵌入对 16 个 codebook 求和。
- **深度解码器**：12 层解码器，每帧生成 codebook 1–15，带独立 CFG。
- **音频分词器**：随附的 Qwen3-TTS 12 Hz 编解码器（Mimi 风格编码器 + 流式解码器），24 kHz，16 个 codebook。参考音频使用同一编解码器编码。
- 检查点内的 `codec_model`（仅供上游训练前向使用的 Mimi 副本）在推理时不构建。

eager 生成循环重新实现了官方 `FastBreezeStreamingRuntime`，并关闭所有快速阶段（用 DynamicCache 替代 CUDA graph），以保持与 AIMDO 动态显存分页及 INT8 权重转换的兼容性。使用 `decode_mode=cuda_graphs` 时，深度解码步骤会被捕获为 CUDA graph——捕获后的解码比走 AIMDO 分页的 eager 路径**快 7–9 倍**（整段生成约快 3–4 倍）；该模式下模型权重常驻显存，不再由 AIMDO 分页。

---

## 十一、许可证

节点包代码：Apache-2.0（见 `LICENSE` 与 `NOTICE`）。随附的模型代码保留其上游许可证（breezeblue-ai/breeze-tts 与 Qwen/Qwen3-TTS 均为 Apache-2.0）。

**模型权重、衍生检查点与自托管输出受 BreezeBlue Research and Non-Commercial License 约束** —— 见 [MODEL_LICENSE](https://huggingface.co/BreezeBlue/Breeze-TTS-2/blob/main/LICENSE)。
商业使用需获得 RESONIA, INC. 的书面授权。
