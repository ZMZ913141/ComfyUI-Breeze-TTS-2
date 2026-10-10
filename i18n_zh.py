# -*- coding: utf-8 -*-
"""
Breeze TTS 2 中文汉化补丁（外挂式，不修改原作者文件）。
在 __init__.py 末尾追加一行即可：
    try: from . import i18n_zh; i18n_zh.apply(NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS)
    except Exception: pass
"""

import logging

logger = logging.getLogger("BreezeTTS2")

_DISPLAY_NAMES_ZH = {
    "BreezeTTS2LoadModel": "Breeze TTS 2 加载模型",
    "BreezeTTS2VoiceClone": "Breeze TTS 2 声音克隆",
    "BreezeTTS2VoiceDesign": "Breeze TTS 2 音色设计",
    "BreezeTTS2VoiceDirection": "Breeze TTS 2 语音导演",
    "BreezeTTS2Speaker": "Breeze TTS 2 说话人配置",
    "BreezeTTS2MultiSpeaker": "Breeze TTS 2 多说话人对话",
    "BreezeWhisperTranscribe": "Breeze Whisper 语音转录",
    "BreezeTTS2RefAudioCleaner": "Breeze TTS 2 参考音频清洗",
    "BreezeTTS2MultiSpeakerExtractor": "Breeze TTS 2 多人说话人分离",
    "BreezeTTS2SpeakerSelector": "Breeze TTS 2 说话人选择",
}

# 英文 tooltip → 中文 tooltip
_TOOLTIP_MAP = {
    # 加载模型
    "int8 hybrid: backbone + text encoder INT8, depth decoder bf16 — bf16 speed at 5.5 GiB (recommended).\nbf16: no quantization — best quality, 7.5 GiB.\nint8: all transformer linears INT8 — smallest at 5.2 GiB, ~60% slower decode.\nint8 text encoder only: decode path stays bf16 — 6.8 GiB.":
        "int8 混合：主干+文本编码器 INT8，深度解码器 bf16 — bf16速度占5.5GB（推荐）。\nbf16：不量化 — 质量最佳，占7.5GB。\nint8：所有线性层 INT8 — 最小5.2GB，解码慢约60%。\n仅文本编码器 int8：解码路径保持bf16 — 占6.8GB。",
    "auto picks bf16 on supporting GPUs, fp32 otherwise.":
        "auto：支持的GPU用bf16，否则用fp32。",
    "auto uses ComfyUI's active torch device.":
        "auto：使用ComfyUI当前torch设备。",
    "auto uses flash_attention_2 when flash_attn is installed, else sdpa.":
        "auto：安装了flash_attn用flash_attention_2，否则用sdpa。",
    "cuda_graphs captures the depth decode loop into CUDA graphs: much faster, but the model weights stay fully resident in VRAM (AIMDO paging is bypassed for it) and the first generation spends a couple of seconds capturing the graphs.":
        "cuda_graphs：把深度解码循环捕获为CUDA图，快很多，但模型权重常驻显存（绕过AIMDO分页），首次生成需花几秒捕获图。",
    "Download the selected checkpoint from Hugging Face when missing locally.":
        "本地缺失时从Hugging Face下载所选模型。",

    # 生成参数
    "Maximum audio frames to generate (12.5 frames per second of speech; the model stops at EOS by itself).":
        "最大生成音频帧数（每秒语音约12.5帧；模型遇到EOS自动停止）。",
    "Backbone sampling temperature. 0.5-0.7 stable, 0.9+ more expressive but may cut words.":
        "主干采样温度。0.5-0.7稳定，0.9+更有表现力但可能吞字。",
    "Backbone top-k (0 disables).": "主干top-k（0=关闭）。",
    "Backbone top-p (1.0 disables).": "主干top-p（1.0=关闭）。",
    "HF-style repetition penalty on generated backbone tokens.": "主干token的重复惩罚（HF风格）。",
    "Depth decoder (codebook 1-15) sampling temperature.": "深度解码器（码本1-15）采样温度。",
    "Depth decoder top-k (0 disables).": "深度解码器top-k（0=关闭）。",
    "Depth decoder top-p (1.0 disables).": "深度解码器top-p（1.0=关闭）。",
    "Skip first N generated frames to avoid cut-off words at the start. 2-3 is typical for Chinese speech.":
        "跳过早N帧生成，避免开头吞字。中文语音建议2-3帧。",
    "0 uses the current random state. A positive value is repeatable.":
        "0=使用当前随机状态；填正数则结果可复现。",

    # 文本相关
    "Text to speak. Vocal events like (laugh) (sigh) (cough) (clears throat) work inline; use [笑] [叹气] etc. in Chinese.":
        "要合成的文本。支持插入语音事件如 (laugh)笑 (sigh)叹气 (cough)咳嗽 (clears throat)清嗓子；中文可用 [笑] [叹气] 等。",
    "Clean reference speech to clone timbre, rhythm, and style from.":
        "干净的参考人声，用于克隆音色、节奏和风格。",
    "The exact transcript of the reference audio.": "参考音频的准确文字稿。",
    "Guidance scale. 1.0 clones the reference as-is; raise it to push away from the reference.":
        "引导系数。1.0=原样克隆参考音色；调高可偏离参考音色。",
    "Natural-language description of the voice to create. Match the instruction language to the text language.":
        "用自然语言描述要创造的音色。描述语言要与文本语言一致。",
    "Guidance scale. 4 is recommended for instruction-following.":
        "引导系数。跟随指令场景建议4.0。",
    "Reference speech whose speaker identity is kept.": "保留说话人身份的参考音频。",
    "Direction for tone, emotion, pace, and delivery applied on top of the cloned voice.":
        "在克隆音色基础上叠加语气、情感、语速和表达方式的导演指令。",
    "Stitch the original reference clip into the output audio: 'none' returns the generated speech only, 'before' plays the reference clip first, 'after' appends it at the end. Purely an output edit; generation is unchanged.":
        "把原始参考片段拼接到输出音频：none=只返回生成语音，before=先放参考片段，after=末尾追加。仅输出编辑，不影响生成。",

    # 说话人配置节点
    "Speaker name the script uses ('Alice: hello'). Case-insensitive.":
        "剧本中使用的说话人名字（如'Alice: 你好'），不区分大小写。",
    "Reference clip from the ComfyUI input folder — click to browse or drag-drop a file onto the node. Select 'none' to design the voice from the instruction instead.":
        "从ComfyUI input文件夹选参考片段 — 点击浏览或拖文件到节点。选'none'则用指令设计音色。",
    "Exact transcript of the reference clip. Leave empty to auto-transcribe with Whisper (auto_transcribe_if_empty). Can be converted to an input to wire the Whisper Transcribe node.":
        "参考片段的准确文字稿。留空则用Whisper自动转录（开启auto_transcribe_if_empty）。可转为输入节点连线Whisper转录。",
    "Transcribe the reference clip with whisper-large-v3-turbo when reference_text is empty. Check the console log: a wrong transcript hurts cloning.":
        "当reference_text为空时用whisper-large-v3-turbo自动转录参考片段。请查看控制台日志：错误的文字稿会严重影响克隆效果。",
    "Voice description. Required when no reference audio is selected (designs the voice). With reference audio it steers tone, emotion, pace, and delivery (voice direction).":
        "音色描述。未选参考音频时必填（设计音色）；有参考音频时用于调整语气、情感、语速（语音导演）。",
    "Guidance for this speaker. 1.0 keeps a cloned voice as-is; around 4.0 is recommended for designed voices and strong directions.":
        "该说话人的引导系数。1.0=原样克隆；设计音色或强导演建议4.0左右。",
    "Optional wired audio (e.g. from a Load Audio node). Overrides the file dropdown when connected.":
        "可选：连线的音频（如从Load Audio节点接入）。连接后覆盖下拉选择。",

    # 多说话人
    "Dialogue script: one 'Name: line' per line (a line without a name continues the previous speaker; '[Name]:' also works), or paste a JSON list like [{\"speaker\": \"Alice\", \"text\": \"Hi!\"}] — handy for LLM-written dialogue. Inline vocal events like (laugh) (sigh) (cough) or [笑] work in both formats. The widget can be converted to an input to wire a string node.":
        "对话剧本：每行一条'名字: 台词'（无名字的行续接上一说话人；也支持'[名字]:'格式），或粘贴JSON列表如 [{\"speaker\": \"Alice\", \"text\": \"你好！\"}] — 适合LLM生成的对话。两种格式都支持 (laugh)笑 (sigh)叹气 或 [笑] 等语音事件。可转为输入节点连线字符串节点。",
    "Seconds of silence between dialogue turns.": "对话轮次之间的静音秒数。",
    "Cast slot {n}. Wire a Breeze TTS 2 Speaker node here.": "角色槽位{n}。在此连线Breeze TTS 2说话人配置节点。",

    # Whisper
    "Load a Whisper ASR pipeline for transcribing reference audio.":
        "加载Whisper语音识别管线，用于转录参考音频。",

    # 参考音频清洗
    "人声/背景分离（用MDX-Net/UVR模型，首次运行会自动下载模型）":
        "人声/背景分离（用MDX-Net/UVR模型，首次运行会自动下载模型）",
    "频谱降噪（noisereduce）": "频谱降噪（noisereduce）",
    "音量归一化到 -3dB": "音量归一化到 -3dB",
    "修剪首尾静音": "修剪首尾静音",
    "输出采样率（Breeze用24000）": "输出采样率（Breeze用24000）",
    "降噪强度：light=轻微降噪保留细节，medium=平衡，aggressive=强力降噪":
        "降噪强度：light=轻微降噪保留细节，medium=平衡，aggressive=强力降噪",
}

# 默认文本（英文 → 中文）
_DEFAULT_TEXT_MAP = {
    "(sigh) It is good to hear your voice again after all this time.":
        "(叹气) 好久不见，再听到你的声音真好。",
    "(sigh) Welcome aboard. Your journey begins now.":
        "(叹气) 欢迎登船，你的旅程现在开始。",
    "(clears throat) We need to discuss what happened last night.":
        "(清嗓子) 我们得谈谈昨晚发生的事。",
    "This is the exact transcript of the reference audio.":
        "这是参考音频的准确文字稿。",
    "A warm, thoughtful young woman with a clear voice and a calm, reflective delivery.":
        "一位温暖、有思想的年轻女性，声音清晰，语气温静而富有思考。",
    "Speak slowly with a restrained, serious tone.":
        "语速缓慢，语气克制而严肃。",
    "Alice": "小明",
}


def _patch_slot(opts: dict) -> None:
    if not isinstance(opts, dict):
        return
    if "tooltip" in opts and opts["tooltip"] in _TOOLTIP_MAP:
        opts["tooltip"] = _TOOLTIP_MAP[opts["tooltip"]]
    if "default" in opts and isinstance(opts["default"], str):
        d = opts["default"]
        if d in _DEFAULT_TEXT_MAP:
            opts["default"] = _DEFAULT_TEXT_MAP[d]
        else:
            # 处理多说话人默认文本
            if d.startswith("Alice:") and "Bob:" in d:
                opts["default"] = (
                    "小明: 嘿小红，演示录完了吗？\n"
                    "小红: (叹气) 快了，再来一条就行。\n"
                    "小明: (笑) 你一小时前也是这么说的！"
                )


def _patch_input_types(it: dict) -> None:
    for slot_group in ("required", "optional"):
        if slot_group not in it:
            continue
        for _key, slot_def in it[slot_group].items():
            if isinstance(slot_def, tuple) and len(slot_def) >= 2:
                slot_type = slot_def[0]
                opts = slot_def[1]
                if isinstance(opts, dict):
                    # 枚举下拉框（slot_type是list）不翻译default值
                    if isinstance(slot_type, list):
                        if "tooltip" in opts and opts["tooltip"] in _TOOLTIP_MAP:
                            opts["tooltip"] = _TOOLTIP_MAP[opts["tooltip"]]
                    else:
                        _patch_slot(opts)


def apply(node_class_mappings: dict, display_name_mappings: dict) -> None:
    # 显示名汉化
    for node_id, zh in _DISPLAY_NAMES_ZH.items():
        if node_id in display_name_mappings:
            display_name_mappings[node_id] = zh

    # monkey-patch INPUT_TYPES 的 tooltip
    for node_id, node_cls in node_class_mappings.items():
        if not hasattr(node_cls, "INPUT_TYPES"):
            continue
        orig = node_cls.INPUT_TYPES.__func__ if hasattr(node_cls.INPUT_TYPES, "__func__") else node_cls.INPUT_TYPES

        @classmethod
        def patched(cls, _orig=orig):
            result = _orig(cls)
            try:
                _patch_input_types(result)
            except Exception:
                pass
            return result

        try:
            node_cls.INPUT_TYPES = patched
        except Exception:
            pass

    logger.info("中文汉化补丁已加载（i18n_zh）")
