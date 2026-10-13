# -*- coding: utf-8 -*-
"""
Breeze TTS 2 —— 基于 MOSS-Transcribe-Diarize 的转写 + 说话人分离节点

用 OpenMOSS 的端到端 0.9B 模型，一次性完成：
  · 长音频转写（50+ 语言）
  · 说话人分离（[S01]、[S02]… 标签 + 时间戳）
  · 声学事件感知
取代原先基于 wespeaker + VAD + 聚类 + audio_separator 的旧流水线。

节点：
  · BreezeTTS2MOSSDiarize      转写 + 说话人分离（输出 transcript / segments / script）
  · BreezeTTS2MOSSSpeakerExtract 按说话人从原始音频截取干净片段（输出 AUDIO / transcript）

依赖（需自行安装，本插件 install 助手会尝试安装轻量部分）：
  pip install moss-transcribe-diarize
  并要求 Transformers >= 5.13（本插件已声明）。
首次运行会自动从 HuggingFace 下载 OpenMOSS-Team/MOSS-Transcribe-Diarize 权重（约 2GB），
节点提供 hf_mirror 下拉，依次尝试『环境变量 HF_ENDPOINT → hf-mirror.com 国内镜像 → 官方 HuggingFace』，
总有一个能下；本地路径则直接加载。
"""

from __future__ import annotations

import logging
import os
import tempfile

import numpy as np
import torch

logger = logging.getLogger("BreezeTTS2")

CATEGORY = "Breeze TTS 2"

# 自定义数据类型：在节点间传递 MOSS 解析后的分段列表
MOSS_SEGMENTS_TYPE = "MOSS_SEGMENTS"
MOSS_REPO = "OpenMOSS-Team/MOSS-Transcribe-Diarize"

# 模型缓存（同一个 repo+device+dtype 只加载一次）
_MODEL_CACHE: dict = {}

# HuggingFace 镜像源（model 参数为仓库 ID 时依次尝试，总有一个能下）
# "" 表示官方默认 https://huggingface.co
HF_MIRROR_OFFICIAL = ""            # 官方 HuggingFace
HF_MIRROR_CN = "https://hf-mirror.com"  # 国内社区镜像


def _endpoints_to_try(hf_mirror: str) -> list:
    """根据下拉选项返回要依次尝试的 HF_ENDPOINT 列表（保持顺序、去重）。

    - "auto"           : 环境变量 HF_ENDPOINT → 国内镜像 → 官方
    - "hf-mirror.com"  : 强制使用国内镜像
    - "huggingface.co" : 强制使用官方源
    """
    if hf_mirror == "hf-mirror.com":
        return [HF_MIRROR_CN]
    if hf_mirror == "huggingface.co":
        return [HF_MIRROR_OFFICIAL]

    # auto：环境变量优先，再回退镜像 / 官方
    env_ep = os.environ.get("HF_ENDPOINT", "").strip()
    cands = []
    if env_ep:
        cands.append(env_ep)
    cands.append(HF_MIRROR_CN)
    cands.append(HF_MIRROR_OFFICIAL)
    seen, out = set(), []
    for c in cands:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


def _set_hf_endpoint(ep: str) -> None:
    """临时设置/清除 HF_ENDPOINT 环境变量（huggingface_hub 下载时读取它）。"""
    if ep:
        os.environ["HF_ENDPOINT"] = ep
    else:
        os.environ.pop("HF_ENDPOINT", None)


# ComfyUI 的 models 目录（custom_nodes/../models），用于本地模型自动回退
_NODE_DIR = os.path.dirname(os.path.abspath(__file__))
_COMFY_MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(_NODE_DIR)), "models")


def _resolve_model_repo(model_repo: str) -> str:
    """解析模型来源：

    - 已是本地文件/目录 → 直接返回；
    - 是 HuggingFace 仓库 ID（如 OpenMOSS-Team/MOSS-Transcribe-Diarize）→
      先在本机 ComfyUI/models 目录找同名文件夹（MOSS-Transcribe-Diarize），
      找到则直接用本地路径加载，避免重复下载 / 绕开网络 SSL 问题；
    - 否则原样返回（交给后续镜像下载逻辑处理）。
    """
    if os.path.isdir(model_repo) or os.path.isfile(model_repo):
        return model_repo
    leaf = model_repo.split("/")[-1].strip()
    if leaf:
        local_candidate = os.path.join(_COMFY_MODELS_DIR, leaf)
        if os.path.isdir(local_candidate):
            logger.info("[MOSS] 找到本地模型，改用: %s", local_candidate)
            return local_candidate
    return model_repo


def _default_model_repo() -> str:
    """节点默认 model 值：本机已有则指向本地目录，否则用 HF 仓库 ID。"""
    local = os.path.join(_COMFY_MODELS_DIR, "MOSS-Transcribe-Diarize")
    return local if os.path.isdir(local) else MOSS_REPO


# --------------------------------------------------------------------------- #
# ComfyUI AUDIO 互转
# --------------------------------------------------------------------------- #
def _comfy_audio_to_np(audio: dict) -> tuple[np.ndarray, int]:
    wav = audio["waveform"]
    if wav.dim() == 3:
        wav = wav[0]
    if wav.dim() == 2:
        wav = wav.mean(dim=0)
    return wav.numpy().astype(np.float32), int(audio["sample_rate"])


def _np_to_comfy(wav: np.ndarray, sr: int) -> dict:
    t = torch.from_numpy(wav.astype(np.float32))
    return {"waveform": t.view(1, 1, -1).contiguous(), "sample_rate": int(sr)}


# --------------------------------------------------------------------------- #
# 轻量音频清洗（MOSS 只做说话人分段，不做声源分离；这里提供可选后处理）
# --------------------------------------------------------------------------- #
def _resample(wav: np.ndarray, orig_sr: int, target_sr: int) -> tuple[np.ndarray, int]:
    if orig_sr == target_sr or target_sr <= 0:
        return wav, orig_sr
    try:
        import librosa
        wav = librosa.resample(wav.astype(np.float32), orig_sr=orig_sr, target_sr=target_sr)
    except Exception as exc:  # noqa: BLE001
        logger.warning("[MOSS] librosa.resample 失败，改用 scipy.signal.resample: %s", exc)
        from scipy import signal
        ratio = target_sr / orig_sr
        n_samples = int(len(wav) * ratio)
        wav = signal.resample(wav, n_samples)
    return wav.astype(np.float32), target_sr


def _highpass(wav: np.ndarray, sr: int, cutoff: float) -> np.ndarray:
    if cutoff <= 0:
        return wav
    from scipy import signal
    sos = signal.butter(4, cutoff, btype="high", fs=sr, output="sos")
    return signal.sosfiltfilt(sos, wav).astype(np.float32)


def _noise_reduce(wav: np.ndarray, sr: int, strength: float) -> np.ndarray:
    try:
        import noisereduce as nr
    except Exception as exc:  # noqa: BLE001
        logger.warning("[MOSS] noisereduce 未安装，跳过降噪: %s", exc)
        return wav
    # 静止噪声抑制；strength 控制 prop_decrease
    return nr.reduce_noise(
        y=wav.astype(np.float32),
        sr=sr,
        stationary=True,
        prop_decrease=max(0.0, min(1.0, float(strength))),
    ).astype(np.float32)


def _trim_silence(wav: np.ndarray, sr: int, top_db: int = 40) -> np.ndarray:
    try:
        import librosa
        wav, _ = librosa.effects.trim(wav.astype(np.float32), top_db=top_db)
        return wav.astype(np.float32)
    except Exception as exc:  # noqa: BLE001
        logger.warning("[MOSS] librosa trim 失败: %s", exc)
        return wav


def _peak_normalize(wav: np.ndarray, target_db: float = -3.0) -> np.ndarray:
    peak = np.max(np.abs(wav))
    if peak < 1e-9:
        return wav
    target_lin = 10.0 ** (target_db / 20.0)
    return (wav / peak * target_lin).astype(np.float32)


def _clean_audio(
    wav: np.ndarray,
    sr: int,
    target_sr: int = 24000,
    high_pass: float = 0.0,
    noise_reduce: bool = False,
    noise_reduce_strength: float = 0.5,
    trim_silence: bool = False,
    normalize: bool = True,
) -> tuple[np.ndarray, int]:
    """轻量后处理：重采样 → 高通 → 降噪 → 静音修剪 → 峰值归一化。"""
    # 1. 重采样（Breeze TTS 建议 24000 Hz）
    wav, sr = _resample(wav, sr, target_sr)
    # 2. 高通滤除低频噪声
    wav = _highpass(wav, sr, high_pass)
    # 3. 频谱降噪（无法去除背景音乐，只能抑制稳态噪声）
    if noise_reduce:
        wav = _noise_reduce(wav, sr, noise_reduce_strength)
    # 4. 修剪首尾静音
    if trim_silence:
        wav = _trim_silence(wav, sr)
    # 5. 峰值归一化（默认 -3 dBFS）
    if normalize:
        wav = _peak_normalize(wav, -3.0)
    return wav, sr


# --------------------------------------------------------------------------- #
# 模型加载
# --------------------------------------------------------------------------- #
def _resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    return device


def _torch_dtype(mode: str, device_str: str):
    if mode == "auto" or not mode:
        return torch.bfloat16 if device_str.startswith("cuda") else torch.float32
    table = {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    return table.get(mode.lower(), torch.bfloat16 if device_str.startswith("cuda") else torch.float32)


def _build_bundle(model_repo: str, torch_device, tdtype, from_kwargs: dict):
    """真正执行 from_pretrained（当前生效的 HF_ENDPOINT 决定走哪个源）。"""
    from transformers import AutoModelForCausalLM, AutoProcessor

    logger.info("[MOSS] 加载模型 %s (device=%s, dtype=%s) ...", model_repo, torch_device, tdtype)
    model = (
        AutoModelForCausalLM.from_pretrained(
            model_repo,
            trust_remote_code=True,
            attn_implementation="sdpa",
            **from_kwargs,
        )
        .to(dtype=tdtype)
        .to(torch_device)
        .eval()
    )
    processor = AutoProcessor.from_pretrained(model_repo, trust_remote_code=True)
    return (model, processor, torch_device, tdtype)


def load_moss_bundle(model_repo: str, device: str, dtype_mode: str, hf_mirror: str = "auto"):
    """加载（并缓存）MOSS-Transcribe-Diarize 模型与 processor。

    - model_repo 为本地路径（文件/目录存在）时直接加载，不走镜像。
    - 否则视为 HuggingFace 仓库 ID，依次尝试各镜像源，直到成功或耗尽。
    """
    device_str = _resolve_device(device)
    # 本地优先：仓库 ID 也先尝试本机 ComfyUI/models 同名目录
    model_repo = _resolve_model_repo(model_repo)
    key = (model_repo, device_str, dtype_mode, hf_mirror)
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]

    try:
        from transformers import AutoModelForCausalLM, AutoProcessor  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "MOSS 节点需要 Transformers 5.x。请先升级：pip install -U transformers（>=5.13）。"
        ) from exc
    try:
        import moss_transcribe_diarize  # noqa: F401  确保依赖已装
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "缺少依赖 moss-transcribe-diarize，请安装：pip install moss-transcribe-diarize"
        ) from exc

    torch_device = torch.device(device_str)
    tdtype = _torch_dtype(dtype_mode, device_str)
    from_kwargs = {"dtype": "auto"} if dtype_mode == "auto" else {"dtype": tdtype}

    # 本地路径：跳过所有镜像，直接加载
    if os.path.isdir(model_repo) or os.path.isfile(model_repo):
        logger.info("[MOSS] 使用本地模型路径: %s", model_repo)
        bundle = _build_bundle(model_repo, torch_device, tdtype, from_kwargs)
        _MODEL_CACHE[key] = bundle
        return bundle

    # HuggingFace 仓库 ID：依次尝试各镜像源
    last_exc = None
    for ep in _endpoints_to_try(hf_mirror):
        _set_hf_endpoint(ep)
        label = ep or "官方 HuggingFace"
        try:
            logger.info("[MOSS] 尝试从 %s 下载/加载 %s", label, model_repo)
            bundle = _build_bundle(model_repo, torch_device, tdtype, from_kwargs)
            _MODEL_CACHE[key] = bundle
            logger.info("[MOSS] 模型加载成功（源：%s）", label)
            return bundle
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            logger.warning("[MOSS] 镜像源 %s 加载失败：%s", label, exc)
            continue

    raise RuntimeError(
        "无法从任何源下载/加载模型 %s。\n"
        "已依次尝试：%s。\n"
        "若全部失败，可手动把权重放到本地目录，并把节点 model 参数填为本地路径；"
        "或设置环境变量 HF_ENDPOINT 指向可用镜像后重试。"
        % (model_repo, " / ".join(e or "官方" for e in _endpoints_to_try(hf_mirror)))
    ) from last_exc


# --------------------------------------------------------------------------- #
# 节点 1：转写 + 说话人分离
# --------------------------------------------------------------------------- #
class BreezeTTS2MOSSDiarize:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "model": (
                    "STRING",
                    {
                        "default": _default_model_repo(),
                        "tooltip": (
                            "MOSS-Transcribe-Diarize 模型来源。优先用本机 ComfyUI/models/MOSS-Transcribe-Diarize；"
                            "若填 HuggingFace 仓库 ID 且本机无同名目录，则按 hf_mirror 自动下载。"
                        ),
                    },
                ),
                "device": (
                    ["auto", "cuda:0", "cpu"],
                    {"default": "auto", "tooltip": "auto 在有 CUDA 时用 cuda:0，否则 cpu。"},
                ),
                "dtype": (
                    ["auto", "bf16", "fp16", "fp32"],
                    {"default": "auto", "tooltip": "计算精度。auto：CUDA 用 bf16、CPU 用 fp32。"},
                ),
                "max_new_tokens": (
                    "INT",
                    {
                        "default": 5120,
                        "min": 256,
                        "max": 65536,
                        "step": 256,
                        "tooltip": "最大生成 token 数。长音频（数十分钟）建议调大（如 65536）以免截断。",
                    },
                ),
                "download_if_missing": (
                    "BOOLEAN",
                    {"default": True, "tooltip": "本地缺失时从 HuggingFace 下载模型权重。"},
                ),
                "hf_mirror": (
                    ["auto", "hf-mirror.com", "huggingface.co"],
                    {
                        "default": "auto",
                        "tooltip": (
                            "模型下载镜像源。auto=依次尝试[环境变量→hf-mirror.com→官方]；"
                            "hf-mirror.com=强制国内镜像；huggingface.co=强制官方。"
                        ),
                    },
                ),
            }
        }

    RETURN_TYPES = ("STRING", MOSS_SEGMENTS_TYPE, "STRING")
    RETURN_NAMES = ("transcript", "segments", "script")
    FUNCTION = "diarize"
    CATEGORY = CATEGORY
    DESCRIPTION = (
        "用 MOSS-Transcribe-Diarize 0.9B 端到端完成「转写 + 说话人分离」。\n"
        "输出：\n"
        "  · transcript：带 [S01]/[S02] 与时间戳的原始转录文本\n"
        "  · segments：结构化分段列表（每段含 start/end/speaker/text），可接『说话人提取』节点\n"
        "  · script：'S01: 文本' 形式的剧本，可直接喂给『多说话人对话』节点做 TTS"
    )

    def diarize(self, audio, model, device, dtype, max_new_tokens, download_if_missing, hf_mirror):
        import soundfile as sf
        from moss_transcribe_diarize import parse_transcript
        from moss_transcribe_diarize.inference_utils import (
            build_transcription_messages,
            generate_transcription,
        )

        wav, sr = _comfy_audio_to_np(audio)
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                sf.write(f.name, wav, sr)
                tmp_path = f.name

            model_obj, processor, torch_device, tdtype = load_moss_bundle(
                model, device, dtype, hf_mirror
            )

            messages = build_transcription_messages(tmp_path)
            result = generate_transcription(
                model_obj,
                processor,
                messages,
                max_new_tokens=int(max_new_tokens),
                do_sample=False,
                device=torch_device,
                dtype=tdtype,
            )
            text = result.get("text", "").strip()

            segs = parse_transcript(text)
            segments = [
                {"start": float(s.start), "end": float(s.end), "speaker": s.speaker, "text": s.text}
                for s in segs
            ]

            # 剧本：以 [Sxx] 标签为说话人名字，每行一段，可直接接 BreezeTTS2MultiSpeaker
            lines = []
            for s in segs:
                t = (s.text or "").strip()
                if t:
                    lines.append(f"{s.speaker}: {t}")
            script = "\n".join(lines)

            logger.info(
                "[MOSS] 完成：%d 个分段，覆盖 %d 个说话人",
                len(segments),
                len({s["speaker"] for s in segments}),
            )
            return (text, segments, script)
        finally:
            if tmp_path:
                try:
                    os.unlink(tmp_path)
                except Exception:  # noqa: BLE001
                    pass


# --------------------------------------------------------------------------- #
# 节点 2：按说话人从原始音频截取干净片段
# --------------------------------------------------------------------------- #
class BreezeTTS2MOSSSpeakerExtract:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO", {"tooltip": "原始音频（与『转写分离』节点输入同一份）"}),
                "segments": (
                    MOSS_SEGMENTS_TYPE,
                    {"tooltip": "『转写分离』节点输出的 segments（含每段说话人与时间戳）。"},
                ),
                "speaker_index": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 99,
                        "step": 1,
                        "tooltip": "选择第几个说话人：0=S01、1=S02 ……，按出现顺序编号。",
                    },
                ),
                "pad_seconds": (
                    "FLOAT",
                    {
                        "default": 0.1,
                        "min": 0.0,
                        "max": 2.0,
                        "step": 0.05,
                        "tooltip": "每段前后留的静音 pad（秒），避免切在字词中间导致破音。",
                    },
                ),
                "target_sr": (
                    "INT",
                    {
                        "default": 24000,
                        "min": 8000,
                        "max": 48000,
                        "step": 1000,
                        "tooltip": "输出采样率。Breeze TTS 建议 24000；填 0 或 -1 保持原采样率。",
                    },
                ),
                "normalize": (
                    "BOOLEAN",
                    {"default": True, "tooltip": "峰值归一化到 -3 dBFS，让克隆参考音频音量适中。"},
                ),
                "trim_silence": (
                    "BOOLEAN",
                    {"default": False, "tooltip": "用 librosa 修剪首尾静音。"},
                ),
                "noise_reduce": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "频谱降噪（noisereduce）。可抑制空调/电流等稳态噪声，但无法去除背景音乐。",
                    },
                ),
                "noise_reduce_strength": (
                    "FLOAT",
                    {
                        "default": 0.5,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.05,
                        "tooltip": "降噪强度：0=不降噪，1=最大抑制。",
                    },
                ),
                "high_pass": (
                    "FLOAT",
                    {
                        "default": 80.0,
                        "min": 0.0,
                        "max": 400.0,
                        "step": 10.0,
                        "tooltip": "高通滤波截止频率(Hz)。80Hz 可去除低频轰鸣；0=关闭。",
                    },
                ),
            }
        }

    RETURN_TYPES = ("AUDIO", "STRING", MOSS_SEGMENTS_TYPE)
    RETURN_NAMES = ("audio", "transcript", "segments")
    FUNCTION = "extract"
    CATEGORY = CATEGORY
    DESCRIPTION = (
        "根据 MOSS 的分段时间戳，从原始音频中截取指定说话人的所有片段并拼接。\n"
        "MOSS 是『转写+说话人分段』模型，不是『人声/背景分离』模型；"
        "因此它按时间把目标说话人的片段切出来，背景音乐仍会保留。\n"
        "本节点额外提供轻量后处理（重采样、高通、降噪、静音修剪、归一化），"
        "用于优化克隆参考音频质量。如需彻底去除背景音乐，仍需额外的人声分离模型。"
    )

    def extract(
        self,
        audio,
        segments,
        speaker_index,
        pad_seconds,
        target_sr,
        normalize,
        trim_silence,
        noise_reduce,
        noise_reduce_strength,
        high_pass,
    ):
        wav, sr = _comfy_audio_to_np(audio)

        if not segments:
            empty = _np_to_comfy(np.zeros(int(0.1 * sr), np.float32), sr)
            return (empty, "无分段数据（请先连接『转写分离』节点的 segments）", [])

        # 按出现顺序列出唯一说话人
        order: list[str] = []
        for s in segments:
            sp = s.get("speaker")
            if sp and sp not in order:
                order.append(sp)
        if not order:
            empty = _np_to_comfy(np.zeros(int(0.1 * sr), np.float32), sr)
            return (empty, "分段中没有有效的说话人标签", [])

        idx = max(0, min(int(speaker_index), len(order) - 1))
        selected = order[idx]

        pad = float(pad_seconds)
        clips: list[np.ndarray] = []
        picked: list[dict] = []
        for s in segments:
            if s.get("speaker") != selected:
                continue
            a = max(0, int((s.get("start", 0.0) - pad) * sr))
            b = min(len(wav), int((s.get("end", 0.0) + pad) * sr))
            if b > a:
                clips.append(wav[a:b].copy())
                picked.append(s)

        if clips:
            out = np.concatenate(clips).astype(np.float32)
        else:
            out = np.zeros(int(0.1 * sr), np.float32)

        # 轻量后处理（MOSS 不做声源分离，这里至少做重采样/归一化/降噪）
        out_sr = int(target_sr) if int(target_sr) > 0 else sr
        out, out_sr = _clean_audio(
            out,
            sr,
            target_sr=out_sr,
            high_pass=float(high_pass),
            noise_reduce=bool(noise_reduce),
            noise_reduce_strength=float(noise_reduce_strength),
            trim_silence=bool(trim_silence),
            normalize=bool(normalize),
        )

        transcript = "\n".join(
            (p.get("text") or "").strip() for p in picked if (p.get("text") or "").strip()
        )
        logger.info(
            "[MOSS] 说话人 %s：截取 %d 段，拼接 %.2fs，输出采样率 %d Hz",
            selected,
            len(picked),
            len(out) / out_sr,
            out_sr,
        )
        return (_np_to_comfy(out, out_sr), transcript, picked)


NODE_CLASS_MAPPINGS = {
    "BreezeTTS2MOSSDiarize": BreezeTTS2MOSSDiarize,
    "BreezeTTS2MOSSSpeakerExtract": BreezeTTS2MOSSSpeakerExtract,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BreezeTTS2MOSSDiarize": "Breeze TTS 2 MOSS 转写分离",
    "BreezeTTS2MOSSSpeakerExtract": "Breeze TTS 2 MOSS 说话人提取",
}
