# -*- coding: utf-8 -*-
"""
Breeze TTS 2 参考音频预处理节点
- 人声/背景声分离
- 降噪
- 音量归一化
- 静音修剪
- 自动质量检测
"""

from __future__ import annotations

import io
import logging
import os
import tempfile

import numpy as np
import torch

logger = logging.getLogger("BreezeTTS2")

AUDIO_TYPE = "AUDIO"


def _comfy_audio_to_np(audio: dict) -> tuple[np.ndarray, int]:
    """ComfyUI audio dict → (float32 mono np, sample_rate)."""
    wav = audio["waveform"]
    if wav.dim() == 3:
        wav = wav[0]
    if wav.dim() == 2:
        wav = wav.mean(dim=0)
    return wav.numpy().astype(np.float32), int(audio["sample_rate"])


def _np_to_comfy(wav: np.ndarray, sr: int) -> dict:
    t = torch.from_numpy(wav.astype(np.float32))
    return {"waveform": t.view(1, 1, -1).contiguous(), "sample_rate": int(sr)}


class BreezeTTS2RefAudioCleaner:
    """参考音频清洗：人声分离 + 降噪 + 归一化 + 修剪。"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": (AUDIO_TYPE,),
                "separate_vocals": (
                    ["enable", "disable"],
                    {"default": "enable", "tooltip": "人声/背景分离（用MDX-Net/UVR模型，首次运行会自动下载模型）"},
                ),
                "denoise": (
                    ["enable", "disable"],
                    {"default": "enable", "tooltip": "频谱降噪（noisereduce）"},
                ),
                "normalize": (
                    ["enable", "disable"],
                    {"default": "enable", "tooltip": "音量归一化到 -3dB"},
                ),
                "trim_silence": (
                    ["enable", "disable"],
                    {"default": "enable", "tooltip": "修剪首尾静音"},
                ),
                "target_sr": (
                    ["24000", "16000", "44100", "48000"],
                    {"default": "24000", "tooltip": "输出采样率（Breeze用24000）"},
                ),
                "aggressiveness": (
                    ["light", "medium", "aggressive"],
                    {"default": "medium", "tooltip": "降噪强度：light=轻微降噪保留细节，medium=平衡，aggressive=强力降噪"},
                ),
            }
        }

    RETURN_TYPES = (AUDIO_TYPE, "STRING")
    RETURN_NAMES = ("clean_audio", "quality_report")
    FUNCTION = "clean"
    CATEGORY = "Breeze TTS 2"
    DESCRIPTION = "参考音频清洗：分离人声、降噪、归一化、修剪静音，输出适合克隆的干净音频。"

    def clean(self, audio, separate_vocals, denoise, normalize, trim_silence, target_sr, aggressiveness):
        wav, sr = _comfy_audio_to_np(audio)
        report_lines = []
        report_lines.append(f"输入: {len(wav)/sr:.2f}秒, {sr}Hz")

        # 1. 人声分离
        if separate_vocals == "enable":
            try:
                import ssl
                ssl._create_default_https_context = ssl._create_unverified_context
                from audio_separator.separator import Separator
                import soundfile as sf

                model_dir = os.path.join(os.path.dirname(__file__), "models", "audio_separator")
                os.makedirs(model_dir, exist_ok=True)
                sep = Separator(
                    model_file_dir=model_dir,
                    output_format="WAV",
                    output_dir=None,
                )
                sep.load_model("model_mel_band_roformer_ep_3005_sdr_11.4360.ckpt")

                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                    sf.write(f.name, wav, sr)
                    tmp_in = f.name
                tmp_out_dir = tempfile.mkdtemp()
                sep.output_dir = tmp_out_dir
                outputs = sep.separate(tmp_in)
                os.unlink(tmp_in)
                # 找输出的人声文件
                vocal_file = None
                for out in outputs if isinstance(outputs, (list, tuple)) else [outputs]:
                    if out and os.path.exists(out):
                        if "vocal" in os.path.basename(out).lower() or " vocals" in os.path.basename(out).lower() or "Vocals" in os.path.basename(out):
                            vocal_file = out
                            break
                if vocal_file is None:
                    # 取第一个存在的输出
                    for out in outputs if isinstance(outputs, (list, tuple)) else [outputs]:
                        if out and os.path.exists(out):
                            vocal_file = out
                            break
                if vocal_file:
                    wav, sr = sf.read(vocal_file)
                    if wav.ndim > 1:
                        wav = wav.mean(axis=1)
                    wav = wav.astype(np.float32)
                    report_lines.append(f"人声分离: 完成 ({os.path.basename(vocal_file)})")
            except Exception as e:
                report_lines.append(f"人声分离: 跳过 ({e})")

        # 2. 重采样到目标率
        target_sr_int = int(target_sr)
        if sr != target_sr_int:
            try:
                import librosa
                wav = librosa.resample(wav, orig_sr=sr, target_sr=target_sr_int)
                sr = target_sr_int
                report_lines.append(f"重采样: → {sr}Hz")
            except Exception:
                pass

        # 3. 降噪
        if denoise == "enable":
            try:
                import noisereduce as nr
                prop_decrease = {"light": 0.3, "medium": 0.6, "aggressive": 0.9}[aggressiveness]
                wav = nr.reduce_noise(y=wav, sr=sr, prop_decrease=prop_decrease, stationary=False)
                report_lines.append(f"降噪: 完成 (强度={aggressiveness})")
            except Exception as e:
                report_lines.append(f"降噪: 跳过 ({e})")

        # 4. 修剪首尾静音
        if trim_silence == "enable":
            try:
                import librosa
                wav, _ = librosa.effects.trim(wav, top_db=30)
                report_lines.append(f"修剪静音: 完成")
            except Exception:
                pass

        # 5. 归一化
        if normalize == "enable":
            peak = np.max(np.abs(wav))
            if peak > 0:
                target_peak = 10 ** (-3 / 20)  # -3dB
                wav = wav * (target_peak / peak)
            report_lines.append(f"归一化: → -3dB")

        # 质量报告
        duration = len(wav) / sr
        rms = np.sqrt(np.mean(wav ** 2))
        peak = np.max(np.abs(wav))
        report_lines.append(f"输出: {duration:.2f}秒, RMS={20*np.log10(rms+1e-10):.1f}dB, Peak={20*np.log10(peak+1e-10):.1f}dB")
        if duration < 2:
            report_lines.append("⚠️ 警告: 音频短于2秒，克隆效果可能不佳")
        if duration > 30:
            report_lines.append("⚠️ 警告: 音频超过30秒，建议截取5-15秒最干净的片段")
        if peak < 0.1:
            report_lines.append("⚠️ 警告: 音量过低")
        report = "\n".join(report_lines)

        return (_np_to_comfy(wav, sr), report)


NODE_CLASS_MAPPINGS = {
    "BreezeTTS2RefAudioCleaner": BreezeTTS2RefAudioCleaner,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BreezeTTS2RefAudioCleaner": "Breeze TTS 2 参考音频清洗",
}
