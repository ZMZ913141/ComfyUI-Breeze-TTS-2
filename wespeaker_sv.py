# -*- coding: utf-8 -*-
"""
说话人声纹嵌入器（本地 ONNX，离线）。

使用 Wespeaker 官方导出的 voxceleb-resnet34-LM ONNX 模型，对 16kHz 单声道
人声片段提取 256 维 L2 归一化说话人向量。该模型在 VoxCeleb 上训练，区分度
高（同人余弦相似度 ~1.0，异人 ~0.25），且不依赖 HuggingFace Hub，可完全离线。

特征提取与 wespeaker 官方 runtime 对齐：
  - torchaudio kaldi 风格 fbank：80 mel bins, 25ms 窗, 10ms 帧移, dither=0.0
  - 逐片段 CMN（减去帧均值）
  - 输入乘 (1<<15) 还原 int16 量级
"""

from __future__ import annotations

import os

import numpy as np
import torch

MODEL_DIR = os.path.join(
    os.path.dirname(__file__), "models", "audio_encoders", "wespeaker-voxceleb-resnet34-LM"
)
ONNX_PATH = os.path.join(MODEL_DIR, "voxceleb_resnet34_LM.onnx")
SR = 16000
EMB_DIM = 256


class WespeakerEmbedder:
    """对一批 16k 单声道 float32 片段提取说话人向量。"""

    def __init__(self, path: str = ONNX_PATH, use_cuda: bool | None = None):
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"未找到 wespeaker ONNX 模型: {path}\n"
                "请先下载 Wespeaker/wespeaker-voxceleb-resnet34-LM 并解压到 "
                "models/audio_encoders/wespeaker-voxceleb-resnet34-LM/。"
            )
        import onnxruntime as ort

        providers: list[str]
        cuda_ok = False
        if use_cuda is None or use_cuda:
            try:
                cuda_ok = "CUDAExecutionProvider" in ort.get_available_providers()
            except Exception:
                cuda_ok = False
        if cuda_ok:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.out_name = self.session.get_outputs()[0].name
        self.providers = providers

    def _fbank(self, y: np.ndarray) -> np.ndarray:
        from torchaudio.compliance.kaldi import fbank

        w = torch.from_numpy(np.asarray(y, np.float32)).unsqueeze(0) * (1 << 15)
        f = fbank(
            w,
            num_mel_bins=80,
            frame_length=25,
            frame_shift=10,
            dither=0.0,
            energy_floor=0.0,
            sample_frequency=16000.0,
        )
        # 逐片段 CMN（与 wespeaker 官方 runtime 对齐）
        f = f - f.mean(0, keepdim=True)
        return f.numpy()[None, ...]  # [1, T, 80]

    @torch.no_grad()
    def embed(self, chunks: list[np.ndarray]) -> np.ndarray:
        """chunks: 一组 float32 16k 单声道片段 → [N, 256] L2 归一化向量。"""
        out: list[np.ndarray] = []
        for c in chunks:
            c = np.asarray(c, np.float32)
            if c.size == 0:
                out.append(np.zeros(EMB_DIM, np.float32))
                continue
            feat = self._fbank(c)
            vec = self.session.run([self.out_name], {self.input_name: feat})[0][0]
            out.append(np.asarray(vec, np.float32))
        e = np.stack(out).astype(np.float32)
        return e / (np.linalg.norm(e, axis=1, keepdims=True) + 1e-9)
