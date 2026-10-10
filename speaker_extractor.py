# -*- coding: utf-8 -*-
"""
Breeze TTS 2 多人说话人分离 / 背景音保留节点

流水线：
  Load Audio
     │
     ├─(可选) 人声/背景分离  audio_separator(mel_band_roformer)
     │        → vocals(人声)  +  instrumental(背景/伴奏)
     ▼
  对 vocals 做 VAD → 滑窗 → wespeaker 声纹向量 → 凝聚聚类 → 说话人分段
     │
     ▼
  按说话人切片并导出（两种模式）：
     · 原样导出：每个说话人一个 WAV，保留原始时间轴（他人说话/停顿处为静音）
     · 自动切片：每个说话人一个子文件夹，每段 utterance 单独一个 WAV
  背景音作为独立通道可单独导出（"仅背景音"）。

产出 SPEAKERS 对象（内存），下游可用「说话人选择」节点挑出某一说话人接克隆流程；
同时把文件写到磁盘（ComfyUI output/BreezeTTS2_Speakers/...）。
"""

from __future__ import annotations

import collections
import logging
import os
import shutil
import tempfile
import time

import numpy as np
import torch

logger = logging.getLogger("BreezeTTS2")

AUDIO_TYPE = "AUDIO"
SPEAKERS_TYPE = "SPEAKERS"

# 与参考音频清洗节点保持一致的分离模型
_SEP_MODEL_NAME = "model_mel_band_roformer_ep_3005_sdr_11.4360.ckpt"
_DIAR_SR = 16000  # 说话人分离固定用 16k
_WIN = 1.5        # 滑窗长度（秒）
_HOP = 0.5        # 滑窗步长（秒）
_MIN_WIN = 0.4    # 最短窗口（秒）

# 重叠感知裁剪（消除串音）用的细窗参数
_FINE_WIN = 1.0    # 重叠裁剪细窗长度（秒）
_FINE_HOP = 0.5    # 重叠裁剪细窗步长（秒）
_FINE_MIN = 0.3    # 重叠裁剪最短细窗（秒）


# --------------------------------------------------------------------------- #
# MossFormer2 两路物理分离（从根上解决两人同时说话的串音）
# --------------------------------------------------------------------------- #
_MOSSFORMER_CKPT_CANDIDATES = [
    os.environ.get("BREEZE_MOSSFORMER_CKPT", ""),
    r"E:\ComfyUI\models\checkpoints\MossFormer2_SS_16K",
    os.path.join(os.path.dirname(os.path.abspath(__file__)),
                 "checkpoints", "MossFormer2_SS_16K"),
]
_mossformer_cv = None      # 懒加载缓存
_mossformer_err = ""       # 最近一次加载失败信息


def _find_mossformer_ckpt():
    for p in _MOSSFORMER_CKPT_CANDIDATES:
        if p and os.path.isfile(os.path.join(p, "last_best_checkpoint.pt")):
            return p
    return None


def _load_mossformer():
    """懒加载 MossFormer2 分离模型；返回 (ClearVoice实例, 错误信息)。失败信息非空即不可用。"""
    global _mossformer_cv, _mossformer_err
    if _mossformer_cv is not None:
        return _mossformer_cv, ""
    if _mossformer_err:
        return None, _mossformer_err
    ckpt = _find_mossformer_ckpt()
    if not ckpt:
        _mossformer_err = "未找到 MossFormer2 权重(last_best_checkpoint.pt)，置于 checkpoints/MossFormer2_SS_16K"
        return None, _mossformer_err
    try:
        import clearvoice
        from clearvoice.network_wrapper import network_wrapper as _NW
        _orig = _NW.load_args_ss

        def _patched(self):
            _orig(self)
            # 构造时即加载权重，必须在此处注入本地绝对路径（库默认是相对路径，会去联网下载）
            self.args.checkpoint_dir = ckpt

        _NW.load_args_ss = _patched
        cv = clearvoice.ClearVoice(
            task='speech_separation', model_names=['MossFormer2_SS_16K'])
        _NW.load_args_ss = _orig
        _mossformer_cv = cv
        return cv, ""
    except Exception as e:  # 任何异常都回退到声纹聚类，不阻断主流程
        _mossformer_err = f"MossFormer2 加载失败: {e}"
        return None, _mossformer_err


def _mossformer_separate(y16, cv):
    """对 16k 单声道 vocal 做两路物理分离，返回 [(T,), (T,), ...]（每路 16k）。"""
    if y16.size == 0:
        return [np.zeros(1, np.float32), np.zeros(1, np.float32)]
    inp = np.asarray(y16, dtype=np.float32).reshape(1, -1)  # (B, T)
    sep = cv(inp)  # t2t 模式 -> decode_data -> (num_spks, B, T)
    n = int(cv.models[0].args.num_spks)
    return [np.asarray(sep[k, 0], dtype=np.float32) for k in range(n)]


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


def _resample(y: np.ndarray, sr: int, target: int) -> np.ndarray:
    if sr == target:
        return y
    import librosa

    return librosa.resample(y, orig_sr=sr, target_sr=target)


def _to_mono(y: np.ndarray) -> np.ndarray:
    if y.ndim > 1:
        y = y.mean(axis=1)
    return y.astype(np.float32)


def _is_silent(y: np.ndarray, peak_eps: float = 1e-3) -> bool:
    """判断一路音频是否近乎静音（如分离模型判定'无独立人声'时输出全零/极弱轨道）。"""
    if y is None or y.size == 0:
        return True
    return float(np.max(np.abs(y))) < peak_eps


# --------------------------------------------------------------------------- #
# VAD（silero，离线 ONNX）
# --------------------------------------------------------------------------- #
_VAD = None


def _load_vad():
    global _VAD
    if _VAD is None:
        from silero_vad import load_silero_vad

        _VAD = load_silero_vad()
    return _VAD


def vad_regions(y: np.ndarray, sr: int = _DIAR_SR) -> list[tuple[float, float]]:
    """返回人声区间列表 [(start_s, end_s), ...]。"""
    from silero_vad import get_speech_timestamps

    model = _load_vad()
    ts = get_speech_timestamps(
        torch.from_numpy(y).float(),
        model,
        sampling_rate=sr,
        threshold=0.5,
        min_speech_duration_ms=400,
        min_silence_duration_ms=400,
        speech_pad_ms=30,
        return_seconds=False,
    )
    return [(s["start"] / sr, s["end"] / sr) for s in ts]


# --------------------------------------------------------------------------- #
# 滑窗 + 聚类 + 后处理
# --------------------------------------------------------------------------- #
def windows_from_regions(regions, win=_WIN, hop=_HOP, min_win=_MIN_WIN):
    wins = []
    for (a, b) in regions:
        if b - a < min_win:
            continue
        if b - a <= win:
            wins.append((a, b))
            continue
        t = a
        while t < b - 1e-6:
            e = min(t + win, b)
            if e - t >= min_win:
                wins.append((t, e))
            t += hop
    return wins


def _cluster(X, thr, num_speakers):
    from sklearn.cluster import AgglomerativeClustering

    if X.shape[0] < 2:
        # AgglomerativeClustering 至少需要 2 个样本；仅 1 个窗口时直接归为同一说话人
        return np.zeros(X.shape[0], dtype=int)
    if num_speakers and num_speakers >= 1:
        return AgglomerativeClustering(
            n_clusters=int(num_speakers), metric="cosine", linkage="average"
        ).fit_predict(X)
    return AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=float(thr),
        metric="cosine",
        linkage="average",
    ).fit_predict(X)


def _refine_and_smooth(X, lab, thr):
    # 1) 用簇质心重新归派（含离群点归派：弱归属且被更强簇吸引则改派）
    for _ in range(4):
        cents = {}
        for c in sorted(set(lab)):
            m = lab == c
            if m.sum():
                v = X[m].mean(0)
                cents[c] = v / (np.linalg.norm(v) + 1e-9)
        keys = sorted(cents)
        C = np.stack([cents[k] for k in keys])
        S = X @ C.T  # [N, n_clusters] 余弦相似度
        new = lab.copy()
        for i in range(len(lab)):
            best = int(S[i].argmax())
            best_sim = float(S[i, best])
            cur = int(lab[i])
            cur_sim = float(S[i, keys.index(cur)]) if cur in keys else -1.0
            # 离群点：当前归属弱，且有明显更强的簇
            if best_sim > 0.5 and (cur_sim < 0.4 or best_sim - cur_sim > 0.2):
                new[i] = keys[best]
        lab = new
        # 2) 合并质心高度相似的簇（相似度 > 1-thr）
        cents2 = {}
        for c in sorted(set(lab)):
            m = lab == c
            if m.sum():
                v = X[m].mean(0)
                cents2[c] = v / (np.linalg.norm(v) + 1e-9)
        keys = sorted(cents2)
        C2 = np.stack([cents2[k] for k in keys])
        remap, keep = {}, []
        for k in keys:
            hit = None
            for kk in keep:
                if float(cents2[k] @ cents2[kk]) > 1 - float(thr):
                    hit = kk
                    break
            if hit is None:
                keep.append(k)
                remap[k] = k
            else:
                remap[k] = hit
        new = np.array([remap[c] for c in lab])
        if np.array_equal(new, lab):
            break
        lab = new
    # 3) 时间中值滤波平滑
    if len(lab) >= 3:
        sm = lab.copy()
        for i in range(1, len(lab) - 1):
            w = lab[max(0, i - 1): i + 2]
            sm[i] = collections.Counter(w.tolist()).most_common(1)[0][0]
        lab = sm
    return lab


def diarize(y, embedder, thr=0.45, num_speakers=0, min_seg=1.0, gap=0.40, min_speaker_dur=2.5):
    """对 16k 单声道人声 y 做说话人分离，返回 [(start_s, end_s, spk_id), ...]。"""
    regs = vad_regions(y)
    if not regs:
        return [], 0
    wins = windows_from_regions(regs)
    if not wins:
        return [], 0
    chunks = [y[int(a * _DIAR_SR): int(b * _DIAR_SR)] for (a, b) in wins]
    X = embedder.embed(chunks)
    lab = _cluster(X, thr, num_speakers)
    lab = _refine_and_smooth(X, lab, thr)
    # 窗口 → 说话人分段（合并相邻同人窗口）
    turns = []
    for (a, b), c in zip(wins, lab):
        if turns and turns[-1][2] == c and a - turns[-1][1] <= gap:
            turns[-1][1] = max(turns[-1][1], b)
        else:
            turns.append([a, b, c])
    turns = [t for t in turns if t[1] - t[0] >= min_seg]
    if not turns:
        return [], 0
    # 按累计语音时长重排 id（0=说话最多），并丢弃过短的簇（背景音乐人声/噪声片段）
    dur = collections.Counter()
    for a, b, c in turns:
        dur[c] += b - a
    keep_clusters = {c for c, d in dur.items() if d >= min_speaker_dur}
    if not keep_clusters:
        # 没有任何簇达到阈值：保留最大簇，避免空输出
        keep_clusters = {dur.most_common(1)[0][0]}
    turns = [t for t in turns if t[2] in keep_clusters]
    order = {old: i for i, (old, _) in enumerate(dur.most_common()) if old in keep_clusters}
    turns = [(a, b, order[c]) for a, b, c in turns]
    return turns, len(order)


# --------------------------------------------------------------------------- #
# 人声 / 背景分离（audio_separator，mel_band_roformer）
# --------------------------------------------------------------------------- #
def _load_sep_model(sep, name: str) -> bool:
    """加载分离模型；严格 SSL 下载失败时退化为不校验证书重试（常见于公司代理/缺失CA）。"""
    try:
        sep.load_model(name)
        return True
    except Exception:
        pass
    # 退化重试：关闭证书校验（仅用于模型下载这一步，不影响推理）
    try:
        import ssl

        import requests

        ssl._create_default_https_context = ssl._create_unverified_context
        try:
            requests.packages.urllib3.disable_warnings()
        except Exception:
            pass
        _orig = requests.get

        def _no_verify(*a, **k):
            k.setdefault("verify", False)
            return _orig(*a, **k)

        requests.get = _no_verify
        sep.load_model(name)
        requests.get = _orig
        return True
    except Exception as e:
        logger.warning("分离模型加载失败: %s", e)
        return False


def separate_vocal_instrumental(wav: np.ndarray, sr: int):
    """返回 (vocals_arr, vocals_sr, instr_arr, instr_sr) 或 None。"""
    tmp_out = None
    try:
        import ssl

        ssl._create_default_https_context = ssl._create_unverified_context
        from audio_separator.separator import Separator
        import soundfile as sf

        model_dir = os.path.join(os.path.dirname(__file__), "models", "audio_separator")
        os.makedirs(model_dir, exist_ok=True)
        # 临时输出目录必须在“构造 Separator 之前”建好并传入。
        # audio_separator 真正落盘用的是 model_instance.output_dir（在 load_model 时
        # 从 Separator.output_dir 固化），事后只改 sep.output_dir 不会同步过去，
        # 会导致分离结果被写进“当前工作目录”（曾把 61MB tmp*.wav 污染到 custom_nodes 目录）。
        tmp_out = tempfile.mkdtemp(prefix="breeze_sep_")
        sep = Separator(model_file_dir=model_dir, output_format="WAV", output_dir=tmp_out)
        ok = _load_sep_model(sep, _SEP_MODEL_NAME)
        if not ok:
            ok = _load_sep_model(sep, "model_mel_band_roformer_ep_3005_sdr_11")
        if not ok:
            shutil.rmtree(tmp_out, ignore_errors=True)
            return None
        # 双保险：显式同步到已加载的模型实例
        _mi = getattr(sep, "model_instance", None)
        if _mi is not None:
            try:
                _mi.output_dir = tmp_out
            except Exception:
                pass
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            sf.write(f.name, wav, sr)
            tmp_in = f.name
        try:
            outputs = sep.separate(tmp_in)
        finally:
            try:
                os.unlink(tmp_in)
            except Exception:
                pass

        # audio_separator 返回的是“相对文件名”，需按 tmp_out 解析成绝对路径
        raw = outputs if isinstance(outputs, (list, tuple)) else [outputs]
        files = []
        for out in raw:
            if not out:
                continue
            p = out if os.path.isabs(out) else os.path.join(tmp_out, out)
            if os.path.exists(p):
                files.append(p)
        vocal_file = instr_file = None
        for out in files:
            low = os.path.basename(out).lower()
            if vocal_file is None and ("vocal" in low or "speech" in low):
                vocal_file = out
            elif instr_file is None and ("instrument" in low or "accomp" in low or "back" in low or "bgm" in low):
                instr_file = out
        if vocal_file is None and files:
            # 没识别出 stems 时，退化为第一个可用文件
            vocal_file = files[0]
        if vocal_file is None:
            shutil.rmtree(tmp_out, ignore_errors=True)
            return None

        try:
            v, vsr = sf.read(vocal_file, dtype="float32", always_2d=False)
            v = _to_mono(v)
            i, isr = (None, 0)
            if instr_file and os.path.exists(instr_file):
                i, isr = sf.read(instr_file, dtype="float32", always_2d=False)
                i = _to_mono(i)
        finally:
            shutil.rmtree(tmp_out, ignore_errors=True)
        return v, int(vsr), i, int(isr)
    except Exception as e:
        logger.warning("人声分离失败，回退到原始音频: %s", e)
        if tmp_out:
            shutil.rmtree(tmp_out, ignore_errors=True)
        return None


# --------------------------------------------------------------------------- #
# 片段清洗（降噪 + 峰值归一化），用于产出可克隆音源
# --------------------------------------------------------------------------- #
def _denoise(y: np.ndarray, sr: int) -> np.ndarray:
    try:
        import noisereduce as nr

        return nr.reduce_noise(y=y, sr=sr, prop_decrease=0.6, stationary=False)
    except Exception:
        return y


def _normalize_peak(y: np.ndarray, target_db: float = -3.0) -> np.ndarray:
    peak = np.max(np.abs(y))
    if peak > 0:
        y = y * (10 ** (target_db / 20) / peak)
    return y.astype(np.float32)


def _clean_segment(y: np.ndarray, sr: int, denoise: bool) -> np.ndarray:
    if denoise:
        y = _denoise(y, sr)
    return _normalize_peak(y, -3.0)


# --------------------------------------------------------------------------- #
# 重叠感知裁剪 / 克隆质量过滤 支撑函数
# --------------------------------------------------------------------------- #
def _embed(y16: np.ndarray, wins, embedder) -> np.ndarray:
    """对给定时间窗 [(a,b),...] 抽声纹并归一化，返回 [M, D]。"""
    chunks = []
    for (a, b) in wins:
        s, e = int(round(a * _DIAR_SR)), int(round(b * _DIAR_SR))
        if e > s:
            chunks.append(y16[s:e])
    if not chunks:
        return np.zeros((0, 1), np.float32)
    X = np.asarray(embedder.embed(chunks), dtype=np.float32)
    if X.ndim != 2 or X.shape[0] == 0:
        return np.zeros((0, 1), np.float32)
    n = np.linalg.norm(X, axis=1, keepdims=True)
    X = X / (n + 1e-9)
    return X


def _spk_centroid(y16: np.ndarray, turns, embedder):
    """取某说话人各 turn 中心窗的声纹平均作为质心（归一化向量）。"""
    vecs = []
    for (a, b, _) in turns:
        mid = (a + b) / 2
        wa, wb = max(a, mid - 0.75), min(b, mid + 0.75)
        if wb - wa >= 0.3:
            vecs.append((wa, wb))
    if vecs:
        X = _embed(y16, vecs, embedder)
        if X.shape[0]:
            c = X.mean(0)
            return c / (np.linalg.norm(c) + 1e-9)
    return None


def _fine_wins(turn):
    a, b = turn[0], turn[1]
    if b - a <= _FINE_WIN:
        return [(a, b)]
    out = []
    t = a
    while t < b - 1e-6:
        e = min(t + _FINE_WIN, b)
        if e - t >= _FINE_MIN:
            out.append((t, e))
        t += _FINE_HOP
    return out


def _refine_turn(y16, turn, embedder, cents, spk_idx, gap=0.25):
    """细窗逐窗重算主导说话人：保留 dominant==本说话人 的窗口，剔除非本说话人
    /模糊重叠窗口。返回 (干净子段[(sa,sb),...], 纯度)。纯度=本说话人主导窗占比。"""
    wins = _fine_wins(turn)
    if not wins or cents.shape[0] == 0:
        return [], 1.0
    X = _embed(y16, wins, embedder)
    if X.shape[0] == 0:
        return [], 1.0
    sims = X @ cents.T  # [M, n_spk] 余弦相似度
    dom = sims.argmax(1)
    purity = float((dom == spk_idx).mean()) if len(dom) else 1.0
    # 连续“本说话人”窗口 → 子段
    sub, run = [], []
    for (wa, wb), d in zip(wins, dom):
        if d == spk_idx:
            run.append((wa, wb))
        else:
            if run:
                sub.append((run[0][0], run[-1][1]))
                run = []
    if run:
        sub.append((run[0][0], run[-1][1]))
    # 合并间隔很小的子段
    merged = []
    for s0, e0 in sub:
        if merged and s0 - merged[-1][1] <= gap:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e0))
        else:
            merged.append((s0, e0))
    return merged, purity


def _trim_to_speech(clip: np.ndarray, sr: int, pad_ms: float = 150) -> np.ndarray:
    """把片段裁到实际人声范围（前后留少量 pad），确保拼接时不会在半句话处被切断。"""
    y16 = clip if sr == _DIAR_SR else _resample(clip, sr, _DIAR_SR)
    regs = vad_regions(y16)
    if not regs:
        return clip
    s = max(0, int((regs[0][0] - pad_ms / 1000) * sr))
    e = min(len(clip), int((regs[-1][1] + pad_ms / 1000) * sr))
    if e <= s:
        return clip
    return clip[s:e]


def _concat_clips(clips, sr: int, pad_s: float = 0.12) -> np.ndarray:
    """把多个干净片段用静音pad拼接成一段连续克隆素材（拼接点落在静音，不会半句切断）。"""
    clips = [c for c in clips if c is not None and len(c) > 0]
    if not clips:
        return np.zeros(int(0.1 * sr), np.float32)
    pad = int(round(pad_s * sr))
    out = [clips[0]]
    for c in clips[1:]:
        out.append(np.zeros(pad, np.float32))
        out.append(c)
    return np.concatenate(out).astype(np.float32)


# --------------------------------------------------------------------------- #
# 切片构建
# --------------------------------------------------------------------------- #
def _build_full_multi(src_arrays, sr: int, turns) -> np.ndarray:
    """原样导出：保持原始时间轴的单说话人轨道（他人/停顿处静音）。
    turns 元素=(a,b,src_idx)，从对应分离流取音频，从而去除串音。"""
    # 以最长源的长度建立静音底
    maxlen = max((len(a) for a in src_arrays), default=1)
    out = np.zeros(maxlen, dtype=np.float32)
    for a, b, si in turns:
        src = src_arrays[si] if si < len(src_arrays) else src_arrays[0]
        s, e = int(round(a * sr)), int(round(b * sr))
        s = max(0, min(s, len(src)))
        e = max(0, min(e, len(src)))
        if e > s:
            out[s:e] = src[s:e]
    return out


# --------------------------------------------------------------------------- #
# 节点 1：多人说话人分离 + 背景音保留
# --------------------------------------------------------------------------- #
class BreezeTTS2MultiSpeakerExtractor:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": (AUDIO_TYPE,),
                "extract_channel": (
                    ["人声(去背景)", "人声+背景", "仅背景音"],
                    {
                        "default": "人声(去背景)",
                        "tooltip": "提取通道：人声(去背景)=纯净人声最适合克隆；人声+背景=人声上叠背景音；仅背景音=只导出背景/伴奏通道",
                    },
                ),
                "separate_vocals": (
                    ["enable", "disable"],
                    {
                        "default": "enable",
                        "tooltip": "是否用 mel_band_roformer 分离人声与背景（首次会自动下载模型）。关闭则直接在原始音频上分离说话人",
                    },
                ),
                "vocal_separation": (
                    ["MossFormer2两路物理分离", "仅声纹聚类(快)"],
                    {
                        "default": "MossFormer2两路物理分离",
                        "tooltip": "vocal → MossFormer2 两路物理分离 → 各路再克隆：从根本上解决两人同时说话的串音（需 MossFormer2_SS_16K 权重，缺失时自动回退到仅声纹聚类）。仅对'人声(去背景)'通道生效",
                    },
                ),
                "num_speakers": (
                    ["自动检测", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10"],
                    {"default": "自动检测", "tooltip": "说话人数量：自动检测按声纹聚类，或强制指定"},
                ),
                "export_mode": (
                    ["原样导出(单人单文件)", "自动切片(子文件夹)"],
                    {
                        "default": "自动切片(子文件夹)",
                        "tooltip": "原样导出=每个说话人一个WAV保留原始时间轴(他人处静音)；自动切片=每个说话人一个子文件夹，每段utterance单独一个WAV",
                    },
                ),
                "overlap_trim": (
                    ["enable", "disable"],
                    {
                        "default": "enable",
                        "tooltip": "重叠感知裁剪：对每个分段细窗逐窗重算主导说话人，剔除被他人主导/模糊重叠的窗口，消除串音(基于声纹，非重模型)",
                    },
                ),
                "min_purity": (
                    "FLOAT",
                    {"default": 0.6, "min": 0.3, "max": 1.0, "step": 0.05,
                     "tooltip": "有效克隆片段的最低纯度(本说话人主导窗占比)；低于则视为串音/不纯，过滤掉。仅 overlap_trim=disable 时用于过滤"},
                ),
                "min_clone_duration": (
                    "FLOAT",
                    {"default": 3.0, "min": 0.5, "max": 15.0, "step": 0.5,
                     "tooltip": "可克隆片段最低时长(秒)：短于此视为时长太短，直接过滤。低于此的片段不会进入克隆素材"},
                ),
                "target_clone_duration": (
                    "FLOAT",
                    {"default": 8.0, "min": 1.0, "max": 30.0, "step": 0.5,
                     "tooltip": "期望的单说话人克隆素材总时长；低于此值会在报告中提示'偏短(已拼接全部有效片段)'，仍输出拼接结果"},
                ),
                "min_segment": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.3, "max": 10.0, "step": 0.1,
                     "tooltip": "自动切片时，最短 utterance 长度（秒），短于此的片段丢弃"},
                ),
                "min_speaker_duration": (
                    "FLOAT",
                    {"default": 3.0, "min": 0.0, "max": 30.0, "step": 0.5,
                     "tooltip": "总语音短于此秒数的说话人视为背景音乐人声/噪声片段并丢弃（设为0保留全部）。已知人数时建议用 num_speakers 强制"},
                ),
                "target_sr": (
                    ["24000", "16000", "44100", "48000"],
                    {"default": "24000", "tooltip": "导出采样率（Breeze 克隆建议 24000）"},
                ),
                "clean_clips": (
                    ["enable", "disable"],
                    {"default": "enable", "tooltip": "对导出的每段做降噪(仅去背景通道)+峰值归一化到-3dB，产出更干净的可克隆音源"},
                ),
                "cluster_threshold": (
                    "FLOAT",
                    {"default": 0.45, "min": 0.2, "max": 0.8, "step": 0.01,
                     "tooltip": "声纹聚类余弦距离阈值：越小分得越细(可能过拆)，越大越粗(可能并人)。0.45 较稳"},
                ),
                "silence_gap_merge": (
                    "FLOAT",
                    {"default": 0.40, "min": 0.0, "max": 2.0, "step": 0.05,
                     "tooltip": "同说话人相邻窗口间隔小于此值(秒)则合并为同一段"},
                ),
                "output_dir": (
                    "STRING",
                    {"default": "", "multiline": False,
                     "tooltip": "导出目录：留空=ComfyUI/output/BreezeTTS2_Speakers/；可填绝对路径"},
                ),
                "filename_prefix": (
                    "STRING",
                    {"default": "Speaker", "multiline": False,
                     "tooltip": "文件名前缀，如 Speaker → Speaker_1.wav / Speaker_1/seg_001.wav"},
                ),
            }
        }

    RETURN_TYPES = (SPEAKERS_TYPE, "STRING")
    RETURN_NAMES = ("speakers", "report")
    FUNCTION = "extract"
    CATEGORY = "Breeze TTS 2"
    DESCRIPTION = "多人对话+背景音 → 分离各说话人并导出高品质可克隆音源；背景音作为独立通道可单独保留。"

    def extract(
        self, audio, extract_channel, separate_vocals, vocal_separation,
        num_speakers, export_mode, overlap_trim, min_segment,
        min_speaker_duration, min_purity, min_clone_duration,
        target_clone_duration, target_sr, clean_clips, cluster_threshold,
        silence_gap_merge, output_dir, filename_prefix,
    ):
        t0 = time.time()
        target_sr = int(target_sr)
        num_spk = 0 if num_speakers == "自动检测" else int(num_speakers)
        do_slice = export_mode.startswith("自动切片")
        denoise = clean_clips == "enable" and extract_channel == "人声(去背景)"

        report = []
        report.append(f"[分离] 通道={extract_channel} 分离={separate_vocals} "
                      f"物理分离={vocal_separation} 人数={num_speakers} "
                      f"导出={export_mode} 阈值={cluster_threshold}")
        report.append(f"[过滤] 重叠裁剪={overlap_trim} 最低纯度={min_purity} "
                      f"最短克隆={min_clone_duration}s 目标克隆={target_clone_duration}s")

        # 1) 原始音频
        orig, orig_sr = _comfy_audio_to_np(audio)
        orig = _to_mono(orig)
        report.append(f"[输入] {len(orig)/orig_sr:.2f}s @ {orig_sr}Hz")

        # 2) 人声/背景分离
        vocals = orig
        vocals_sr = orig_sr
        instr = None
        instr_sr = 0
        if separate_vocals == "enable" and extract_channel != "仅背景音":
            sep = separate_vocal_instrumental(orig, orig_sr)
            if sep is not None:
                v, vsr, i, isr = sep
                if _is_silent(v):
                    # 分离模型判定此音频无独立人声（纯器乐/锣鼓/强伴奏等）→ 人声轨近乎静音。
                    # 若沿用空人声轨，VAD 检测不到任何语音 → 空输出。自动回退到原始音频作为人声源。
                    vocals, vocals_sr = orig, orig_sr
                    instr, instr_sr = None, 0
                    report.append("[分离] ⚠ 分离得到的人声轨近乎静音(该音频可能无独立人声/纯器乐)，"
                                  "已自动回退到原始音频作为人声源")
                else:
                    vocals, vocals_sr = v, vsr
                    instr, instr_sr = i, isr
                    report.append(f"[分离] 人声 {len(vocals)/vocals_sr:.2f}s @ {vocals_sr}Hz"
                                  + (f"；背景 {len(i)/isr:.2f}s" if i is not None else "；背景缺失"))
            else:
                report.append("[分离] 未成功，使用原始音频作为人声")
        elif extract_channel == "仅背景音":
            sep = separate_vocal_instrumental(orig, orig_sr)
            if sep is not None and sep[2] is not None:
                i, isr = sep[2], sep[3]
                instr, instr_sr = i, isr
                report.append(f"[分离] 背景 {len(i)/isr:.2f}s @ {isr}Hz")
            else:
                report.append("[分离] 未成功，回退到原始音频作为背景")

        # 3) 确定用于导出的通道（在 target_sr 下）
        if extract_channel == "仅背景音":
            chan_arr = _resample(instr if instr is not None else orig,
                                 instr_sr if instr is not None else orig_sr, target_sr)
            chan_label = "background"
        elif extract_channel == "人声+背景" and instr is not None:
            va = _resample(vocals, vocals_sr, target_sr)
            ia = _resample(instr, instr_sr, target_sr)
            chan_arr = (va + ia).astype(np.float32)
            chan_label = "vocals+background"
        else:
            chan_arr = _resample(vocals, vocals_sr, target_sr)
            chan_label = "vocals"

        # 4) 说话人分离（对 vocals 做，背景通道除外）
        # speakers_turns: list of (label, st_list, tag)，st_list 元素 = (start_s, end_s, src_idx)
        # src_idx 指向 src_16k / chan_streams 中对应的音频源（MossFormer2 各路或原 vocal）
        speakers_turns = []
        chan_streams = [chan_arr]  # target_sr 导出源（默认=人声/背景轨道，在 else 分支按实际重设）
        src_16k = [chan_arr]       # 16k 声纹逐窗嵌入源（在 else 分支按实际重设）
        if extract_channel == "仅背景音":
            dur = len(chan_arr) / target_sr
            speakers_turns.append(("Background", [(0.0, dur, 0)], "background"))
            n_spk = 1
            report.append(f"[分离] 背景通道：单轨道，时长 {dur:.2f}s")
        else:
            y16 = _resample(vocals, vocals_sr, _DIAR_SR)
            try:
                from .wespeaker_sv import WespeakerEmbedder
            except ImportError:
                from wespeaker_sv import WespeakerEmbedder

            embedder = WespeakerEmbedder()
            report.append(f"[声纹] wespeaker ONNX ({'/'.join(embedder.providers)})")

            # 是否启用 MossFormer2 两路物理分离（仅对'人声(去背景)'生效）
            use_mf = (vocal_separation == "MossFormer2两路物理分离"
                      and extract_channel == "人声(去背景)")
            cv, mf_err = (None, "")
            if use_mf:
                cv, mf_err = _load_mossformer()
                if cv is None:
                    report.append(f"[分离] MossFormer2 不可用({mf_err})，回退到仅声纹聚类")
                    use_mf = False
                else:
                    report.append(f"[分离] MossFormer2 两路物理分离 CUDA={cv.models[0].device}")

            if use_mf:
                # ---- vocal → 两路物理分离 → 各路再聚类 → 跨流全局归并 ----
                streams16 = _mossformer_separate(y16, cv)   # 每路 (T,) @16k
                chan_streams = [_resample(s, _DIAR_SR, target_sr) for s in streams16]
                src_16k = list(streams16)
                # 4.0) 每路独立 diarize（局部窗口），再汇总做全局聚类
                all_wins, all_embs = [], []   # (src_idx, a, b) + emb
                for si, s16 in enumerate(streams16):
                    lt, _ = diarize(s16, embedder, thr=cluster_threshold, num_speakers=0,
                                    min_seg=float(min_segment), gap=float(silence_gap_merge),
                                    min_speaker_dur=0.0)
                    for (a, b, _) in lt:
                        emb = _embed(s16, [(a, b)], embedder)
                        if emb.shape[0]:
                            all_embs.append(emb[0])
                            all_wins.append((si, a, b))
                if not all_wins:
                    report.append("[分离] MossFormer2 两路均未检测到人声，无输出。"
                                  "建议：关闭「人声分离」(separate_vocals=disable) 直接对原音频分离，"
                                  "或确认该音频确有清晰人声")
                    return ([], "\n".join(report))
                Xg = np.stack(all_embs).astype(np.float32)
                glab = _cluster(Xg, cluster_threshold, num_spk)
                glab = _refine_and_smooth(Xg, glab, cluster_threshold)
                n_spk = len(set(int(g) for g in glab))
                # 4.0.1) 按全局说话人聚合窗口（携带 src_idx）
                g_turns = collections.defaultdict(list)
                for (si, a, b), g in zip(all_wins, glab):
                    g_turns[int(g)].append((a, b, si))
                # 4.1) 全局质心（从各窗口所属分离流取嵌入）
                raw_c = []
                for g in range(n_spk):
                    vecs = []
                    for (a, b, si) in g_turns[g]:
                        mid = (a + b) / 2
                        wa, wb = max(a, mid - 0.75), min(b, mid + 0.75)
                        if wb - wa >= 0.3:
                            e = _embed(src_16k[si], [(wa, wb)], embedder)
                            if e.shape[0]:
                                vecs.append(e[0])
                    if vecs:
                        c = np.mean(vecs, 0)
                        raw_c.append(c / (np.linalg.norm(c) + 1e-9))
                    else:
                        raw_c.append(None)
                dim = next((c.shape[0] for c in raw_c if c is not None), 256)
                cents = np.stack([(c if c is not None else np.zeros(dim))
                                  for c in raw_c]).astype(np.float32)
                # 4.2) 重叠感知裁剪 + 克隆质量过滤（逐窗从其所属分离流取音频）
                do_trim = overlap_trim == "enable"
                min_cd = float(min_clone_duration)
                min_pu = float(min_purity)
                tgt_cd = float(target_clone_duration)
                for g in range(n_spk):
                    label = f"{filename_prefix}_{g+1}"
                    refined = []
                    for (a, b, si) in g_turns[g]:
                        if do_trim:
                            subs, _ = _refine_turn(src_16k[si], (a, b, si),
                                                  embedder, cents, g)
                            for (sa, sb) in subs:
                                refined.append((sa, sb, si, 1.0))
                        else:
                            purity = 1.0
                            if cents.shape[0] == n_spk:
                                _, purity = _refine_turn(src_16k[si], (a, b, si),
                                                        embedder, cents, g)
                            refined.append((a, b, si, purity))
                    valid = [(a, b, si) for (a, b, si, p) in refined
                             if (b - a) >= min_cd and p >= min_pu]
                    total = sum(b - a for a, b, _ in valid)
                    if not valid:
                        report.append(f"  ● {label}: ⚠ 无有效克隆片段(全部过短/串音)，已跳过")
                        continue
                    short_flag = " (偏短·已拼接全部有效片段)" if total < tgt_cd else ""
                    speakers_turns.append((label, valid, "speaker"))
                    report.append(f"  ● {label}: 有效 {len(valid)}段, 语音 {total:.2f}s{short_flag}")
            else:
                # ---- 原路径：直接对混合 vocal 做声纹聚类 ----
                src_16k = [y16]
                turns, n_spk = diarize(
                    y16, embedder, thr=cluster_threshold, num_speakers=num_spk,
                    min_seg=float(min_segment), gap=float(silence_gap_merge),
                    min_speaker_dur=float(min_speaker_duration),
                )
                if not turns:
                    report.append("[分离] 未检测到人声，无输出。"
                                  "建议：关闭「人声分离」(separate_vocals=disable) 直接对原音频分离，"
                                  "或降低 min_segment / 更换含清晰人声的音频")
                    return ([], "\n".join(report))
                spk_turns = collections.defaultdict(list)
                for t in turns:
                    spk_turns[t[2]].append(t)
                raw_c = [_spk_centroid(y16, spk_turns[s], embedder) for s in range(n_spk)]
                dim = next((c.shape[0] for c in raw_c if c is not None), 256)
                cents = np.stack([(c if c is not None else np.zeros(dim))
                                  for c in raw_c]).astype(np.float32)
                do_trim = overlap_trim == "enable" and cents.shape[0] == n_spk and n_spk > 0
                min_cd = float(min_clone_duration)
                min_pu = float(min_purity)
                tgt_cd = float(target_clone_duration)
                for spk in range(n_spk):
                    label = f"{filename_prefix}_{spk+1}"
                    refined = []
                    for t in spk_turns[spk]:
                        if do_trim:
                            subs, _ = _refine_turn(y16, t, embedder, cents, spk)
                            for (sa, sb) in subs:
                                refined.append((sa, sb, 0, 1.0))
                        else:
                            purity = 1.0
                            if cents.shape[0] == n_spk:
                                _, purity = _refine_turn(y16, t, embedder, cents, spk)
                            refined.append((t[0], t[1], 0, purity))
                    valid = [(a, b, 0) for (a, b, _, p) in refined
                             if (b - a) >= min_cd and p >= min_pu]
                    total = sum(b - a for a, b, _ in valid)
                    if not valid:
                        report.append(f"  ● {label}: ⚠ 无有效克隆片段(全部过短/串音)，已跳过")
                        continue
                    short_flag = " (偏短·已拼接全部有效片段)" if total < tgt_cd else ""
                    speakers_turns.append((label, valid, "speaker"))
                    report.append(f"  ● {label}: 有效 {len(valid)}段, 语音 {total:.2f}s{short_flag}")

        # 4.9) 全局兜底：所有说话人都没产出有效片段 → 明确说明原因与可调项，不抛异常
        if not speakers_turns:
            report.append("[结果] 未产出任何可用说话人/克隆片段。可尝试："
                          "① 关闭「人声分离」(separate_vocals=disable) 直接对原音频分离；"
                          "② 调低「最短克隆时长」(min_clone_duration) 或「最低纯度」(min_purity)；"
                          "③ 更换含清晰人声的音频。")

        # 5) 导出目录
        base = output_dir.strip()
        if not base:
            try:
                import folder_paths

                base = os.path.join(folder_paths.get_output_directory(), "BreezeTTS2_Speakers")
            except Exception:
                base = os.path.join(os.path.dirname(__file__), "BreezeTTS2_Speakers")
        run_dir = os.path.join(base, time.strftime("%Y%m%d_%H%M%S"))
        os.makedirs(run_dir, exist_ok=True)

        import soundfile as sf

        speakers_obj = []
        for label, st, tag in speakers_turns:
            files = []
            if do_slice:
                sub = os.path.join(run_dir, label)
                os.makedirs(sub, exist_ok=True)
                segs_comfy = []
                clips = []
                for idx, (a, b, si) in enumerate(st, 1):
                    src = chan_streams[si] if si < len(chan_streams) else chan_arr
                    s, e = int(round(a * target_sr)), int(round(b * target_sr))
                    s = max(0, min(s, len(src)))
                    e = max(0, min(e, len(src)))
                    if e <= s:
                        continue
                    clip = src[s:e].copy()
                    clip = _trim_to_speech(clip, target_sr)  # 裁到实际人声，避免半句切断
                    if clean_clips == "enable":
                        clip = _clean_segment(clip, target_sr, denoise)
                    clips.append(clip)
                    path = os.path.join(sub, f"seg_{idx:03d}.wav")
                    sf.write(path, clip, target_sr)
                    files.append(path)
                    segs_comfy.append(_np_to_comfy(clip, target_sr))
                # 主音频 = 拼接全部有效片段（静音 pad 拼接，落在句间不切断）
                main = _concat_clips(clips, target_sr)
                if clean_clips == "enable":
                    main = _normalize_peak(main, -3.0)
                main_path = os.path.join(run_dir, f"{label}.wav")
                sf.write(main_path, main, target_sr)
                files.append(main_path)
                main_comfy = _np_to_comfy(main, target_sr)
                dur = float(sum(b - a for a, b, _ in st))
                report.append(f"    → {sub}/ ({len(files)}文件, 主音频 {len(main)/target_sr:.2f}s)")
                speakers_obj.append({
                    "label": label, "index": len(speakers_obj), "tag": tag,
                    "duration": dur, "num_turns": len(st), "num_segments": len(segs_comfy),
                    "audio": main_comfy, "segments": segs_comfy, "files": files,
                })
            else:
                # 原样导出：保留原始时间轴的单文件（只放干净子段，串音处留静音）
                full = _build_full_multi(chan_streams, target_sr, st)
                if clean_clips == "enable":
                    full = _normalize_peak(full, -3.0)
                path = os.path.join(run_dir, f"{label}.wav")
                sf.write(path, full, target_sr)
                dur = len(full) / target_sr
                speakers_obj.append({
                    "label": label, "index": len(speakers_obj), "tag": tag,
                    "duration": dur, "num_turns": len(st), "num_segments": 0,
                    "audio": _np_to_comfy(full, target_sr), "segments": [], "files": [path],
                })
                report.append(f"    → {path} ({dur:.2f}s)")

        if not speakers_obj:
            report.append("[完成] ⚠ 没有任何说话人产出有效克隆片段：可能全员过短/串音，或输入无清晰人声。"
                          "已按您的要求不报错——请给出更清晰、更长或重叠更少的音频后重试。")
            return ([], "\n".join(report))

        report.append(f"[完成] 说话人={n_spk} 有效导出={len(speakers_obj)}个 目录={run_dir} ({time.time()-t0:.1f}s)")
        return (speakers_obj, "\n".join(report))


# --------------------------------------------------------------------------- #
# 节点 2：说话人选择（挑出某一说话人接克隆流程）
# --------------------------------------------------------------------------- #
class BreezeTTS2SpeakerSelector:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "speakers": (SPEAKERS_TYPE,),
                "speaker_index": (
                    "INT",
                    {"default": 0, "min": 0, "max": 99, "step": 1,
                     "tooltip": "选择第几个说话人（0=第一个，按语音时长降序）"},
                ),
                "segment_index": (
                    "INT",
                    {"default": -1, "min": -1, "max": 999, "step": 1,
                     "tooltip": "-1=返回该说话人主音频；>=0=只返回第N段切片(自动切片模式)"},
                ),
            }
        }

    RETURN_TYPES = (AUDIO_TYPE, "STRING")
    RETURN_NAMES = ("audio", "info")
    FUNCTION = "select"
    CATEGORY = "Breeze TTS 2"
    DESCRIPTION = "从分离结果中挑出某一说话人（或某一段切片）作为克隆用的参考音频。"

    def select(self, speakers, speaker_index, segment_index):
        if not speakers:
            return (_np_to_comfy(np.zeros(2400, np.float32), 24000), "无说话人数据")
        idx = max(0, min(int(speaker_index), len(speakers) - 1))
        sp = speakers[idx]
        if segment_index >= 0 and sp["segments"]:
            si = max(0, min(int(segment_index), len(sp["segments"]) - 1))
            audio = sp["segments"][si]
            info = f"{sp['label']} 第{si+1}段 (共{len(sp['segments'])}段)"
        else:
            audio = sp["audio"]
            info = (f"{sp['label']} | 时长 {sp['duration']:.2f}s | "
                    f"轮次 {sp['num_turns']} | 切片 {sp['num_segments']} | 通道 {sp['tag']}")
        return (audio, info)


NODE_CLASS_MAPPINGS = {
    "BreezeTTS2MultiSpeakerExtractor": BreezeTTS2MultiSpeakerExtractor,
    "BreezeTTS2SpeakerSelector": BreezeTTS2SpeakerSelector,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BreezeTTS2MultiSpeakerExtractor": "Breeze TTS 2 多人说话人分离",
    "BreezeTTS2SpeakerSelector": "Breeze TTS 2 说话人选择",
}
