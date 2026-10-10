import sys, traceback, importlib.util
from pathlib import Path

plugin_dir = Path(r'E:\ComfyUI\custom_nodes\ComfyUI-Breeze-TTS-2')
spec = importlib.util.spec_from_file_location(
    "breeze_tts",
    plugin_dir / "__init__.py",
    submodule_search_locations=[str(plugin_dir)]
)
mod = importlib.util.module_from_spec(spec)
sys.modules["breeze_tts"] = mod
try:
    spec.loader.exec_module(mod)
    print('Module loaded')
    
    from breeze_tts.loader import load_breeze_bundle
    from breeze_tts.nodes import _generate_audio
    
    print('Loading model...')
    bundle = load_breeze_bundle(
        'int8 hybrid (recommended)',
        'bfloat16',
        'cuda',
        'sdpa',
        False,
        'cuda_graphs',
    )
    print('Model loaded OK')
    
    print('Generating audio...')
    result = _generate_audio(
        bundle=bundle,
        text='你好，这是一个测试。',
        instruction='',
        ref_audio=None,
        ref_text=None,
        cfg_scale=2.0,
        max_new_tokens=200,
        temperature=0.9,
        top_k=50,
        top_p=0.95,
        repetition_penalty=1.1,
        depth_temperature=0.9,
        depth_top_k=50,
        depth_top_p=0.95,
        warmup_frames=10,
        seed=42,
    )
    print(f'Generated audio: {result["audio"].shape}')
    print('SUCCESS')
except Exception as e:
    traceback.print_exc()
    print(f'FAILED: {e}')
