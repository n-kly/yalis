import torch

from yalis.attention.backends import AttentionBackend
from yalis.attention.registry import get_attention
from yalis.attention.nowmp_thresh.threshold_attention_nowmp import (
    init_nowmp_state,
)
from yalis.constants import EnginePhase


def _check_registration():
    for backend in (
        AttentionBackend.THRESH,
        AttentionBackend.THRESH_ATTN_NOWMP,
        AttentionBackend.DOUBLE_SPARSE,
    ):
        fn = get_attention(backend.value)
        print(f"registered {backend.value}: {fn.__name__}")


def _smoke_thresh_warmup_cpu():
    torch.manual_seed(0)
    B, H, D = 1, 2, 4
    t_max = 8
    device = torch.device("cpu")
    dtype = torch.float32

    q = torch.randn((B, H, 1, D), device=device, dtype=dtype)
    k = torch.randn((B, H, 1, D), device=device, dtype=dtype)
    v = torch.randn((B, H, 1, D), device=device, dtype=dtype)
    k_cache = torch.zeros((B, H, t_max, D), device=device, dtype=dtype)
    v_cache = torch.zeros((B, H, t_max, D), device=device, dtype=dtype)
    token_counter = torch.zeros((B,), device=device, dtype=torch.int32)
    generation_counter = torch.ones((B,), device=device, dtype=torch.int32)
    warmup_quantiles = torch.zeros((B, H, 2), device=device, dtype=dtype)
    retain_perc = torch.zeros((B, 1), device=device, dtype=torch.float32)
    powerlaw_a = torch.ones((B, H), device=device, dtype=dtype)
    powerlaw_b = torch.ones((B, H), device=device, dtype=dtype)
    rotary_cos = torch.ones((t_max, D), device=device, dtype=dtype)
    rotary_sin = torch.zeros((t_max, D), device=device, dtype=dtype)

    fn = get_attention(AttentionBackend.THRESH.value)
    out = fn(
        q=q,
        k=k,
        v=v,
        phase=EnginePhase.DECODE_SINGLE,
        k_cache=k_cache,
        v_cache=v_cache,
        cache_seqlens=token_counter,
        rotary_cos=rotary_cos,
        rotary_sin=rotary_sin,
        generation_counter=generation_counter,
        warmup_quantiles=warmup_quantiles,
        warmup=True,
        threshold_percentile=0.5,
        retain_perc=retain_perc,
        powerlaw_a=powerlaw_a,
        powerlaw_b=powerlaw_b,
    )
    print("thresh warmup output shape:", tuple(out.shape))


def _smoke_nowmp_cuda():
    if not torch.cuda.is_available():
        print("CUDA not available; skipping nowmp decode smoke test.")
        return

    torch.manual_seed(0)
    B, H, D = 1, 2, 16
    t_max = 8
    device = torch.device("cuda")
    dtype = torch.float16

    q = torch.randn((B, H, 1, D), device=device, dtype=dtype)
    k = torch.randn((B, H, 1, D), device=device, dtype=dtype)
    v = torch.randn((B, H, 1, D), device=device, dtype=dtype)
    k_cache = torch.zeros((B, H, t_max, D), device=device, dtype=dtype)
    v_cache = torch.zeros((B, H, t_max, D), device=device, dtype=dtype)
    token_counter = torch.zeros((B,), device=device, dtype=torch.int32)
    retain_perc = torch.zeros((B, 1), device=device, dtype=torch.float32)
    rotary_cos = torch.ones((t_max, D), device=device, dtype=dtype)
    rotary_sin = torch.zeros((t_max, D), device=device, dtype=dtype)
    nowmp_state = init_nowmp_state(B, H, device=device)

    fn = get_attention(AttentionBackend.THRESH_ATTN_NOWMP.value)
    out = fn(
        q=q,
        k=k,
        v=v,
        phase=EnginePhase.DECODE_SINGLE,
        k_cache=k_cache,
        v_cache=v_cache,
        cache_seqlens=token_counter,
        rotary_cos=rotary_cos,
        rotary_sin=rotary_sin,
        threshold_percentile=0.5,
        nowmp_state=nowmp_state,
        retain_perc=retain_perc,
    )
    print("nowmp output shape:", tuple(out.shape))


def main():
    _check_registration()
    _smoke_thresh_warmup_cpu()
    _smoke_nowmp_cuda()


if __name__ == "__main__":
    main()
