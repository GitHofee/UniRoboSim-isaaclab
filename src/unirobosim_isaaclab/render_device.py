"""Keep RTX physical GPU selection distinct from CUDA's masked ordinal."""
from collections.abc import Mapping


def renderer_gpu_override(environment: Mapping[str, str], device: str) -> int | None:
    """Resolve an explicitly verified RTX index, never reinterpret UUID/MIG masks."""
    explicit = environment.get("UNIROBOSIM_ISAACLAB_RENDER_GPU")
    if explicit is not None:
        if not explicit.isascii() or not explicit.isdecimal():
            raise ValueError("UNIROBOSIM_ISAACLAB_RENDER_GPU must be a nonnegative RTX physical index")
        return int(explicit)
    visible = environment.get("CUDA_VISIBLE_DEVICES")
    if visible is None:
        return None  # Preserve the SDK's unmasked device selection.
    if visible.strip() == "0" and device in ("cuda", "cuda:0"):
        return 0
    raise ValueError(
        "Masked CUDA device requires an explicit verified UNIROBOSIM_ISAACLAB_RENDER_GPU "
        "RTX physical index; CUDA ordinals, UUIDs and MIG identifiers are not RTX indices"
    )
