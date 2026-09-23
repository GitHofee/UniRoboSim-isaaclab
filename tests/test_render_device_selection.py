import pytest
from pathlib import Path
import runpy
renderer_gpu_override = runpy.run_path(str(Path(__file__).parents[1] / "src/unirobosim_isaaclab/render_device.py"))["renderer_gpu_override"]

@pytest.mark.parametrize("mask", ["3", "3,1", "GPU-deadbeef", "MIG-GPU-abc/1/0", "", "-1"])
def test_mask_not_guessed(mask):
    with pytest.raises(ValueError, match="explicit verified"):
        renderer_gpu_override({"CUDA_VISIBLE_DEVICES": mask}, "cuda:0")

@pytest.mark.parametrize("mask", ["3", "GPU-deadbeef", "MIG-GPU-abc/1/0", "3,1"])
def test_verified_renderer_index_separate_from_mask(mask):
    assert renderer_gpu_override({"CUDA_VISIBLE_DEVICES": mask, "UNIROBOSIM_ISAACLAB_RENDER_GPU": "3"}, "cuda:0") == 3

@pytest.mark.parametrize("value", ["-1", "GPU-abc", "3,1", "3.0", "٣"])
def test_invalid_override(value):
    with pytest.raises(ValueError):
        renderer_gpu_override({"UNIROBOSIM_ISAACLAB_RENDER_GPU": value}, "cuda:0")

def test_legacy_unmasked_and_identity_mask():
    assert renderer_gpu_override({}, "cuda:2") is None
    assert renderer_gpu_override({"CUDA_VISIBLE_DEVICES": "0"}, "cuda:0") == 0
