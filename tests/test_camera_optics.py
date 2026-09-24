from types import SimpleNamespace

import pytest

from unirobosim_isaaclab.camera_optics import author_opencv_pinhole, read_opencv_pinhole


class Attribute:
    def __init__(self):
        self.value = None

    def Set(self, value):
        self.value = value
        return True

    def Get(self):
        return self.value


class Prim:
    def __init__(self, available=True):
        self.attributes = {}
        self.available = available

    def ApplyAPI(self, name):
        assert name == "OmniLensDistortionOpenCvPinholeAPI"
        return self.available

    def GetAttribute(self, name):
        return self.attributes.setdefault(name, Attribute())


def test_full_native_lens_roundtrip_keeps_noncentral_anisotropic_intrinsics():
    calibration = SimpleNamespace(
        projection_model="opencv_pinhole",
        intrinsics=(612.3, 0.0, 639.2, 0.0, 617.4, 523.8, 0.0, 0.0, 1.0),
        distortion_coefficients=(-.1, .2, .003, -.004, .05, .006, .007, -.008),
    )
    prim = Prim()
    author_opencv_pinhole(prim, calibration, 1280, 1056, gf=SimpleNamespace(Vec2i=lambda *v: v))
    assert read_opencv_pinhole(prim) == (calibration.intrinsics, calibration.distortion_coefficients)
    assert prim.GetAttribute("omni:lensdistortion:opencvPinhole:imageSize").Get() == (1280, 1056)
    # Effective readback must reflect the renderer's live state, not the request.
    prim.GetAttribute("omni:lensdistortion:opencvPinhole:fx").Set(620.0)
    assert read_opencv_pinhole(prim)[0][0] == 620.0


def test_missing_renderer_schema_is_not_silently_metadata_only():
    with pytest.raises(RuntimeError, match="lacks OpenCV"):
        author_opencv_pinhole(Prim(False), SimpleNamespace(projection_model="opencv_pinhole"), 640, 400, gf=None)


def test_plain_camera_has_no_distortion_override():
    assert read_opencv_pinhole(Prim()) is None
