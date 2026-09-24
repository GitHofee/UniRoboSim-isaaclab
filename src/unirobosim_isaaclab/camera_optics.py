"""Author and read calibrated native RTX lenses without changing image pixels."""

from __future__ import annotations

_COEFFICIENTS = ("k1", "k2", "p1", "p2", "k3", "k4", "k5", "k6")
_PREFIX = "omni:lensdistortion:opencvPinhole:"


def author_opencv_pinhole(camera_prim, calibration, width: int, height: int, *, gf) -> None:
    """Use the same RTX schema as Isaac Sim's official OpenCV camera API.

    Fail closed when the installed renderer schema is unavailable. Merely saving
    calibration metadata while rendering an undistorted image is not supported.
    """
    if calibration.projection_model != "opencv_pinhole":
        raise ValueError("unsupported calibrated camera projection")
    if not camera_prim.ApplyAPI("OmniLensDistortionOpenCvPinholeAPI"):
        raise RuntimeError("installed Isaac renderer lacks OpenCV pinhole lens schema")
    values = {
        "omni:lensdistortion:model": "opencvPinhole",
        _PREFIX + "fx": calibration.intrinsics[0],
        _PREFIX + "fy": calibration.intrinsics[4],
        _PREFIX + "cx": calibration.intrinsics[2],
        _PREFIX + "cy": calibration.intrinsics[5],
        _PREFIX + "imageSize": gf.Vec2i(width, height),
        **{_PREFIX + key: value for key, value in zip(_COEFFICIENTS, calibration.distortion_coefficients, strict=True)},
        **{_PREFIX + key: 0.0 for key in ("s1", "s2", "s3", "s4")},
    }
    for name, value in values.items():
        attribute = camera_prim.GetAttribute(name)
        if not attribute or not attribute.Set(value):
            raise RuntimeError(f"cannot author calibrated camera attribute {name}")
    # These are authored USD camera values in native USD units, not SI values.
    for field, attribute_name in (
        ("focal_length", "focalLength"),
        ("focus_distance", "focusDistance"),
        ("horizontal_aperture", "horizontalAperture"),
        ("vertical_aperture", "verticalAperture"),
        ("fisheye_resolution_budget", "fisheyeResolutionBudget"),
    ):
        value = getattr(calibration, field, None)
        if value is not None:
            attribute = camera_prim.GetAttribute(attribute_name)
            if not attribute:
                from pxr import Sdf

                attribute = camera_prim.CreateAttribute(attribute_name, Sdf.ValueTypeNames.Float)
            if not attribute.Set(value):
                raise RuntimeError(f"cannot author camera optics attribute {attribute_name}")


def read_opencv_pinhole(camera_prim):
    """Return effective K and distortion from the live USD camera, or None."""
    model = camera_prim.GetAttribute("omni:lensdistortion:model")
    if not model or model.Get() != "opencvPinhole":
        return None
    def value(name):
        attr = camera_prim.GetAttribute(_PREFIX + name)
        if not attr or attr.Get() is None:
            raise RuntimeError(f"calibrated camera is missing {name}")
        return float(attr.Get())
    return (
        (value("fx"), 0.0, value("cx"), 0.0, value("fy"), value("cy"), 0.0, 0.0, 1.0),
        tuple(value(name) for name in _COEFFICIENTS),
    )
