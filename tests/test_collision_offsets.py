"""Contact response validation is independent of planning shape inflation."""

from types import SimpleNamespace

import pytest

from unirobosim_isaaclab._collision_offsets import collision_offset_provenance
from unirobosim_isaaclab.native_protocols import NativePlanningError


def api(contact, rest, authored=(True, True)):
    def attr(value, explicit):
        return SimpleNamespace(Get=lambda: value, HasAuthoredValueOpinion=lambda: explicit)

    return SimpleNamespace(
        GetContactOffsetAttr=lambda: attr(contact, authored[0]),
        GetRestOffsetAttr=lambda: attr(rest, authored[1]),
    )


@pytest.mark.parametrize("contact,rest", [(0.0005, 0.0), (0.02, -0.01), (0.0, -0.01), (0.02, 0.01)])
def test_valid_contact_response_preserves_signed_values_and_converts_stage_units(contact, rest):
    result = collision_offset_provenance(api(contact, rest), 0.01)
    assert result["contact"]["stage_units"] == contact
    assert result["rest"]["stage_units"] == rest
    assert result["contact"]["metres"] == contact * 0.01
    assert result["rest"]["metres"] == rest * 0.01


@pytest.mark.parametrize(
    "contact,rest",
    [
        (-0.1, 0),
        (0.001, 0.001),
        (0.001, 0.002),
        (float("nan"), 0),
        (float("inf"), 0),
        (0.001, float("nan")),
        (0.001, float("inf")),
        (True, 0),
        (0.001, False),
        ("0.001", 0),
        (None, 0),
        (10**1000, 0),
        (0.001, -(10**1000)),
        (3.4028234663852886e38, 0),
    ],
)
def test_invalid_response_rejected_without_native_mutation(contact, rest):
    with pytest.raises(NativePlanningError) as exc:
        collision_offset_provenance(api(contact, rest), 1.0)
    assert exc.value.code == "collision_geometry_unsupported"


@pytest.mark.parametrize("authored", [(False, False), (True, False), (False, True), (True, True)])
def test_auto_sentinel_stays_symbolic(authored):
    result = collision_offset_provenance(api(-float("inf"), -float("inf"), authored), 1.0)
    if not any(authored):
        assert result is None
    else:
        assert result["contact"]["metres"] == result["rest"]["metres"] == "auto"


def test_one_automatic_offset_does_not_invent_unknown_pair_order():
    result = collision_offset_provenance(api(-float("inf"), 0.01), 1.0)
    assert result["contact"]["metres"] == "auto"
    assert result["rest"]["metres"] == 0.01


@pytest.mark.parametrize("units", [0.0, -1.0, float("nan"), float("inf"), True])
def test_invalid_stage_units_rejected(units):
    with pytest.raises(NativePlanningError):
        collision_offset_provenance(api(0.0005, 0), units)


def test_legacy_zero_pair_is_preserved_for_sdk_normalization():
    result = collision_offset_provenance(api(0.0, 0.0), 1.0)
    assert result["contact"]["stage_units"] == 0.0
    assert result["rest"]["stage_units"] == 0.0
