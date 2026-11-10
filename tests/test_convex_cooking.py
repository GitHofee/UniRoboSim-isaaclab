from __future__ import annotations

from types import SimpleNamespace

import pytest

from unirobosim_isaaclab._convex_cooking import (
    ConvexCookingParameters,
    apply_convex_cooking_parameters,
    convex_cooking_parameters,
)
from unirobosim_isaaclab.native_protocols import NativePlanningError


class Attribute:
    def __init__(self, value=None, *, authored=False, varying=False):
        self.value = value
        self.authored = authored
        self.varying = varying

    def Get(self):
        return self.value

    def Set(self, value):
        self.value = value
        return True

    def HasAuthoredValueOpinion(self):
        return self.authored

    def ValueMightBeTimeVarying(self):
        return self.varying


class Prim:
    def __init__(self, values=None):
        self.attrs = values or {}
        self.applied = []

    def GetAttribute(self, name):
        return self.attrs.get(name)


DEFAULTS = {'errorPercentage': 10., 'hullVertexLimit': 64, 'maxConvexHulls': 32,
            'minThickness': .001, 'shrinkWrap': False, 'voxelResolution': 500000}


def modules():
    schemas = {}
    for name in ('PhysxConvexDecompositionCollisionAPI', 'PhysxConvexHullCollisionAPI'):
        prefix = name.removeprefix('Physx').removesuffix('API')
        prefix = 'physx'+prefix
        keys = DEFAULTS if 'Decomposition' in name else ('hullVertexLimit', 'minThickness')
        defaults = {prefix+':'+key: DEFAULTS[key] for key in keys}

        class Schema:
            @staticmethod
            def GetSchemaAttributeNames(inherited, defaults=defaults):
                assert inherited is False
                return tuple(defaults)

            @staticmethod
            def Apply(prim, name=name, defaults=defaults):
                prim.applied.append(name)
                prim.attrs.update({key: Attribute(value) for key, value in defaults.items()})
                return True

        schemas[name] = Schema
    fallback = {**{key: value for key, value in DEFAULTS.items()}}
    definition = SimpleNamespace(GetAttributeFallbackValue=lambda name: fallback[name.split(':')[1]])
    return SimpleNamespace(PhysxSchema=SimpleNamespace(**schemas),
                           Usd=SimpleNamespace(SchemaRegistry=lambda: SimpleNamespace(
                               FindAppliedAPIPrimDefinition=lambda name: definition)))


@pytest.mark.parametrize('approximation', ('convexHull', 'convexDecomposition'))
def test_defaults_and_custom_values_round_trip_without_source_mutation(approximation):
    sdk = modules()
    initial = convex_cooking_parameters(sdk, Prim(), approximation)
    assert initial is not None
    assert len(initial.attributes) == (2 if approximation == 'convexHull' else 6)
    target = dict(initial.attributes)
    explicit_defaults = Prim({key: Attribute(value, authored=True) for key, value in target.items()})
    assert convex_cooking_parameters(sdk, explicit_defaults, approximation).identity == initial.identity
    key = next(k for k in target if k.endswith(':hullVertexLimit'))
    source = Prim({key: Attribute(128, authored=True)})
    custom = convex_cooking_parameters(sdk, source, approximation)
    assert custom is not None and source.applied == []
    assert custom.identity != initial.identity
    assert custom.identity == convex_cooking_parameters(sdk, source, approximation).identity
    assert dict(custom.attributes)[key] == 128
    temporary = Prim()
    apply_convex_cooking_parameters(sdk, temporary, custom)
    assert temporary.applied == [custom.schema]
    assert {k: a.Get() for k, a in temporary.attrs.items()} == dict(custom.attributes)


@pytest.mark.parametrize('attr', [Attribute(float('nan')), Attribute(float('inf')),
                                 Attribute(None, authored=True), Attribute(4, varying=True), Attribute('bad')])
def test_unsupported_parameters_fail_closed(attr):
    with pytest.raises(NativePlanningError, match='collision_geometry_unsupported'):
        convex_cooking_parameters(modules(), Prim({'physxConvexHullCollision:hullVertexLimit': attr}), 'convexHull')


def test_nonconvex_approximation_untouched():
    assert convex_cooking_parameters(None, None, 'none') is None


def test_cooking_cache_separates_parameters_and_reuses_identical_inputs():
    from unirobosim_isaaclab.native_planning import _PlanningAdmission

    admission = object.__new__(_PlanningAdmission)
    admission._convex_cache = {}
    calls = []
    component = ((b'cached-component', 1, 1),)
    admission._persistent_mesh_cache = SimpleNamespace(get=lambda key: calls.append(key) or component)
    mesh = SimpleNamespace(source_sha256='same-mesh')
    first = ConvexCookingParameters('PhysxConvexHullCollisionAPI', (('hullVertexLimit', 64),))
    second = ConvexCookingParameters('PhysxConvexHullCollisionAPI', (('hullVertexLimit', 128),))
    assert admission._cook_convex_components(mesh, 'convexHull', first) == component
    assert admission._cook_convex_components(mesh, 'convexHull', first) == component
    assert len(calls) == 1
    assert admission._cook_convex_components(mesh, 'convexHull', second) == component
    assert len(calls) == len(set(calls)) == 2
