"""Static backend identity and capabilities."""

from unirobosim import (
    CapabilityDeclaration,
    CapabilityId,
    CapabilitySet,
    FrozenMap,
    ProviderDescriptor,
)

CAPABILITIES = CapabilitySet(
    (
        CapabilityDeclaration(
            CapabilityId("profile.core-robotics@1"),
            FrozenMap(
                {
                    "array_layout": "batch-first",
                    "coordinate_system": "right-handed-z-up",
                    "quaternion_order": "xyzw",
                    "units": "si",
                }
            ),
        ),
        CapabilityDeclaration(
            CapabilityId("world.multi-environment@1"),
            FrozenMap({"isolation": "physx-environment-origins"}),
        ),
        CapabilityDeclaration(CapabilityId("state.articulation@1")),
        CapabilityDeclaration(CapabilityId("control.articulation.position@1")),
        CapabilityDeclaration(CapabilityId("control.articulation.velocity@1")),
        CapabilityDeclaration(CapabilityId("control.articulation.effort@1")),
        CapabilityDeclaration(
            CapabilityId("profile.soft-matter@1"),
            FrozenMap({"physics": "physx", "state_layout": "batch-point-xyz", "point_count": "fixed"}),
            limitations=("dynamic topology and remeshing are unsupported",),
        ),
        CapabilityDeclaration(
            CapabilityId("state.deformable.surface@1"),
            FrozenMap({"topology": "triangles", "point_count": "fixed"}),
            limitations=("Isaac Lab 3.0 surface deformables expose no kinematic target buffer",),
        ),
        CapabilityDeclaration(
            CapabilityId("state.deformable.volume@1"),
            FrozenMap({"topology": "tetrahedra", "point_count": "fixed"}),
        ),
        CapabilityDeclaration(
            CapabilityId("physics.deformable.self-collision@1"),
            FrozenMap({"physics": "physx"}),
        ),
        CapabilityDeclaration(
            CapabilityId("control.deformable.points@1"),
            FrozenMap({"frame": "environment-local-world", "modes": ["position"], "topologies": ["volume"]}),
            limitations=(
                "position commands are limited to nodes declared kinematic at build time",
                "surface position, point velocity, and point force commands are unsupported",
            ),
        ),
    )
)

DESCRIPTOR = ProviderDescriptor(
    provider_id="nvidia.isaaclab",
    display_name="UniRoboSim Isaac Lab 3.0",
    version="0.1.0a0",
    contract_version="v0alpha2",
    capabilities=CAPABILITIES,
    metadata=FrozenMap(
        {
            "isaac_lab_release": "3.0.0-beta2",
            "isaaclab_distribution": "6.1.17",
            "isaaclab_physx_distribution": "1.1.3",
            "isaacsim_distribution": "6.0.1.0",
            "python": "3.12",
        }
    ),
)
