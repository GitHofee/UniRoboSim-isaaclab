"""Backend identity and launch-profile-aware capabilities."""

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
        CapabilityDeclaration(
            CapabilityId("state.rigid_body@1"),
            FrozenMap({"frame": "environment-local-world", "pose_origin": "root-link"}),
        ),
        CapabilityDeclaration(
            CapabilityId("control.rigid_body.wrench@1"),
            FrozenMap(
                {
                    "application_point": "center-of-mass",
                    "frame": "environment-local-world",
                    "persistence": "until-overwrite-or-reset",
                }
            ),
        ),
        CapabilityDeclaration(
            CapabilityId("contact.binary@1"),
            FrozenMap({"source": "net-normal-force-threshold"}),
            limitations=("unavailable for USD-bridged rigid bodies in particle-fluid worlds",),
        ),
        CapabilityDeclaration(
            CapabilityId("contact.net_normal_force@1"),
            FrozenMap({"aggregation": "all-partners", "frame": "environment-local-world"}),
            limitations=(
                "friction, torque, impulses, contact points, and partner attribution are not included",
                "unavailable for USD-bridged rigid bodies in particle-fluid worlds",
            ),
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
        CapabilityDeclaration(
            CapabilityId("state.fluid.particles@1"),
            FrozenMap(
                {
                    "physics": "physx-pbd",
                    "representation": "particles",
                    "point_count": "fixed",
                    "state_layout": "batch-point-xyz",
                }
            ),
            limitations=(
                "dynamic particle count and surface reconstruction are unsupported",
                "particle-fluid worlds cannot contain tensor-backed articulations or deformables in this profile",
            ),
        ),
        CapabilityDeclaration(
            CapabilityId("control.fluid.particles@1"),
            FrozenMap({"frame": "environment-local-world", "modes": ["position", "velocity"]}),
            limitations=("commands synchronize through native USD particle state; force mode is unsupported",),
        ),
        CapabilityDeclaration(
            CapabilityId("debug.sink.native_overlay@1"),
            FrozenMap(
                {
                    "frame": "environment-local-world",
                    "primitives": ["point_set", "line_list"],
                    "stable_ids": True,
                }
            ),
        ),
    )
)

CAMERA_CAPABILITIES = (
    CapabilityDeclaration(
        CapabilityId("sensor.camera@1"),
        FrozenMap({"schedule": "synchronous", "pose_frame": "environment-local-world"}),
        limitations=("link attachment and asynchronous schedules are unsupported",),
    ),
    CapabilityDeclaration(
        CapabilityId("sensor.camera.rgb@1"),
        FrozenMap({"dtype": "uint8", "layout": "environment-height-width-rgb", "renderer": "isaac-rtx"}),
    ),
    CapabilityDeclaration(
        CapabilityId("sensor.camera.depth@1"),
        FrozenMap({"dtype": "float32", "unit": "metre", "no_hit": 0.0, "metric": "distance-to-camera"}),
    ),
)

DESCRIPTOR = ProviderDescriptor(
    provider_id="nvidia.isaaclab",
    display_name="UniRoboSim Isaac Lab 3.0",
    version="0.3.0a0",
    contract_version="v0alpha4",
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


def descriptor_for_config(config: object) -> ProviderDescriptor:
    """Expose camera capabilities only for a launch profile that can render them."""

    if bool(getattr(config, "enable_cameras", False)) and bool(getattr(config, "render", False)):
        capabilities = CapabilitySet((*CAPABILITIES, *CAMERA_CAPABILITIES))
        return ProviderDescriptor(
            provider_id=DESCRIPTOR.provider_id,
            display_name=DESCRIPTOR.display_name,
            version=DESCRIPTOR.version,
            contract_version=DESCRIPTOR.contract_version,
            capabilities=capabilities,
            metadata=DESCRIPTOR.metadata,
        )
    return DESCRIPTOR
