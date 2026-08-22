"""Backend identity and launch-profile-aware capabilities."""

from unirobosim import (
    CapabilityDeclaration,
    CapabilityId,
    CapabilitySet,
    FrozenMap,
    ProviderDescriptor,
)

from ._version import DISTRIBUTION_VERSION

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
            CapabilityId("asset.formats@1"),
            FrozenMap(
                {
                    "rigid_body": ["model/vnd.usd"],
                    "articulation": ["model/vnd.usd"],
                }
            ),
        ),
        CapabilityDeclaration(
            CapabilityId("asset.normalization@1"),
            FrozenMap(
                {
                    "rigid_body": {
                        "media_type": "model/vnd.usd",
                        "profile": "isaaclab.dynamic-rigid-usd@1",
                    }
                }
            ),
            limitations=(
                "rigid normalization rejects articulations and skinned meshes",
                "functional cavities require convex decomposition or SDF collision",
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
                    "render_modes": ["particles", "isosurface"],
                    "state_layout": "batch-point-xyz",
                }
            ),
            limitations=(
                "dynamic particle count is unsupported",
                "isosurface rendering is opt-in and consumes additional GPU memory and render time",
                "particle-fluid worlds bridge articulations through raw USD/Omni Physics state and cannot contain "
                "deformables in this profile",
                "articulation collisions affect particles, but particle reaction loads on articulations are "
                "unsupported",
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
                    "primitives": [
                        "point_set",
                        "line_list",
                        "coordinate_axes",
                        "text",
                        "bounding_box",
                        "trajectory",
                    ],
                    "stable_ids": True,
                    "stable_key": "layer-group-id",
                    "text_fallback": "ascii-vector-strokes-or-question-mark",
                }
            ),
        ),
        CapabilityDeclaration(
            CapabilityId("render.browser-scene@1"),
            FrozenMap(
                {
                    "drag_modes": ["kinematic"],
                    "draggable_entities": ["rigid_body"],
                    "scene_representation": "portable-proxy",
                }
            ),
            limitations=("browser visuals are portable proxies rather than streamed USD geometry",),
        ),
        CapabilityDeclaration(
            CapabilityId("planning.scene@2"),
            FrozenMap(
                {
                    "authority_thread": "synchronous",
                    "axis_convention": "right_handed_z_up",
                    "collision_authority": "composed-usd-and-physx-effective",
                    "geometry_read_limit_bytes": 64 * 1024 * 1024,
                    "representation_fallback": False,
                    "resource_layout": "catalog-pinned-v1",
                    "single_representation_per_geometry": True,
                }
            ),
            limitations=(
                "the first admitted profile includes only rigid objects and articulations plus the provider ground",
                "inline cube and sphere colliders and proven PhysX convexHull meshes are supported; any other "
                "effective collider fails planning admission",
                "collision groups, non-default filtering, authored collision margins, and persistent cross-entity "
                "constraints fail planning admission",
                "named frames are exposed only from locked planning-frame declarations",
                "planning asset provenance currently requires a local file-backed asset or a procedural primitive",
                "camera entities may coexist with planning but are not physical planning-scene entities",
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
    version=DISTRIBUTION_VERSION,
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
        anti_aliasing = str(getattr(config, "anti_aliasing", "fxaa"))
        texture_streaming = bool(getattr(config, "texture_streaming", False))
        render_on_step = bool(getattr(config, "render_on_step", True))
        fluid_render_mode = str(getattr(config, "fluid_render_mode", "particles"))
        return ProviderDescriptor(
            provider_id=DESCRIPTOR.provider_id,
            display_name=DESCRIPTOR.display_name,
            version=DESCRIPTOR.version,
            contract_version=DESCRIPTOR.contract_version,
            capabilities=capabilities,
            metadata=FrozenMap(
                {
                    **DESCRIPTOR.metadata.to_dict(),
                    "camera_anti_aliasing": anti_aliasing,
                    "camera_texture_streaming": texture_streaming,
                    "render_on_step": render_on_step,
                    "fluid_render_mode": fluid_render_mode,
                }
            ),
        )
    return DESCRIPTOR
