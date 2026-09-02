"""Backend identity and launch-profile-aware capabilities."""

from unirobosim import (
    CHECKPOINT_CAPABILITY_ID,
    COMPOSITE_WORLD_SCHEMA_VERSION,
    PHYSICAL_WORLD_SCHEMA_VERSION,
    RENDER_STATE_CAPABILITY_ID,
    WORLD_SCHEMA_VERSION,
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
                    "composite_scene": ["model/vnd.usd"],
                    "static_scene": ["model/vnd.usd"],
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
            CHECKPOINT_CAPABILITY_ID,
            FrozenMap(
                {
                    "atomic_scope": "world",
                    "clock_rewind": False,
                    "fidelity": "physical",
                    "payload_schema": "nvidia.isaaclab.physical-checkpoint/1",
                }
            ),
        ),
        CapabilityDeclaration(
            CapabilityId("scene.static@1"),
            FrozenMap(
                {
                    "asset_format": "model/vnd.usd",
                    "motion": "static",
                    "replication": "per-environment",
                }
            ),
            limitations=(
                "static-scene assets containing rigid bodies or articulations are rejected",
                "static-scene state and drag commands are unsupported",
            ),
        ),
        CapabilityDeclaration(
            CapabilityId("scene.composite@1"),
            FrozenMap(
                {
                    "asset_format": "model/vnd.usd",
                    "composition": "reference-once-per-environment",
                    "motion": "mixed-physics",
                    "reset": "contained-rigid-and-articulation-state",
                }
            ),
            limitations=(
                "composite containers do not accept commands directly",
                "only explicitly declared embedded entities expose state and control",
            ),
        ),
        CapabilityDeclaration(
            CapabilityId("scene.composite.unbound-rigid-mode@1"),
            FrozenMap(
                {
                    "metadata_key": "composite_unbound_rigid_mode",
                    "modes": ["authored", "kinematic", "static"],
                    "default": "authored",
                    "authoring": "current-stage-override-before-first-reset",
                    "unbound_definition": "rigid-body-not-owned-by-fastsim-embedded-binding",
                    "embedded_link_protection": "exact-link-or-nearest-rigid-ancestor-within-composite",
                    "private_joint_bodies": {
                        "kinematic": "kinematic",
                        "static": "static-collider",
                    },
                    "embedded_joint_protection": "exact-relative-prim-path",
                    "private_joint_prims": "disabled-in-current-stage-before-first-reset",
                    "joint_prims": "schemas-preserved",
                    "write_order": "disable-private-joints-then-freeze-unbound-bodies",
                    "static_authoring": "remove-unbound-rigid-body-api-preserve-collision",
                    "collision": {"authored": "preserved", "kinematic": "preserved", "static": "preserved"},
                    "multi_environment": "identical-relative-selection-required",
                }
            ),
        ),
        CapabilityDeclaration(
            CapabilityId("entity.embedded-binding@1"),
            FrozenMap(
                {
                    "admission": "build-time",
                    "composition": "no-respawn",
                    "path_authority": "container-relative-prim-path",
                }
            ),
        ),
        CapabilityDeclaration(
            CapabilityId("entity.scale.rigid@1"),
            FrozenMap({"axes": "xyz", "semantics": "physical"}),
        ),
        CapabilityDeclaration(
            CapabilityId("entity.scale.articulation.uniform@1"),
            FrozenMap({"axes": "uniform-xyz", "semantics": "physical"}),
        ),
        CapabilityDeclaration(
            CapabilityId("entity.scale.static_scene@1"),
            FrozenMap({"axes": "xyz", "semantics": "physical"}),
        ),
        CapabilityDeclaration(
            CapabilityId("entity.scale.composite_scene@1"),
            FrozenMap(
                {
                    "axes": "xyz",
                    "authoring": "usd-file-config-scale",
                    "semantics": "native-effective-physical",
                    "planning_geometry": "native-effective-inventory",
                    "invalid_asset_behavior": "fail-closed",
                }
            ),
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
        CapabilityDeclaration(
            RENDER_STATE_CAPABILITY_ID,
            FrozenMap(
                {
                    "atomic_scope": "frame",
                    "fluid_payloads": ["array-value", "packed-float32-le"],
                    "physics_advance": False,
                    "render_invalidation": "once-per-frame",
                    "state_kinds": [
                        "articulation-joints",
                        "articulation-root",
                        "rigid-root",
                        "particle-fluid-range",
                    ],
                }
            ),
        ),
        CapabilityDeclaration(
            CapabilityId("state.kinematics.selected@1"),
            FrozenMap(
                {
                    "complexity": "selected-targets-only",
                    "frame": "environment-local-world",
                    "targets": ["articulation-root", "articulation-link"],
                    "geometry_materialization": False,
                }
            ),
        ),
        CapabilityDeclaration(
            CapabilityId("state.articulation.axis-units@1"),
            FrozenMap({"position_units": ["rad", "m"], "velocity_units": ["rad/s", "m/s"]}),
        ),
        CapabilityDeclaration(CapabilityId("control.articulation.position@1")),
        CapabilityDeclaration(
            CapabilityId("control.articulation.position.axis-units@1"),
            FrozenMap({"target_units": ["rad", "m"]}),
        ),
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
        CapabilityDeclaration(CapabilityId("scene.snapshot@1")),
        CapabilityDeclaration(CapabilityId("scene.delta@1")),
        CapabilityDeclaration(CapabilityId("scene.command.pose@1")),
        CapabilityDeclaration(
            CapabilityId("scene.command.attachment@1"),
            FrozenMap(
                {
                    "constraint": "fixed_6dof",
                    "child_entity_kinds": ("rigid_body",),
                    "parent_entity_kinds": ("rigid_body", "articulation"),
                    "preserve_current_relative_pose": True,
                    "reset_clears": True,
                }
            ),
        ),
        CapabilityDeclaration(
            CapabilityId("scene.command.drag@1"),
            FrozenMap({"entity_kinds": ["rigid_body"], "modes": ["kinematic"]}),
            limitations=("constraint drag is not exposed by the adapter",),
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
                "static-scene entities fail planning admission until their complete collider forest can be published",
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

DEBUG_RENDER_CAPABILITIES = (
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
                    "mesh_instance",
                ],
                "mesh_styles": ["solid", "wireframe", "solid_with_edges"],
                "mesh_topology_cache": "immutable-resource-id",
                "stable_ids": True,
                "stable_key": "layer-group-id",
                "text_fallback": "ascii-vector-strokes-or-question-mark",
                "offscreen": True,
            }
        ),
        limitations=("requires a render-capable launch profile; headless-physics is unsupported",),
    ),
)

CAMERA_CAPABILITIES = (
    CapabilityDeclaration(
        CapabilityId("sensor.camera@1"),
        FrozenMap(
            {
                "mount_parent_kinds": ["articulation"],
                "pose_frames": ["environment-local-world", "parent-local"],
                "schedule": "synchronous",
            }
        ),
        limitations=("asynchronous schedules are unsupported",),
    ),
    CapabilityDeclaration(
        CapabilityId("sensor.camera.rgb@1"),
        FrozenMap({"dtype": "uint8", "layout": "environment-height-width-rgb", "renderer": "isaac-rtx"}),
    ),
    CapabilityDeclaration(
        CapabilityId("sensor.camera.depth@1"),
        FrozenMap({"dtype": "float32", "unit": "metre", "no_hit": 0.0, "metric": "distance-to-camera"}),
    ),
    CapabilityDeclaration(
        CapabilityId("sensor.camera.normals@1"),
        FrozenMap({"dtype": "float32", "frame": "camera-local", "layout": "environment-height-width-xyz"}),
    ),
)

DESCRIPTOR = ProviderDescriptor(
    provider_id="nvidia.isaaclab",
    display_name="UniRoboSim Isaac Lab 3.0",
    version=DISTRIBUTION_VERSION,
    contract_version="v0alpha6",
    capabilities=CAPABILITIES,
    supported_world_schema_versions=(
        WORLD_SCHEMA_VERSION,
        PHYSICAL_WORLD_SCHEMA_VERSION,
        COMPOSITE_WORLD_SCHEMA_VERSION,
    ),
    metadata=FrozenMap(
        {
            "isaac_lab_release": "3.0.0-beta2",
            "isaaclab_distribution": "6.1.17",
            "isaaclab_physx_distribution": "1.1.3",
            "isaacsim_distribution": "6.0.1.0",
            "python": "3.12",
            "runtime_profiles": [
                {
                    "id": "source-isaaclab-3.0.0-beta2",
                    "isaaclab_distribution": "6.1.17",
                    "isaacsim_distribution": "6.0.1.0",
                    "torch_distribution": "2.11.0",
                },
                {
                    "id": "ngc-isaaclab-3.0.0",
                    "isaaclab_release": "3.0.0",
                    "isaaclab_distribution": "6.1.11",
                    "isaacsim_release": "6.0.1",
                    "torch_distribution": "2.10.0",
                },
            ],
        }
    ),
)


def descriptor_for_config(config: object) -> ProviderDescriptor:
    """Expose render-backed capabilities only for profiles that can provide them."""

    render = bool(getattr(config, "render", False))
    enable_cameras = bool(getattr(config, "enable_cameras", False))
    extras = (*DEBUG_RENDER_CAPABILITIES,) if render else ()
    if render and enable_cameras:
        extras = (*extras, *CAMERA_CAPABILITIES)
    if extras:
        capabilities = CapabilitySet((*CAPABILITIES, *extras))
        anti_aliasing = str(getattr(config, "anti_aliasing", "fxaa"))
        texture_streaming = bool(getattr(config, "texture_streaming", False))
        render_on_step = bool(getattr(config, "render_on_step", True))
        max_render_hz = getattr(config, "max_render_hz", None)
        fluid_render_mode = str(getattr(config, "fluid_render_mode", "particles"))
        return ProviderDescriptor(
            provider_id=DESCRIPTOR.provider_id,
            display_name=DESCRIPTOR.display_name,
            version=DESCRIPTOR.version,
            contract_version=DESCRIPTOR.contract_version,
            capabilities=capabilities,
            supported_world_schema_versions=DESCRIPTOR.supported_world_schema_versions,
            metadata=FrozenMap(
                {
                    **DESCRIPTOR.metadata.to_dict(),
                    "camera_anti_aliasing": anti_aliasing,
                    "camera_texture_streaming": texture_streaming,
                    "render_on_step": render_on_step,
                    "max_render_hz": max_render_hz,
                    "fluid_render_mode": fluid_render_mode,
                }
            ),
        )
    return DESCRIPTOR
