# explore_rbnx

Autonomous frontier exploration skill for robonix. Driven by the
LLM/pilot via `rbnx ask "explore"` (or any equivalent intent that
triggers the `robonix/skill/explore/explore` tool).

See `CAPABILITY.md` for the full interface + behaviour spec.

Exploration duration, first-map wait, navigation legs, sweeps and settling use
the deployment's selected `reference`. The controller's ROS node uses simulation
time for `ros_time`. Status queries, cancellation requests, failure backoff and
shutdown joins retain real limits. Reference interruption or generation change
ends the task and cancels its navigation run; old map, pose and coverage data
cannot continue in a new generation. Repeated frozen evidence does not declare
exploration complete. Quiet completion requires changed map or pose evidence.

## Quickstart

```bash
# from the package root
bash scripts/build.sh   # rbnx codegen + docker build
# then add to your deploy manifest under skill: and rbnx boot
```

Manifest snippet:

```yaml
skill:
  - name: explore
    path: ../../../explore_rbnx
    config: {}
```

The skill registers 3 MCP tools with atlas. Pilot's tool catalog
will include `robonix/skill/explore/{explore, status, cancel}`.

## Layout

```
explore_rbnx/
├── package_manifest.yaml    # 3 caps + build/start hooks
├── capabilities/            # package-local TOMLs + .srv files
│   ├── explore.v1.toml
│   ├── status.v1.toml
│   ├── cancel.v1.toml
│   └── srv/{Explore,GetExploreStatus,CancelExplore}.srv
├── explore_skill/           # python module
│   ├── atlas_bridge.py      # entrypoint: register + serve MCP
│   ├── frontier.py          # WFD + clustering + scoring + safety
│   └── controller.py        # state machine, frontier loop, sweep
├── scripts/{build,start}.sh
├── docker/{Dockerfile, entrypoint.sh, no_shm_profile.xml}
├── CAPABILITY.md            # LLM-readable spec (registered with atlas)
└── README.md
```

License: MulanPSL-2.0
