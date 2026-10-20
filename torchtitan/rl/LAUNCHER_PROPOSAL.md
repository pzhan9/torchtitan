# Proposal: scheduler launchers for TorchTitan RL, starting with SLURM

Status: proposal for review. No code in this PR.

This document proposes how TorchTitan RL should launch jobs on cluster
schedulers. The immediate goal is to upstream FairTitan's SLURM launcher. The
design is a small scheduler-neutral launcher interface on the Monarch actor
system, and SLURM is its first implementation. A second scheduler integration
already exists outside this repo and follows the same pattern. Kubernetes
(Monarch's `KubernetesJob`) is a likely third.

prime-rl's SLURM launcher is the reference design, because prime-rl has run
multi-node RL on SLURM for a long time and has settled many of the same
questions. Once reviewers agree on the structure, this document is the handoff
for the implementation PRs: Sections 4 to 7 are the spec, and Section 9 splits
the work into PRs.

Code referenced here (all permalinks):

| Repo | Snapshot | What to read |
| --- | --- | --- |
| FairTitan (`fairinternal/fairtitan`) | [`5f91286`](https://github.com/fairinternal/fairtitan/tree/5f912863b93f02454a4a096999b735346d28910c/launcher/rl) | `launcher/rl/`: the SLURM launcher being upstreamed |
| prime-rl (`PrimeIntellect-ai/prime-rl`) | [`9546001`](https://github.com/PrimeIntellect-ai/prime-rl/tree/9546001f7361f580836445f254ee958f8c41e4fc) | `src/prime_rl/entrypoints/rl.py`, `src/prime_rl/templates/`, `packages/prime-rl-configs/src/prime_rl/configs/` |
| TorchTitan | `3b1714ab0` (main, 2026-10-20) | `torchtitan/rl/actors/`, `torchtitan/rl/train.py`, `torchtitan/config/loader.py` |

---

## 1. Summary

- **Recommended:** add one config field, `launcher: MonarchLauncher.Config | None = None`, to
  `MonarchActorSystem.Config`. None means run on this host, as today. The
  scheduler is chosen by the config's type (`SlurmLauncher.Config`, later
  others), the same pattern `Controller.Config.actor_system` already uses.
- **One base class, `MonarchLauncher`, with two scheduler-specific steps:**
  - on the submit host: `submit(plan)` asks the scheduler for the hosts;
  - inside the job: `acquire_hosts(layout)` attaches to them and returns
    Monarch host meshes.

  Smaller hooks cover GPUs per node, how the controller detects it is inside
  the job, and releasing hosts at the end.
- **Shared by every scheduler, in this repo:**
  - the layout planner (world sizes to nodes per role);
  - role naming and job naming;
  - the submit command with `--dry-run`;
  - launch artifacts;
  - the env layers;
  - the `provision` and `stop` integration.

  `python -m torchtitan.rl.train` is the in-job controller command for every
  scheduler. It stays unchanged.
- **First implementation: `SlurmLauncher`.** It submits one Monarch `SlurmJob`
  in batch mode, so the controller runs inside the allocation.
- **Taken from prime-rl:**
  - an optional scheduler config block that also switches the run off this
    host (`slurm: SlurmConfig | None`, "If None, runs locally");
  - scheduler fields default to the cluster's defaults, and every
    site-specific value comes from the user's config;
  - environment variables merged in layers;
  - JIT caches on node-local storage;
  - `--dry-run`;
  - launch artifacts under the run directory.
- **Not taken from prime-rl:** its Jinja sbatch templates and the bash process
  orchestration inside them. Monarch's job runners and mesh supervision already
  handle placement, rendezvous, teardown, and failure propagation.
- **Stays outside this repo:**
  - site data (partitions, accounts, NICs, W&B endpoints, dataset paths);
  - environment installers;
  - non-public scheduler implementations, which subclass `MonarchLauncher`
    from their own packages.
- **Needs upstream fixes:** a few FairTitan workarounds are bugs in Monarch or
  TorchStore (Section 8). Fix them there rather than copying the workarounds
  into this repo.

---

## 2. Background

### 2.1 What FairTitan has today

[`launcher/rl/`](https://github.com/fairinternal/fairtitan/tree/5f912863b93f02454a4a096999b735346d28910c/launcher/rl) (~1.4k lines including tests) has three layers:

| Layer | Files | Role |
| --- | --- | --- |
| Shell wrapper | [`submit.sh`](https://github.com/fairinternal/fairtitan/blob/5f912863b93f02454a4a096999b735346d28910c/launcher/rl/submit.sh) | Activates the venv and sets `PYTHONPATH`, `CUDA_HOME`, NIXL's `UCX_MODULE_DIR`, and Triton/Inductor caches in `/tmp`. Contains the `HOSTNAME` workaround (L58) and moves a stale `.monarch/job_state.pkl` aside (L100). Then execs the login-node launcher. |
| Login-node launcher | [`launcher.py`](https://github.com/fairinternal/fairtitan/blob/5f912863b93f02454a4a096999b735346d28910c/launcher/rl/launcher.py), [`slurm.py`](https://github.com/fairinternal/fairtitan/blob/5f912863b93f02454a4a096999b735346d28910c/launcher/rl/slurm.py) | Parses the config. `SlurmLauncher.plan` (L122) turns world sizes into full-node meshes `{"trainer": n, "generator_i": m}`. Prints the plan (`--dry-run` stops here). Applies the environment to `os.environ` (L171). Builds a Monarch `SlurmJob` (L184; `exclusive=True`, `bind_to="::"` at L230) and calls `apply(client_script="python -m launcher.rl.main ...")`. |
| In-allocation controller | [`main.py`](https://github.com/fairinternal/fairtitan/blob/5f912863b93f02454a4a096999b735346d28910c/launcher/rl/main.py) | `job_load().state()` produces `HostMeshes` (`slurm.py` L243), then it runs the controller on them. |
| Config and site data | [`config_registry.py`](https://github.com/fairinternal/fairtitan/blob/5f912863b93f02454a4a096999b735346d28910c/launcher/rl/config_registry.py), [`env_defaults.py`](https://github.com/fairinternal/fairtitan/blob/5f912863b93f02454a4a096999b735346d28910c/launcher/rl/env_defaults.py), [`slurm_bootstrap/sitecustomize.py`](https://github.com/fairinternal/fairtitan/blob/5f912863b93f02454a4a096999b735346d28910c/launcher/rl/slurm_bootstrap/sitecustomize.py) | `SlurmRLConfig(Controller.Config)` adds a `launcher` field (L50). The FAIR recipes are wrapped by `_with_launcher` (L56). FAIR defaults: `partition="g3"`, `qos="g3_lowest"`, `account="faircw-pytorch-access"`, GB300 UCX NIC names, the FAIR W&B URL, HF offline, and proxy unsets. Per-job cache directories are set from a `sitecustomize`. |

The core design is sound and Monarch-native: one allocation, one batch job, and
the controller runs inside it. But it was written against an older TorchTitan
API and does not run on main as-is:

1. **CLI.** It uses `ConfigManager` (Tyro) and per-field overrides such as
   `--launcher.time-limit=00:30:00`. #4908 replaced Tyro with `ConfigLoader`,
   which deliberately exposes only `--module`, `--config`, `--override`,
   `--comm-backend`, `--output-dir`, `--resume-step` and `--print-config`
   ([`torchtitan/config/README.md`](../config/README.md)).
2. **Host placement hook.** It imports `HostMeshes` and `spawn_proc_mesh` from
   `torchtitan.rl.train` and calls
   `setup_async(trainer_mesh=..., generator_meshes=...)`. #5391 moved both into
   [`actors/monarch/provisioning.py`](actors/monarch/provisioning.py) behind
   the `ActorSystem` interface. A launcher now supplies hosts with
   `MonarchActorSystem(config, host_meshes=...)` and
   `Controller.setup_async(actor_system=...)`
   ([`actors/monarch/README.md`](actors/monarch/README.md)).
3. **Monarch import boundary.** "No module outside this package imports Monarch
   or this package" ([`actors/monarch/README.md`](actors/monarch/README.md)).
   FairTitan's `slurm.py` imports `monarch.job` directly, so the upstreamed
   modules have to live inside `torchtitan/rl/actors/monarch/`.

### 2.2 How prime-rl does it

prime-rl has no separate SLURM entrypoint. The `rl` launcher switches on
whether `config.slurm` is set
([`entrypoints/rl.py` L527](https://github.com/PrimeIntellect-ai/prime-rl/blob/9546001f7361f580836445f254ee958f8c41e4fc/src/prime_rl/entrypoints/rl.py#L527)).
`rl_slurm` (L419) writes the resolved configs, renders a Jinja sbatch template
into `<run_dir>/launcher/rl.sbatch` (`write_slurm_script`, L264), and either
stops (`--dry-run`) or runs `sbatch`.

Its config is split into two blocks:

- **`deployment`: topology, independent of the cluster.**
  [`configs/rl.py` L105-155](https://github.com/PrimeIntellect-ai/prime-rl/blob/9546001f7361f580836445f254ee958f8c41e4fc/packages/prime-rl-configs/src/prime_rl/configs/rl.py#L105-L155).
  A discriminated union of `single_node` (`num_train_gpus`, `num_infer_gpus`)
  and `multi_node` (`num_train_nodes`, `num_infer_nodes`,
  `num_infer_replicas`, `nodes_per_fsdp_group`, `orchestrator_on_inference`),
  plus `gpus_per_node`.
- **`slurm`: the scheduler and cluster.**
  [`configs/shared.py` L133-189](https://github.com/PrimeIntellect-ai/prime-rl/blob/9546001f7361f580836445f254ee958f8c41e4fc/packages/prime-rl-configs/src/prime_rl/configs/shared.py#L133-L189).
  `job_name`, `partition`, `account`, `time`, `nodelist`, `exclude`,
  `project_dir` (sources `.env` and activates `.venv`), `pre_run_command`,
  `template_path` (escape hatch for custom clusters), `shared_fs`, and
  `cleanup_grace_period`. Every scheduler field defaults to None, meaning the
  cluster default. Nothing site-specific is hardcoded.

prime-rl supports only SLURM, so it has no scheduler abstraction to borrow. Its
topology / scheduler split is still the useful idea: the scheduler-neutral half
of this design (Section 4.3) corresponds to `deployment`, and the
scheduler-specific half (Section 4.5) corresponds to `slurm`.

Other relevant behavior:

- **Bundled templates**
  ([`templates/`](https://github.com/PrimeIntellect-ai/prime-rl/tree/9546001f7361f580836445f254ee958f8c41e4fc/src/prime_rl/templates)),
  chosen by deployment type (`auto_setup_slurm_template`, `configs/rl.py`
  L754). `single_node_rl.sbatch.j2` re-runs `rl` inside the allocation.
  `multi_node_rl.sbatch.j2` (~630 lines) does all the process orchestration in
  bash:
  - assigns roles by `SLURM_PROCID`;
  - wires `MASTER_ADDR` and routers;
  - cleans up stale processes (L196-223);
  - runs `srun --kill-on-bad-exit=1` (L235);
  - uses "done" files so a clean exit stops every node (L71-76, L589-628);
  - waits for an in-flight checkpoint before teardown (`cleanup_grace_period`,
    L603).
- **Env layering**
  ([`utils/process.py` L20-47](https://github.com/PrimeIntellect-ai/prime-rl/blob/9546001f7361f580836445f254ee958f8c41e4fc/src/prime_rl/utils/process.py#L20-L47)
  and `write_slurm_script`):
  `DEFAULT_COMMON_ENV_VARS` < per-role defaults < `config.env_vars` <
  `config.<role>.env_vars`. The defaults put Triton, vLLM and FlashInfer caches
  in node-local, per-uid `/tmp` directories, because shared-FS caches "hang or
  crash". Fabric details such as `NCCL_IB_HCA` are detected on the node (L251)
  instead of being hardcoded.
- **Launch artifacts.** The user command, the resolved config(s) and the
  generated script go under `<run_dir>/launcher/`, and SLURM logs go to
  `launcher/logs/`.
- **Docs**
  ([`docs/scaling.md` SLURM section](https://github.com/PrimeIntellect-ai/prime-rl/blob/9546001f7361f580836445f254ee958f8c41e4fc/docs/scaling.md#slurm)):
  a SLURM config is "a thin overlay" on a base config. The checkout and venv
  must be on a shared filesystem. For unusual clusters, copy a bundled
  template.

### 2.3 What a second scheduler requires

A second scheduler integration for TorchTitan RL already exists outside this
repo. It is not open source. It has the same structure as FairTitan's
launcher:

- a `Controller.Config` subclass with a `launcher` field;
- a submit step that turns world sizes into nodes per role, generates a unique
  job name, and forwards the `--module/--config` arguments;
- an in-job controller that re-parses the same config, attaches to the job's
  hosts, builds `HostMeshes`, and runs the controller.

About two thirds of each launcher is duplicated logic. Where the second
integration differs from SLURM batch mode, the base interface must allow for
it:

| Concern | SLURM batch mode | The second scheduler |
| --- | --- | --- |
| Where the controller runs | On the allocation's head node, next to GPU procs | On a dedicated controller host (its own role in the job) |
| How the controller finds its job | Monarch's batch job state plus `MONARCH_BATCH_JOB=1` | A job handle passed to the controller |
| Who ends the worker hosts | Monarch's batch runner kills the `srun` workers | The controller must shut down the host meshes it attached to |
| How code reaches the nodes | Shared filesystem; workers run the submitting interpreter | Packaged with the job; a wrapper script prepares each host |
| How env reaches the nodes | SLURM's `--export=ALL` from the submit host | Set per role in the job spec |
| GPUs per node | A config value | Derived from the host type, with an override |
| Sub-node roles | The draft allowed a shared single node | One role per host; a role smaller than a host still gets a whole host |
| Role / mesh names | `generator_0`, `generator_1`, ... | `generator` for one, `generator_<i>` for several |
| Monarch transport | TCP, with `bind_to` for dual-stack clusters | Scheduler-specific (TLS over IPv6) |

Each launcher also sets some per-role values that depend only on the
parallelism config, not on the scheduler. For example, HybridEP's NVLink domain
size is `min(ep_degree, gpus_per_node)` for each role. These belong in the
actor system, so every launcher gets them (Section 4.4).

---

## 3. Decisions

| Choice | Adopt? | TorchTitan version |
| --- | --- | --- |
| prime-rl: optional scheduler block; None runs locally | **Yes** | `MonarchActorSystem.Config.launcher: MonarchLauncher.Config \| None = None`; the scheduler is chosen by config type (Section 4.1) |
| Support more than one scheduler | **Yes, now** | Two launchers with the same shape already exist (Section 2.3). One base class, shared planner and shared submit command; SLURM is the first implementation. |
| prime-rl: scheduler fields default to None (cluster default); nothing site-specific in the framework | **Yes** | `SlurmLauncher` defaults `partition`, `account`, `qos` and `time` to None. FAIR values move to a FairTitan profile (Section 6). |
| prime-rl: separate `deployment` block for topology | **No; derive the layout** | Recipes already fix world sizes through `trainer.parallelism`, `generator.parallelism` and `num_generators`. A second topology block would be a second source of truth. The only topology input is the launcher's GPUs per node (Section 4.3). |
| prime-rl: single-node and multi-node layouts | **Yes, one planner for every scheduler** | Shared single node when everything fits; otherwise whole nodes per role (Section 4.3). |
| prime-rl: same command submits and runs | **Partly** | A separate, scheduler-neutral submit command with the same arguments as `torchtitan.rl.train`. In the job, the unchanged `torchtitan.rl.train` runs (Section 4.2). |
| prime-rl: Jinja templates and `template_path` | **No** | Monarch's job classes write the job spec. SLURM's escape hatch is `sbatch_args` plus a site `python_exe` wrapper. A setup-command hook depends on Monarch (Section 8). |
| prime-rl: bash process orchestration | **No** | Monarch's job runners and mesh supervision, plus `release_hosts` for schedulers whose controller owns the hosts (Section 5) |
| prime-rl: `cleanup_grace_period` | **Later** | Section 10 |
| prime-rl: env layering; node-local caches by default | **Yes** | Section 4.6 |
| prime-rl: `--dry-run`; artifacts under the run directory | **Yes** | Section 4.7 |

---

## 4. Design

### 4.1 Launcher interface and config

The launcher config goes on the Monarch actor system's config, not on
`Controller.Config`:

- Deciding where services run is the actor system's job. The `ActorSystem`
  docstring says it "owns where they run (`provision`)".
- This keeps Monarch and scheduler code inside `actors/monarch/`.
- Another actor framework (e.g. Ray) would bring its own cluster integration
  with its own `ActorSystem`.
- It replaces the `Controller.Config` subclasses both existing launchers use
  (`SlurmRLConfig` and its counterpart), so recipes keep returning a plain
  `Controller.Config`.

```python
# torchtitan/rl/actors/monarch/launcher.py  (new; scheduler-neutral)

@dataclass(frozen=True, slots=True)
class Layout:
    """Where the trainer and generators run, derived from world sizes (Section 4.3)."""

    gpus_per_node: int
    mesh_nodes: dict[str, int]
    """Host mesh name -> node count, e.g. {"trainer": 2, "generator_0": 1}."""
    role_mesh: dict[str, str]
    """Role name -> host mesh name. Several roles may share one mesh."""

    def host_meshes(self, meshes: dict[str, HostMesh]) -> HostMeshes:
        """Map acquired meshes to roles, passing a shared mesh as the same object."""


@dataclass(frozen=True, slots=True)
class LaunchPlan:
    """Everything resolved on the submit host; printed by --dry-run."""

    job_name: str                    # unique: <job_name>-<6 hex chars>, sanitized
    layout: Layout
    controller_args: list[str]       # forwarded to `python -m torchtitan.rl.train`
    env: dict[str, str]              # merged env (Section 4.6)
    dump_folder: str


class MonarchLauncher(Configurable, abc.ABC):
    """Gets hosts for an RL run from a scheduler, as Monarch host meshes."""

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        job_name: str = "torchtitan-rl"
        env: dict[str, str] = field(default_factory=dict)
        share_single_node: bool = True
        """Put every role on one shared node when they fit (Section 4.3)."""

    # Shared, concrete
    def plan(self, config: Controller.Config, args: list[str]) -> LaunchPlan: ...
    def layout(self, trainer_world_size: int, generator_world_size: int,
               num_generators: int) -> Layout: ...

    # Scheduler-specific
    @abc.abstractmethod
    def gpus_per_node(self) -> int: ...
    @abc.abstractmethod
    def submit(self, plan: LaunchPlan) -> str:
        """Submit the job from the submit host; return a job handle."""
    @abc.abstractmethod
    def in_job(self) -> bool:
        """Whether this process is the controller inside a submitted job."""
    @abc.abstractmethod
    async def acquire_hosts(self, layout: Layout) -> dict[str, HostMesh]:
        """Inside the job: attach to the job's hosts, keyed by layout mesh name."""
    async def release_hosts(self, meshes: dict[str, HostMesh]) -> None:
        """Called from MonarchActorSystem.stop. Default: no-op."""
```

```python
# torchtitan/rl/actors/monarch/system.py  (changed)

class MonarchActorSystem(ActorSystem):
    @dataclass(kw_only=True, slots=True)
    class Config(ActorSystem.Config):
        launcher: MonarchLauncher.Config | None = None
        """Scheduler that provides the hosts, chosen by config type, e.g.
        SlurmLauncher.Config(...). None runs every role on this host."""
```

Notes for the implementer:

- **Implementations outside this repo** subclass `MonarchLauncher` and its
  `Config` in their own packages, and recipes set them on
  `config.actor_system.launcher`. Nothing in this repo imports them, and the
  base class must not import any scheduler client library. Only `monarch` is
  allowed.
- **`plan` builds the job name.** It generates the unique job name
  (`<job_name>-<uuid4().hex[:6]>`, sanitized to `[A-Za-z0-9_-]`). Schedulers
  with a name-length limit truncate the prefix in `submit`, never the suffix.
- **Extending env and lists.** `env` and the list fields are lists or dicts so
  recipes and site profiles can extend them. Mind the shallow-copy rule in
  `AGENTS.md` when a profile function returns shared defaults.
- **Changing controller arguments.** A launcher that must change them (for
  example, forcing `--output-dir` onto storage that every host can reach) does
  so in `submit` by editing `plan.controller_args`. The dry run prints the
  arguments before that change, so such a launcher should also log the final
  command.

### 4.2 Shared control flow

```
submit host                                       inside the job (controller process)
-----------                                       -----------------------------------
python -m torchtitan.rl.actors.monarch.submit     python -m torchtitan.rl.train --module M --config C [...]
    [--dry-run] --module M --config C [...]         Controller.setup_async()
  config = ConfigLoader().load(args)                  MonarchActorSystem.provision()
  launcher = config.actor_system.launcher.build()       launcher set and no host_meshes passed:
  plan = launcher.plan(config, args)                      require launcher.in_job()
  print plan; write artifacts (Section 4.7)               layout = launcher.layout(world sizes)
  if --dry-run: exit                                      meshes = await launcher.acquire_hosts(layout)
  handle = launcher.submit(plan)                          spawn_proc_mesh(host_meshes=layout.host_meshes(meshes))
  print handle                                        ... training ...
                                                    MonarchActorSystem.stop()
                                                      stop proc meshes; await launcher.release_hosts(meshes)
```

- **`torchtitan.rl.train` stays unchanged.** It is the controller command for
  every scheduler. Neither FairTitan's `main.py` nor the second integration's
  controller entry point is needed any more.
- **Layout is computed twice, the same way.** `provision` already receives the
  trainer and generator parallelism and `num_generators`, so it recomputes the
  layout from the same config the submit step used. Both existing launchers
  already re-derive it inside the job. Making `layout` a pure function of
  world sizes and `gpus_per_node()` means the two sides cannot disagree.
- **The existing hook is unchanged.** If `host_meshes` is passed to the
  `MonarchActorSystem` constructor, `provision` uses it and ignores the
  launcher.
- **Guard against running outside the job.** If a launcher is set and
  `in_job()` is false, `provision` raises a `ValueError` that names the submit
  command. Running a scheduler recipe directly with `torchtitan.rl.train` then
  fails clearly instead of spawning on the submit host.
- **Why not a re-entrant `train.py`, as in prime-rl?** prime-rl's `rl` command
  re-enters only in its single-node template. Its multi-node template runs
  separate per-component commands. A re-entrant `train.py` would need to tell
  "submit" apart from "run". A separate submit command with identical
  arguments keeps `train.py` free of scheduler code. Users get prime-rl's real
  benefit either way: the recipe alone decides local or cluster, and the
  arguments are the same.
- **Entrypoint name.** The recommendation is
  `torchtitan/rl/actors/monarch/submit.py`, which respects the Monarch import
  boundary. A shorter `python -m torchtitan.rl.submit` would need an optional
  hook on the `ActorSystem` ABC (open question Q1).
- **CLI.** The submit command takes exactly `ConfigLoader`'s flags plus
  `--dry-run`, and forwards everything except `--dry-run` verbatim.
  Scheduler settings live in recipes, consistent with "put training behavior
  in the recipe". FairTitan's `--launcher.time-limit=...` overrides are gone
  with Tyro. Whether to add operational flags such as `--time` is open
  question Q2.
- **Why a Python module, not a shell script.** The submit step has to:
  - call the recipe function to get the launcher config;
  - derive the layout with the same function `provision` uses in the job;
  - submit through Monarch's job classes (for SLURM, `SlurmJob.apply` also
    saves the job state the controller reloads with `job_load`);
  - dispatch to the scheduler by launcher config type;
  - fail with a `ValueError` before anything is queued.

  None of this can be done in shell without duplicating the recipe or calling
  Python anyway. prime-rl works the same way (its `rl` command is Python; only
  the generated sbatch script is shell), and so does FairTitan (`submit.sh`
  sets up the environment, then runs the Python `launcher.py`).

#### Where users customize, and the `submit.py` docstring

Shell is still the right tool for environment setup that must happen before
Python starts. That setup is site-specific, so this repo ships no
`submit.sh`. Users who need one write their own thin wrapper. The three places
to customize:

| What | Where |
| --- | --- |
| Cluster and job settings: partition, account, QoS, time, GPUs per node, job env | The recipe (Python); sites share a cluster-profile function (Section 6) |
| Submit-host setup before Python starts: venv, `PYTHONPATH`, `CUDA_HOME`, install-specific library paths | An optional user `submit.sh` that ends with `exec python -m torchtitan.rl.actors.monarch.submit "$@"` |
| Compute-node setup before each worker's Python starts | An optional wrapper script set as the launcher's `python_exe` (SLURM), or the scheduler's own host-setup hook |

The split exists so that a run can be reproduced from its `--module/--config`
alone. Settings that only a wrapper script sets would not be recorded in the
recipe or in the launch artifacts.

Users run the submit command directly, so the module docstring is where they
learn this split. `submit.py` should open with a docstring along these lines:

```python
"""Submit a TorchTitan RL job to a cluster scheduler.

    python -m torchtitan.rl.actors.monarch.submit [--dry-run] \
        --module MODULE --config CONFIG [ConfigLoader options]

The recipe must set ``config.actor_system.launcher`` (for example
``MonarchActorSystem.Config(launcher=SlurmLauncher.Config(...))``). This command
loads the recipe, derives the layout, prints the launch plan, and submits the
job; ``--dry-run`` stops after printing. Inside the job the scheduler runs
``python -m torchtitan.rl.train`` with the same arguments.

Where to put customizations:

- Cluster and job settings (partition, account, time, GPUs per node, job
  environment variables) belong in the recipe, so a run is reproducible from
  its ``--module``/``--config``. Share them across recipes with a function that
  returns the launcher config.
- Setup that must happen before Python starts on the submit host (activating a
  venv, ``PYTHONPATH``, ``CUDA_HOME``, library paths for your install) belongs in
  your own wrapper script, for example::

      #!/bin/bash
      # my_submit.sh
      source /shared/envs/titan/bin/activate
      export CUDA_HOME=/shared/envs/titan/cuda
      exec python -m torchtitan.rl.actors.monarch.submit "$@"

  You do not need a wrapper if your environment is already active. Do not set
  recipe values in the wrapper: they would not be recorded with the run.
- Setup that must happen before each worker starts on the compute nodes
  belongs in a wrapper set as the launcher's ``python_exe`` (SLURM), or in the
  scheduler's own host-setup mechanism.

Run this where the scheduler is reachable (for SLURM, a login node). With
SLURM, the job runs the interpreter and code this command was started from,
so both must be on a filesystem the compute nodes share.
"""
```

The docstring is the user-facing reference for this split. The README
(Section 7) links to it instead of repeating it.

### 4.3 Layout: from world sizes to meshes

One planner in `launcher.py`, used by every scheduler. Inputs: the trainer
world size and the per-generator world size (`compute_trainer_world_size` and
`compute_generator_world_size` in
[`provisioning.py`](actors/monarch/provisioning.py)), `num_generators`, and
`gpus_per_node()`. Write `g` for GPUs per node:

| Case | `mesh_nodes` | `role_mesh` |
| --- | --- | --- |
| `share_single_node` and `trainer_ws + num_generators * gen_ws <= g` | `{"node": 1}` | every role -> `"node"`; `spawn_proc_mesh` gets the *same* object for all roles, so `_check_gpus_per_host` gives the trainer the first GPUs and each generator the next ones |
| Otherwise, per role: `ws <= g` -> 1 node; `ws % g == 0` -> `ws // g` nodes | `{"trainer": t, "generator": n}` or `{"trainer": t, "generator_0": n, ...}` | each role -> its own mesh |
| Some role has `ws > g` and `ws % g != 0` | `ValueError` naming the role, its world size, and `g` | |

- **The single-node case** matches prime-rl's `single_node` deployment. It is
  common for small models on 8-GPU nodes, for example trainer TP=4 plus
  generator TP=4 on one node.
- **Sub-node roles.** A role smaller than a node gets a whole node and leaves
  the rest idle; log a warning. This generalizes FairTitan, which rejects
  sub-node roles. It also matches the second integration, which gives such a
  role one host. A launcher that never wants the shared single-node layout
  sets `share_single_node=False`.
- **Role names** come from one helper used by the planner, by
  `acquire_hosts`, and by `MonarchActorSystem.spawn_generator_router`.
  Recommendation: adopt the convention `spawn_generator_router` already uses
  for actor names, `generator` for one generator and `generator_<i>` for
  several, so mesh names and actor names match (open question Q3). FairTitan
  always uses `generator_<i>` today and has to change either way.
- **Later.** Packing several sub-node roles onto shared nodes across a
  multi-node job is already supported by `spawn_proc_mesh` (roles can share a
  mesh object). It is still later work (Section 10).

### 4.4 Shared versus scheduler-specific

| Shared, in `actors/monarch/` | Scheduler-specific, in each `MonarchLauncher` |
| --- | --- |
| Layout planner and role names (4.3) | GPUs per node (a config value, or derived from a host type) |
| Job name generation | Submitting (`SlurmJob.apply`, or another scheduler's client) |
| `LaunchPlan`, `--dry-run`, plan printing, launch artifacts (4.7) | How the controller detects it is in the job (`in_job`) and finds its hosts (`acquire_hosts`) |
| Env layers 1, 2 and 3 (4.6) | How env reaches each host (SLURM `--export=ALL` plus `unset_env`, or per-role job-spec env), and layer 4 |
| `provision` / `stop` integration, login-host guard | Who ends the worker hosts (`release_hosts`) |
| Per-role values derived from parallelism, in the spawned-proc bootstrap (`_preimport_torch`, `_bootstrap_generator`): e.g. HybridEP's NVLink domain size, `min(ep_degree, gpus_per_node)` | Code and environment shipping (shared filesystem vs packaging), host preparation scripts |
| | Controller placement (allocation head node vs dedicated host), Monarch transport, scheduler-only placement constraints |

Per-role values derived from parallelism need the role's GPUs per node at
bootstrap. Pass them through the spawn env that `_spawn_proc_mesh` already
builds per role, rather than having each launcher compute them.

### 4.5 First implementation: `SlurmLauncher`

```python
# torchtitan/rl/actors/monarch/slurm.py  (new)

class SlurmLauncher(MonarchLauncher):
    """Submits one Monarch SlurmJob in batch mode; the controller runs in the allocation."""

    @dataclass(kw_only=True, slots=True)
    class Config(MonarchLauncher.Config):
        gpus_per_node: int = 8
        partition: str | None = None       # None -> cluster default, like prime-rl
        account: str | None = None
        qos: str | None = None
        time: str | None = None            # sbatch --time
        cpus_per_task: int | None = None
        mem: str | None = None
        nodelist: str | None = None
        exclude: str | None = None
        sbatch_args: list[str] = field(default_factory=list)  # extra "#SBATCH" flags
        log_dir: str | None = None         # None -> <dump_folder>/slurm
        unset_env: list[str] = field(default_factory=list)
        bind_to: str | None = None         # Monarch worker bind address, e.g. "::"
        python_exe: str | None = None      # None -> sys.executable
```

| Method | Behavior |
| --- | --- |
| `gpus_per_node()` | `config.gpus_per_node` |
| `submit(plan)` | Apply `plan.env` and `unset_env` to `os.environ`, then `SlurmJob(meshes=plan.layout.mesh_nodes, gpus_per_node=..., partition=..., account=..., qos=..., time_limit=..., cpus_per_task=..., mem=..., slurm_args=[nodelist, exclude, *sbatch_args], log_dir=..., job_name=plan.job_name, python_exe=..., exclusive=True, bind_to=...)`, then `.apply(client_script=shlex.join([python_exe, "-m", "torchtitan.rl.train", *plan.controller_args]))`. Return the SLURM job id. |
| `in_job()` | `monarch.job`'s `in_batch_job()` (`MONARCH_BATCH_JOB=1`, set by Monarch's batch runner) |
| `acquire_hosts(layout)` | `state = job_load().state()`, then `{name: getattr(state, name) for name in layout.mesh_nodes}` |
| `release_hosts` | Default no-op. `monarch._src.job._slurm_batch` terminates the workers when the controller exits. |

Notes for the implementer:

- Field names follow the sbatch flags (`time`, not `time_limit`), as prime-rl
  does, because SLURM users read these docs.
- Validation (`ValueError`): `gpus_per_node > 0`, positive `cpus_per_task`,
  and non-empty strings where set. Port FairTitan's checks
  (`slurm.py` L100-117), minus the FAIR-specific defaults.
- The controller, router, batcher and rollout workers run on the head node,
  next to that node's GPU procs. This matches prime-rl's default
  (orchestrator on trainer rank 0's node). FairTitan's SWE recipes raise
  `cpus_per_task` to 128 for this reason. A dedicated controller node for
  SLURM is later work (prime-rl has `orchestrator_on_inference`).

### 4.6 Environment

Both prime-rl and FairTitan found independently that per-node environment
matters. Merge it in layers, later layers winning:

1. **`RL_ENV_DEFAULTS`** ([`_env.py`](_env.py)), for all RL processes, local
   or not. Unchanged.
2. **`LAUNCHER_ENV_DEFAULTS`** (new, in `launcher.py`), shared by every
   scheduler and restricted to *generic* entries. Node-local, per-uid compile
   caches:
   - `TRITON_CACHE_DIR=/tmp/torchtitan-<uid>/triton`
   - `TORCHINDUCTOR_CACHE_DIR=/tmp/torchtitan-<uid>/inductor`
   - `VLLM_CACHE_ROOT=/tmp/torchtitan-<uid>/vllm`
   - `FLASHINFER_WORKSPACE_BASE=/tmp/torchtitan-<uid>/flashinfer`

   Also `PYTHONUNBUFFERED=1`. The evidence:
   - prime-rl `utils/process.py` L25-46 ("processes sharing ~/.triton on a
     shared FS hang or crash"; partial vLLM/FlashInfer caches "make TP ranks
     load different compiled artifacts and hang");
   - FairTitan `submit.sh` L60-65;
   - FairTitan's `sitecustomize.py` (the FlashInfer autotune cache failing
     with `ESTALE` on NFS).
3. **`launcher.env`** from the recipe or site profile. Fabric- and
   site-specific values go here: UCX devices, TorchStore transport flags, W&B
   URL, HF offline, and so on.
4. **Scheduler-specific.** For SLURM, `unset_env` is applied last.

`plan` merges layers 2 and 3 into `plan.env`. Each launcher decides how
`plan.env` reaches every host:

- **SLURM v1.** Like FairTitan (`launcher.py`: "Slurm exports the launcher's
  os.environ to the allocation by default"), `submit` applies `plan.env` to
  `os.environ`. The batch controller and every `srun` worker inherit it
  through SLURM's default `--export=ALL`. Document that clusters setting
  `SBATCH_EXPORT=NONE` are unsupported until Monarch can write explicit
  `export` lines into the script. prime-rl does write them into its script,
  which is the more robust long-term design (Section 8).
- **Login-node leaks.** `--export=ALL` also carries login-node-only variables
  (proxies, `OTEL_EXPORTER_OTLP_ENDPOINT`), which is why FairTitan has an
  unset list. The upstream default for `unset_env` is empty, and the FAIR list
  moves to the FAIR profile.
- **Schedulers with a job spec** set `plan.env` on each role directly. Values
  end up in the job spec, so recipes must not put secrets in `launcher.env`.

Values that depend on the job (anything containing `SLURM_JOB_ID` or another
scheduler's job id) cannot be computed on the submit host. Set them in the
spawned-proc bootstrap: `_preimport_torch` and `_bootstrap_generator` in
[`provisioning.py`](actors/monarch/provisioning.py) already run in every
spawned proc. That replaces FairTitan's `sitecustomize.py`, and the
`python-slurm` executable alias that exists only to keep `sitecustomize` on
the path (`submit.sh` L55-58). `apply_env_defaults` uses `setdefault`, so
values from `launcher.env` take precedence in the spawned procs.

Per-role env from config (prime-rl's `trainer.env_vars` and
`inference.env_vars`) is easy to add on `MonarchLauncher.Config` through the
existing per-role spawn env, but v1 does not need it.

### 4.7 Dry run and launch artifacts

- **`--dry-run`** resolves the config, runs every validation in `plan` (layout
  included), prints the plan, and exits without submitting. The plan lists:
  job name, layout (meshes, nodes, roles per mesh, world sizes), GPUs per
  node, dump folder, controller arguments, env keys set, and the launcher
  config fields (for SLURM: partition, QoS, account, time, log dir, Python,
  unset env). FairTitan's `_print_plan` (`launcher.py` L58-80) is the
  template.
- **On a real submit**, the shared code writes
  `<dump_folder>/launcher/submit.json` with the argv, the plan, the launcher
  config type and fields, the env names (not values, which can hold secrets),
  and the returned job handle. It also writes the resolved config through the
  existing `Controller.Config.maybe_log` / `debug.save_config_file` machinery.
  prime-rl's equivalent is `write_launch_artifacts` plus `write_config` under
  `<run_dir>/launcher/`.
- **SLURM logs.** stdout and stderr go to `log_dir`, which defaults to
  `<dump_folder>/slurm`. FairTitan does the same. prime-rl's dry run also
  writes the exact sbatch script, but Monarch's
  `SlurmJob._submit_slurm_job` builds the script in memory and pipes it to
  `sbatch`. Writing it needs a small Monarch API; until then, print the
  `SlurmJob` arguments (Section 8).

---

## 5. Failure and teardown

The contract for every launcher:

- If any role fails, the controller exits non-zero and the job releases all of
  its hosts.
- After a clean finish, the job ends, and no worker is left holding hosts.

Most of prime-rl's `multi_node_rl.sbatch.j2` exists to provide this:
`srun --kill-on-bad-exit=1`, done files, and a `wait -n` loop. In TorchTitan:

- **Failures reach the controller.** A failed mesh arrives as a supervision
  event, which shows up as `KeyboardInterrupt`. FairTitan re-raises it so the
  job records a failure (`main.py` L62-66), and `torchtitan.rl.train` already
  does the same.
- **Clean exits stay clean.** `torchtitan.rl.train` calls `os._exit(0)` only
  after a clean finish, so a Monarch shutdown abort is not reported as a
  failure.
- **Releasing hosts.** `MonarchActorSystem.stop` stops the proc meshes, then
  calls `launcher.release_hosts(meshes)` (best-effort, logged).
  - SLURM: a no-op, because `monarch._src.job._slurm_batch` terminates the
    `srun` workers in a `finally` and exits with the controller's return code.
  - Schedulers whose worker hosts idle in a bootstrap loop until told to exit:
    `release_hosts` shuts down the host meshes. Without this, a finished job
    would hold its hosts.

The implementation PR should state this contract in the docs and test it with
a fake launcher (Section 9). Later work: waiting for an in-flight checkpoint on
SIGTERM or the time limit (prime-rl's `cleanup_grace_period`), and requeue.
Stale-process cleanup (prime-rl L196-223) is not needed with exclusive nodes
and Monarch-managed process groups.

---

## 6. What stays in FairTitan

After the upstream PRs land, FairTitan's `launcher/rl/` shrinks to site data
and an environment script:

```python
# fairtitan launcher/rl/clusters.py  (sketch)
def fair_coreweave_gb300(*, job_name: str) -> SlurmLauncher.Config:
    return SlurmLauncher.Config(
        gpus_per_node=4,
        partition="g3",
        qos="g3_lowest",
        account="faircw-pytorch-access",
        job_name=job_name,
        bind_to="::",                     # dual-stack /etc/hosts vs DNS, see slurm.py L218-229
        env=dict(FAIR_ENV_DEFAULTS),      # UCX NICs, TorchStore NIXL flags, W&B URL, HF offline, ...
        unset_env=list(FAIR_ENV_UNSETS),  # proxies, OTEL endpoint
    )

# fairtitan recipes
def rl_grpo_qwen3_14b() -> Controller.Config:
    config = torchtitan_recipes.rl.alphabet_sort.rl_grpo_qwen3_14b()
    config.hf_assets_path = "/checkpoint/pytorch/shared/Qwen3-14B"
    ...
    config.actor_system = MonarchActorSystem.Config(
        launcher=fair_coreweave_gb300(job_name="torchtitan-rl-qwen3-14b")
    )
    return config
```

This is the TorchTitan version of prime-rl's "thin overlay": a Python function
replaces an overlay TOML. Also staying in FairTitan:

- `install_titan_rl.sh` (venv, nightlies, NIXL and TorchStore pins);
- the parts of `submit.sh` that depend on the install, such as `CUDA_HOME`
  from the venv and `UCX_MODULE_DIR` from the NIXL wheel. These can become a
  `python_exe` wrapper script or stay a pre-submit shell step;
- dataset staging, recipe-family `run.sh` preflights, and run-id handling.

To be deleted from FairTitan once upstream lands: `slurm.py`, `launcher.py`,
`main.py`, `SlurmRLConfig`, and `slurm_bootstrap/` (after the Section 8 fixes).

Out-of-repo scheduler integrations migrate the same way: a `MonarchLauncher`
subclass in their own package replaces their `Controller.Config` subclass,
submit script and controller entry point.

---

## 7. User-facing docs

- **[`actors/monarch/README.md`](actors/monarch/README.md):** add a "Launchers"
  section. Cover the `MonarchLauncher` contract (the methods, the layout, the
  failure contract in Section 5), and how an out-of-repo package adds a
  scheduler. Add `launcher.py`, `slurm.py` and `submit.py` to its module
  table.
- **[`torchtitan/rl/README.md`](README.md):** add a "Run on SLURM" section:
  - **Requirements.**
    - The TorchTitan checkout and venv are on a filesystem shared by all
      nodes.
    - Submit from that venv, because workers use the submitting interpreter
      (`sys.executable`).
    - SLURM exports the submit environment (the default).
  - **Writing a SLURM recipe.** A recipe function sets
    `config.actor_system = MonarchActorSystem.Config(launcher=SlurmLauncher.Config(...))`.
    Show one full example. Per `torchtitan_recipes/README.md`,
    cluster-specific recipes stay out of the public recipe set unless the
    project exercises them.
  - **Submitting.**
    `python -m torchtitan.rl.actors.monarch.submit --dry-run --module ... --config ...`,
    then the same command without `--dry-run`. Link to the `submit.py`
    docstring (Section 4.2) for what goes in the recipe versus an optional
    user `submit.sh` versus a `python_exe` wrapper.
  - **Layouts.** The supported layouts and the error you get otherwise.
  - **Where things go.** The env layers and default node-local caches, and
    where logs and artifacts are written.
  - **Failure behavior.** A failure on any node ends the job with a non-zero
    exit.

---

## 8. Upstream dependencies (Monarch / TorchStore)

Per `AGENTS.md` ("Investigate root cause before patching"), do not copy these
FairTitan workarounds into TorchTitan. Fix them upstream and track them here.

| Issue | FairTitan workaround | Ask |
| --- | --- | --- |
| Inherited `HOSTNAME` from the login node confuses TorchStore under Monarch `SlurmJob` | `unset HOSTNAME` plus a `sitecustomize` that resets it per process, loaded through a `python-slurm` alias (`submit.sh` L55-58, `sitecustomize.py`) | Root-cause it in Monarch's batch runner or in TorchStore. If v1 needs a stopgap, put it in the spawned-proc bootstrap with a TODO and a link. |
| The job state is written to `./.monarch/job_state.pkl` in the CWD (hardcoded in `SlurmJob._create`), so only one job can be submitted per working directory | Moves a stale file aside before each submit (`submit.sh` L97-104) | A job-state path parameter, so the launcher can use `<dump_folder>/launcher/job_state.pkl` |
| No way to get the generated sbatch script | Dry run prints the arguments only | `SlurmJob.render()` or a dry-run mode, for prime-rl-style script artifacts |
| No setup preamble in the script; only `#SBATCH` flags pass through `slurm_args` | Setup happens in `submit.sh` on the login node | An optional preamble or setup-command hook (prime-rl's `pre_run_command`), and explicit `export` lines so that `--export=ALL` is not required |
| `bind_to` is needed on dual-stack clusters (`/etc/hosts` IPv6 vs DNS IPv4) | `bind_to="::"` (`slurm.py` L218-230) | Confirm which `torchmonarch` release exposes `SlurmJob(bind_to=...)`, and bump `torchtitan/rl/requirements.txt` (currently `torchmonarch>=0.7.0`) if needed |
| Concurrent writers to FlashInfer's legacy autotune JSON on NFS (`ESTALE`) | Per-job `VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR` | Covered by the node-local cache defaults (Section 4.6). Remove once the pinned vLLM has the upstream fix (vllm-project/vllm#55671). |

---

## 9. Implementation plan

Each PR is self-contained and has CPU unit tests (no scheduler in CI). Port and
adapt FairTitan's
[`tests/test_slurm.py`](https://github.com/fairinternal/fairtitan/blob/5f912863b93f02454a4a096999b735346d28910c/launcher/rl/tests/test_slurm.py).

**PR 1: launcher interface and shared flow.**
- Add `actors/monarch/launcher.py` (`MonarchLauncher`, `Layout`, `LaunchPlan`,
  the planner, role and job naming).
- Add `actors/monarch/submit.py` (CLI, `--dry-run`, plan printing), with the
  module docstring from Section 4.2.
- Add `MonarchActorSystem.Config.launcher`, the `provision` branch with the
  submit-host guard, and `release_hosts` in `stop`.
- Make `spawn_generator_router` use the shared role-name helper.
- Tests, all with a fake in-memory `MonarchLauncher`:
  - the layout table: shared single node, `share_single_node=False`, a
    sub-node role gets one node (with a warning), whole nodes per role,
    rejection;
  - `Layout.host_meshes` passes the same object for a shared mesh;
  - `submit` argv forwarding (`--dry-run` stripped, everything else
    verbatim);
  - `provision` raises when `in_job()` is false;
  - `provision` uses `acquire_hosts`, and `stop` calls `release_hosts` even
    when a mesh stop fails;
  - `launcher=None` behaves exactly as today.

**PR 2: `SlurmLauncher`.**
- Add `actors/monarch/slurm.py` (Section 4.5).
- Tests:
  - mapping from config to `SlurmJob` arguments (patch `monarch.job.SlurmJob`
    and assert its kwargs, including None fields left out);
  - the controller command is `torchtitan.rl.train` with the forwarded
    arguments, shell-quoted;
  - `acquire_hosts` against a fake job state.

**PR 3: environment and artifacts.**
- Add `LAUNCHER_ENV_DEFAULTS` and the env merge (Section 4.6).
- Add job-scoped values and per-role parallelism-derived values in the
  spawned-proc bootstrap.
- Write launch artifacts (Section 4.7).
- Tests: merge order, SLURM `unset_env` wins, the bootstrap sets job-scoped
  paths only when a job id is present.

**PR 4: docs and one example.** The README sections (Section 7). Optionally,
an end-to-end smoke script for maintainers with SLURM access: a 2-step DAPO
Math run on one node, then with full-node roles on two nodes. The PR records
the job ids and logs.

**In parallel:** the Section 8 asks to Monarch and TorchStore. Then remove the
matching stopgaps.

**Follow-ups outside this repo:**
- FairTitan moves to `SlurmLauncher` plus a cluster profile (Section 6).
- Out-of-repo scheduler integrations move to `MonarchLauncher` subclasses.

---

## 10. Non-goals for v1

- Packing several sub-node roles onto shared nodes across a multi-node job,
  and a dedicated controller node for SLURM.
- Requeue and preemption handling, and waiting for an in-flight checkpoint
  before teardown.
- More implementations in this repo. The interface is designed for them;
  Kubernetes through Monarch's `KubernetesJob` is the natural next one, and is
  left to a later proposal.
- A scheduler abstraction for actor frameworks other than Monarch. Each
  `ActorSystem` brings its own.
- Environment installation, dataset staging, and site profiles in this repo.
- Pretraining (`torchtitan.train`) on a scheduler. `multinode_trainer.slurm`
  is unchanged.

---

## 11. Open questions for reviewers

1. **Entrypoint location.** Should the entrypoint be
   `python -m torchtitan.rl.actors.monarch.submit` (recommended: no change to
   the `ActorSystem` ABC, and the Monarch import boundary holds), or
   `python -m torchtitan.rl.submit` through an optional `ActorSystem.submit`
   hook?
2. **Operational scheduler flags.** Should the submit command accept
   operational flags such as `--time`, `--qos`, `--partition`, `--nodelist`
   or `--exclude`, like `--output-dir`? Or should these stay recipe-only,
   using a new recipe function or a registered `--override`? Recommendation:
   recipe-only in v1. Any flags would be per-launcher, which also argues for
   keeping the shared CLI minimal.
3. **Role names.** Should the names be `generator` for one generator and
   `generator_<i>` for several (recommended; matches today's actor names), or
   always `generator_<i>` (FairTitan today)?
4. **Layout.** Is deriving the layout automatically (Section 4.3) acceptable,
   or do reviewers want an explicit field like prime-rl's `deployment.type` so
   a recipe states its intent? Is giving a sub-node role a whole node (with a
   warning) the right default, rather than an error?
5. **Generic env defaults.** Which entries belong in `LAUNCHER_ENV_DEFAULTS`
   in this repo? The proposal includes only node-local caches and
   `PYTHONUNBUFFERED`.
6. **Config location.** Is `MonarchActorSystem.Config.launcher` the right
   home, rather than a `Controller.Config` subclass (what both existing
   launchers use today)?
