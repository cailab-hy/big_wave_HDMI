"""Evaluation-only tracking metrics.

Nothing in this module is used by training. The recorder attaches itself to an
already-constructed environment, reads exactly the same buffers that the motion
tracking rewards read (see
``active_adaptation/envs/mdp/commands/hdmi/rewards.py``) and never writes to the
environment state.

Timing note
-----------
``_Env._step`` runs, in order:

    physics substeps -> ``_update()`` -> ``_compute_reward()`` -> ``command_manager.update()``

``command_manager.update()`` advances the reference motion (``self.t += 1``), so
the reference buffers (``ref_joint_pos``, ``ref_body_pos_w``, ...) are only
aligned with the current robot state *before* it runs. The recorder therefore
hooks ``env._update_callbacks``, i.e. the same window in which the rewards are
evaluated.

The tracking command does not override ``reset``, so right after an episode
reset the reference buffers still hold the *previous* episode's last reference
until ``update()`` runs at the end of the first step. That single step is
therefore skipped by the recorder (it would compare the freshly reset robot
against the end of the previous motion). Nothing about the environment is
changed - only which samples the metric aggregates.
"""

from __future__ import annotations

import math
from typing import Dict, List, Sequence

import torch


# Reward classes (in the hdmi reward module) that define the metric we want,
# in decreasing order of preference. Each entry is (class names, frame).
_BODY_SOURCES = (
    (("keypoint_pos_error_local",), "local"),
    (("keypoint_pos_tracking_local_product",), "local"),
    (("keypoint_pos_error",), "world"),
    (("keypoint_pos_tracking_product",), "world"),
)
_JOINT_SOURCES = (
    ("joint_pos_error",),
    ("joint_pos_tracking_product",),
)


class TrackingErrorRecorder:
    """Per-step joint / body tracking error accumulator.

    Errors are accumulated per environment; :meth:`pop` returns (and clears) the
    accumulated sums for a set of environments, which is how a finished episode
    is turned into a single record.
    """

    def __init__(self, env, verbose: bool = True):
        self.env = env
        self.command_manager = env.command_manager
        self.device = env.device
        self.num_envs = env.num_envs
        self.verbose = verbose

        self.joint_names: List[str] = []
        self.body_names: List[str] = []
        self.body_frame = "local"
        self.joint_source = "none"
        self.body_source = "none"

        self.has_joint_metric = self._resolve_joints()
        self.has_body_metric = self._resolve_bodies()

        f64 = dict(dtype=torch.float64, device=self.device)
        i64 = dict(dtype=torch.int64, device=self.device)
        self.joint_abs_sum = torch.zeros(self.num_envs, **f64)
        self.joint_sq_sum = torch.zeros(self.num_envs, **f64)
        self.joint_count = torch.zeros(self.num_envs, **i64)
        self.body_abs_sum = torch.zeros(self.num_envs, **f64)
        self.body_sq_sum = torch.zeros(self.num_envs, **f64)
        self.body_count = torch.zeros(self.num_envs, **i64)
        self.steps = torch.zeros(self.num_envs, **i64)
        self.num_nonfinite = torch.zeros((), **i64)
        self.num_skipped_steps = torch.zeros((), **i64)
        # first step of an episode: reference buffers are still the previous
        # episode's (see module docstring)
        self._skip = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)

        if self.has_joint_metric or self.has_body_metric:
            env._update_callbacks.append(self.record)

        if verbose:
            self.print_config()

    # ------------------------------------------------------------------
    # metric definition: reuse whatever the task already defines
    # ------------------------------------------------------------------
    def _reward_instances(self, class_names: Sequence[str]):
        from active_adaptation.envs.mdp.commands.hdmi import rewards as hdmi_rewards

        classes = tuple(
            getattr(hdmi_rewards, name) for name in class_names if hasattr(hdmi_rewards, name)
        )
        if not classes:
            return []
        found = []
        for group in getattr(self.env, "reward_groups", {}).values():
            for func in group.funcs.values():
                if isinstance(func, classes):
                    found.append(func)
        return found

    def _warn_tolerance(self, instances):
        for inst in instances:
            tolerance = getattr(inst, "tolerance", None)
            if tolerance is not None and float(tolerance.abs().max()) > 0.0:
                print(
                    f"[WARN] {type(inst).__name__} uses a non-zero tolerance; "
                    "the evaluation metric reports the raw error instead."
                )
                return

    def _resolve_joints(self) -> bool:
        cm = self.command_manager
        if not hasattr(cm, "ref_joint_pos") or not hasattr(cm, "tracking_joint_names"):
            return False

        names: List[str] = []
        for class_names in _JOINT_SOURCES:
            instances = self._reward_instances(class_names)
            if instances:
                self._warn_tolerance(instances)
                matched = set()
                for inst in instances:
                    matched.update(inst.joint_names)
                names = [n for n in cm.tracking_joint_names if n in matched]
                self.joint_source = "+".join(sorted({type(i).__name__ for i in instances}))
                break
        if not names:
            names = list(cm.tracking_joint_names)
            self.joint_source = "command_manager.tracking_joint_names"

        self.joint_names = names
        self.joint_indices_asset = [cm.asset.joint_names.index(n) for n in names]
        self.joint_indices_motion = [cm.tracking_joint_names.index(n) for n in names]
        return len(names) > 0

    def _resolve_bodies(self) -> bool:
        cm = self.command_manager
        if not hasattr(cm, "ref_body_pos_w") or not hasattr(cm, "tracking_keypoint_names"):
            return False

        names: List[str] = []
        for class_names, frame in _BODY_SOURCES:
            instances = self._reward_instances(class_names)
            if instances:
                self._warn_tolerance(instances)
                matched = set()
                for inst in instances:
                    matched.update(inst.body_names)
                names = [n for n in cm.tracking_keypoint_names if n in matched]
                self.body_frame = frame
                self.body_source = "+".join(sorted({type(i).__name__ for i in instances}))
                break
        if not names:
            names = list(cm.tracking_keypoint_names)
            self.body_frame = "local"
            self.body_source = "command_manager.tracking_keypoint_names"

        self.body_names = names
        self.body_indices_asset = [cm.asset.body_names.index(n) for n in names]
        self.body_indices_motion = [cm.tracking_keypoint_names.index(n) for n in names]
        return len(names) > 0

    def print_config(self):
        print("[Eval] tracking metric configuration")
        if self.has_joint_metric:
            print(f"  joint error   : {len(self.joint_names)} joints from {self.joint_source}")
            print(f"                  {self.joint_names}")
        else:
            print("  joint error   : unavailable for this task")
        if self.has_body_metric:
            print(
                f"  body error    : {len(self.body_names)} bodies from {self.body_source} "
                f"({self.body_frame} frame)"
            )
            print(f"                  {self.body_names}")
        else:
            print("  body error    : unavailable for this task")

    # ------------------------------------------------------------------
    # per-step recording
    # ------------------------------------------------------------------
    def _body_pos_pair(self):
        """Current / reference body positions, in the frame used by the reward."""
        from isaaclab.utils.math import quat_apply_inverse, yaw_quat

        cm = self.command_manager
        body_pos_asset = cm.asset.data.body_link_pos_w[:, self.body_indices_asset]
        body_pos_motion = cm.ref_body_pos_w[:, self.body_indices_motion]
        if self.body_frame != "local":
            return body_pos_asset, body_pos_motion

        # identical to keypoint_pos_error_local / keypoint_pos_tracking_local_product:
        # express both in the yaw-only, ground-projected root frame so that the
        # world-frame drift of the root does not enter the metric.
        num_bodies = len(self.body_indices_asset)
        root_pos_asset = cm.robot_root_pos_w.clone()
        root_pos_motion = cm.ref_root_pos_w.clone()
        root_pos_asset[..., 2] = 0.0
        root_pos_motion[..., 2] = 0.0
        root_quat_asset = yaw_quat(cm.robot_root_quat_w)
        root_quat_motion = yaw_quat(cm.ref_root_quat_w)

        root_pos_asset = root_pos_asset.unsqueeze(1).expand(-1, num_bodies, -1)
        root_pos_motion = root_pos_motion.unsqueeze(1).expand(-1, num_bodies, -1)
        root_quat_asset = root_quat_asset.unsqueeze(1).expand(-1, num_bodies, -1)
        root_quat_motion = root_quat_motion.unsqueeze(1).expand(-1, num_bodies, -1)

        return (
            quat_apply_inverse(root_quat_asset, body_pos_asset - root_pos_asset),
            quat_apply_inverse(root_quat_motion, body_pos_motion - root_pos_motion),
        )

    def _accumulate(self, error, valid, abs_sum, sq_sum, count):
        """error: [num_envs, N] non-negative per-element error."""
        finite = torch.isfinite(error)
        keep = finite & valid.unsqueeze(1)
        self.num_nonfinite += (~finite & valid.unsqueeze(1)).sum()
        error = torch.where(keep, error, torch.zeros_like(error)).double()
        abs_sum += error.sum(dim=1)
        sq_sum += error.square().sum(dim=1)
        count += keep.sum(dim=1)

    @torch.no_grad()
    def record(self):
        cm = self.command_manager
        valid = ~self._skip
        if self.has_joint_metric:
            joint_pos = cm.asset.data.joint_pos[:, self.joint_indices_asset]
            ref_joint_pos = cm.ref_joint_pos[:, self.joint_indices_motion]
            # [num_envs, num_joints], rad
            joint_error = (joint_pos - ref_joint_pos).abs()
            self._accumulate(
                joint_error, valid, self.joint_abs_sum, self.joint_sq_sum, self.joint_count
            )

        if self.has_body_metric:
            body_pos, ref_body_pos = self._body_pos_pair()
            # [num_envs, num_bodies], m
            body_error = (body_pos - ref_body_pos).norm(dim=-1)
            self._accumulate(
                body_error, valid, self.body_abs_sum, self.body_sq_sum, self.body_count
            )

        self.steps += valid
        self.num_skipped_steps += self._skip.sum()
        self._skip.fill_(False)

    # ------------------------------------------------------------------
    # episode bookkeeping
    # ------------------------------------------------------------------
    def peek(self, env_ids: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {
            "joint_abs_sum": self.joint_abs_sum[env_ids].clone(),
            "joint_sq_sum": self.joint_sq_sum[env_ids].clone(),
            "joint_count": self.joint_count[env_ids].clone(),
            "body_abs_sum": self.body_abs_sum[env_ids].clone(),
            "body_sq_sum": self.body_sq_sum[env_ids].clone(),
            "body_count": self.body_count[env_ids].clone(),
            "steps": self.steps[env_ids].clone(),
        }

    def clear(self, env_ids: torch.Tensor | None = None):
        if env_ids is None:
            env_ids = slice(None)
        self.joint_abs_sum[env_ids] = 0.0
        self.joint_sq_sum[env_ids] = 0.0
        self.joint_count[env_ids] = 0
        self.body_abs_sum[env_ids] = 0.0
        self.body_sq_sum[env_ids] = 0.0
        self.body_count[env_ids] = 0
        self.steps[env_ids] = 0
        self._skip[env_ids] = True


class EvaluationAccumulator:
    """Aggregates finished episodes into the final evaluation statistics."""

    def __init__(self, recorder: TrackingErrorRecorder, num_episodes: int):
        self.recorder = recorder
        self.num_episodes = num_episodes

        self.joint_abs_sum = 0.0
        self.joint_sq_sum = 0.0
        self.joint_count = 0
        self.body_abs_sum = 0.0
        self.body_sq_sum = 0.0
        self.body_count = 0

        self.num_success = 0
        self.episode_joint_mean: List[float] = []
        self.episode_body_mean: List[float] = []
        self.episode_success: List[float] = []
        self.episode_len: List[float] = []
        # identifies the trial an episode came from: which environment, and the
        # how-many-th episode of that environment
        self.episode_env_id: List[int] = []
        self.episode_index: List[int] = []
        # failure diagnosis
        self.termination_counts: Dict[str, int] = {}
        self.failed_len_sum = 0.0
        self.success_len_sum = 0.0

    @property
    def completed(self) -> int:
        return len(self.episode_success)

    @property
    def remaining(self) -> int:
        return self.num_episodes - self.completed

    def add_episodes(
        self,
        env_ids: torch.Tensor,
        success: torch.Tensor,
        episode_len: torch.Tensor,
        episode_index: torch.Tensor | None = None,
        termination: Dict[str, torch.Tensor] | None = None,
    ):
        """Record one finished episode per entry of ``env_ids``."""
        if env_ids.numel() == 0:
            return
        rec = self.recorder.peek(env_ids)

        self.episode_env_id.extend(env_ids.tolist())
        if episode_index is None:
            self.episode_index.extend([-1] * env_ids.numel())
        else:
            self.episode_index.extend(episode_index.tolist())

        joint_count = rec["joint_count"].double()
        body_count = rec["body_count"].double()
        safe_joint = joint_count.clamp_min(1.0)
        safe_body = body_count.clamp_min(1.0)

        self.joint_abs_sum += float(rec["joint_abs_sum"].sum())
        self.joint_sq_sum += float(rec["joint_sq_sum"].sum())
        self.joint_count += int(rec["joint_count"].sum())
        self.body_abs_sum += float(rec["body_abs_sum"].sum())
        self.body_sq_sum += float(rec["body_sq_sum"].sum())
        self.body_count += int(rec["body_count"].sum())

        self.episode_joint_mean.extend((rec["joint_abs_sum"] / safe_joint).tolist())
        self.episode_body_mean.extend((rec["body_abs_sum"] / safe_body).tolist())
        self.episode_len.extend(episode_len.float().tolist())

        # success is a per-step flag; taking it at the terminal step counts each
        # episode at most once (see _Env._compute_reward / RobotTracking.success)
        success_flags = (success > 0.5).float()
        self.num_success += int(success_flags.sum())
        self.episode_success.extend(success_flags.tolist())

        # why did the failed episodes end? stats["termination"][key] holds each
        # termination condition's flag at the terminal step (_Env._compute_termination)
        failed = success_flags < 0.5
        num_failed = int(failed.sum())
        if num_failed:
            self.failed_len_sum += float(episode_len.float()[failed].sum())
            attributed = torch.zeros_like(failed)
            for key, flag in (termination or {}).items():
                fired = (flag.reshape(-1) > 0.5) & failed
                self.termination_counts[key] = self.termination_counts.get(key, 0) + int(fired.sum())
                attributed |= fired
            # no termination flag -> ran out of max_episode_length before the
            # motion finished
            self.termination_counts["(timeout / no flag)"] = (
                self.termination_counts.get("(timeout / no flag)", 0)
                + int((failed & ~attributed).sum())
            )
        if int(success_flags.sum()):
            self.success_len_sum += float(episode_len.float()[success_flags > 0.5].sum())

    def _fully_successful_envs(self) -> List[int]:
        total: Dict[int, int] = {}
        good: Dict[int, int] = {}
        for env, ok in zip(self.episode_env_id, self.episode_success):
            total[env] = total.get(env, 0) + 1
            good[env] = good.get(env, 0) + int(ok > 0.5)
        return sorted(env for env in total if good[env] == total[env])

    def summary(self) -> Dict[str, object]:
        rec = self.recorder
        n = self.completed

        def _mean(values):
            return float(sum(values) / len(values)) if values else float("nan")

        def _std(values):
            if len(values) < 2:
                return 0.0
            mean = _mean(values)
            return math.sqrt(sum((v - mean) ** 2 for v in values) / (len(values) - 1))

        out: Dict[str, object] = {
            "num_episodes": n,
            "num_success": self.num_success,
            "success_rate": (self.num_success / n) if n else float("nan"),
            "episode_len_mean": _mean(self.episode_len),
            "episode_len_std": _std(self.episode_len),
            "joint_metric_available": rec.has_joint_metric,
            "body_metric_available": rec.has_body_metric,
            "joint_names": list(rec.joint_names),
            "body_names": list(rec.body_names),
            "body_frame": rec.body_frame,
            "joint_metric_source": rec.joint_source,
            "body_metric_source": rec.body_source,
            "num_nonfinite_samples": int(rec.num_nonfinite),
            # one step per episode is skipped, see the module docstring
            "num_skipped_steps": int(rec.num_skipped_steps),
            "num_joint_samples": self.joint_count,
            "num_body_samples": self.body_count,
            # trials that succeeded, as "<env_id>:<episode_index>"
            "success_episodes": [
                f"{env}:{idx}"
                for env, idx, ok in zip(
                    self.episode_env_id, self.episode_index, self.episode_success
                )
                if ok > 0.5
            ],
            # environments with at least one successful episode
            "success_env_ids": sorted(
                {
                    env
                    for env, ok in zip(self.episode_env_id, self.episode_success)
                    if ok > 0.5
                }
            ),
            # environments in which every finished episode succeeded
            "all_success_env_ids": self._fully_successful_envs(),
            # which termination condition ended the failed episodes
            "failure_breakdown": {
                k: v for k, v in sorted(
                    self.termination_counts.items(), key=lambda kv: -kv[1]
                ) if v
            },
            "failed_episode_len_mean": (
                self.failed_len_sum / (n - self.num_success) if n - self.num_success else 0.0
            ),
            "success_episode_len_mean": (
                self.success_len_sum / self.num_success if self.num_success else 0.0
            ),
        }
        if self.joint_count:
            out["joint_pos_error_mean"] = self.joint_abs_sum / self.joint_count
            out["joint_pos_error_rmse"] = math.sqrt(self.joint_sq_sum / self.joint_count)
            out["joint_pos_error_episode_std"] = _std(self.episode_joint_mean)
        if self.body_count:
            out["body_pos_error_mean"] = self.body_abs_sum / self.body_count
            out["body_pos_error_rmse"] = math.sqrt(self.body_sq_sum / self.body_count)
            out["body_pos_error_episode_std"] = _std(self.episode_body_mean)
        return out


def format_summary(summary: Dict[str, object]) -> str:
    line = "=" * 60
    n = summary["num_episodes"]
    rows = [
        line,
        "                    Evaluation Results",
        line,
        f"{'Total Episodes':<32}: {n}",
        f"{'Successful Episodes':<32}: {summary['num_success']}",
        f"{'Success Rate':<32}: {summary['success_rate'] * 100:.2f} %",
        f"{'Mean Episode Length':<32}: {summary['episode_len_mean']:.1f} steps",
        "",
        "Joint Position Tracking Error",
    ]
    if "joint_pos_error_mean" in summary:
        rows += [
            f"  {'Mean':<30}: {summary['joint_pos_error_mean']:.4f} rad",
            f"  {'RMSE':<30}: {summary['joint_pos_error_rmse']:.4f} rad",
            f"  {'Joints / Samples':<30}: {len(summary['joint_names'])} / {summary['num_joint_samples']}",
        ]
    else:
        rows.append(f"  {'Mean':<30}: n/a (not available for this task)")
    rows += ["", "Body Position Tracking Error"]
    if "body_pos_error_mean" in summary:
        rows += [
            f"  {'Mean':<30}: {summary['body_pos_error_mean']:.4f} m",
            f"  {'RMSE':<30}: {summary['body_pos_error_rmse']:.4f} m",
            f"  {'Frame':<30}: {summary['body_frame']}",
            f"  {'Bodies / Samples':<30}: {len(summary['body_names'])} / {summary['num_body_samples']}",
        ]
    else:
        rows.append(f"  {'Mean':<30}: n/a (not available for this task)")
    if summary["num_nonfinite_samples"]:
        rows += ["", f"[WARN] {summary['num_nonfinite_samples']} non-finite error samples were skipped"]
    rows.append(line)
    return "\n".join(rows)


def format_failure_breakdown(summary: Dict[str, object]) -> str:
    """Which termination condition ended the failed episodes."""
    line = "=" * 60
    n = summary["num_episodes"]
    failed = n - summary["num_success"]
    rows = [
        line,
        f"       Failure Breakdown  ({failed} / {n} episodes failed)",
        line,
    ]
    if not failed:
        rows += ["  (no failures)", line]
        return "\n".join(rows)
    breakdown = summary.get("failure_breakdown", {})
    for key, count in breakdown.items():
        # conditions can fire together, so the shares do not have to sum to 100%
        rows.append(f"  {key:<36}: {count:>5}  ({count / failed * 100:5.1f}% of failures)")
    rows += [
        "",
        f"  {'mean length of failed episodes':<36}: {summary['failed_episode_len_mean']:.1f} steps",
        f"  {'mean length of successful episodes':<36}: {summary['success_episode_len_mean']:.1f} steps",
        line,
    ]
    return "\n".join(rows)


def format_success_episodes(summary: Dict[str, object], per_line: int = 10) -> str:
    """List only the trials that succeeded, as ``<env_id>:<episode_index>``."""
    line = "=" * 60
    def _key(entry: str):
        env, _, idx = entry.partition(":")
        return (int(env), int(idx) if idx else 0)

    episodes = sorted(summary.get("success_episodes", []), key=_key)
    seed = summary.get("seed")
    rows = [
        line,
        f"       Successful Trials  ({len(episodes)} / {summary['num_episodes']})",
        line,
        f"seed = {seed}   |   entry format = <env_id>:<episode_index>",
        "",
    ]
    if not episodes:
        rows.append("  (none)")
    else:
        width = max(len(e) for e in episodes) + 2
        for i in range(0, len(episodes), per_line):
            chunk = episodes[i:i + per_line]
            rows.append("  " + "".join(e.ljust(width) for e in chunk).rstrip())
    env_ids = summary.get("success_env_ids", [])
    all_env_ids = summary.get("all_success_env_ids", [])
    rows += [
        "",
        f"env ids with >=1 success ({len(env_ids)}): {env_ids}",
        f"env ids with all episodes successful ({len(all_env_ids)}): {all_env_ids}",
        line,
    ]
    return "\n".join(rows)


@torch.inference_mode()
def evaluate_episodes(
    env,
    policy,
    num_episodes: int,
    seed: int = 0,
    progress_every: int | None = None,
    max_steps: int | None = None,
    deterministic: bool = True,
):
    """Roll out ``policy`` until exactly ``num_episodes`` episodes have finished.

    Episodes are tracked per environment, so environments that terminate at
    different simulation steps are handled independently; an environment simply
    keeps starting new episodes after each reset until the global budget is met.
    """
    from torchrl.envs import ExplorationType, set_exploration_type

    base_env = env
    for _ in range(8):  # unwrap TransformedEnv layers
        inner = getattr(base_env, "base_env", None)
        if inner is None or inner is base_env:
            break
        base_env = inner

    base_env.eval()
    env.eval()
    env.set_seed(seed)

    recorder = TrackingErrorRecorder(base_env)
    accumulator = EvaluationAccumulator(recorder, num_episodes)

    if progress_every is None:
        progress_every = max(1, base_env.num_envs)
    progress_every = max(1, int(progress_every))
    if max_steps is None:
        rounds = math.ceil(num_episodes / base_env.num_envs) + 2
        max_steps = int(rounds * base_env.max_episode_length)

    exploration_type = ExplorationType.MODE if deterministic else ExplorationType.RANDOM

    td_ = env.reset()
    recorder.clear()
    next_report = progress_every
    step = 0
    # how-many-th episode each environment is currently running
    episode_index = torch.zeros(base_env.num_envs, dtype=torch.long, device=base_env.device)

    print(f"[Eval] running {num_episodes} episodes with {base_env.num_envs} environments "
          f"({'deterministic' if deterministic else 'stochastic'} actions)")

    torch.compiler.cudagraph_mark_step_begin()
    with set_exploration_type(exploration_type):
        while accumulator.completed < num_episodes and step < max_steps:
            td_ = policy(td_)
            td, td_ = env.step_and_maybe_reset(td_)
            step += 1

            done = td["next", "done"].reshape(-1)
            if bool(done.any()):
                done_ids = done.nonzero().reshape(-1)
                # only the first `remaining` finished episodes are aggregated so
                # that exactly `num_episodes` results enter the statistics
                keep = done_ids[: accumulator.remaining]
                stats = td["next", "stats"]
                termination = None
                if "termination" in stats.keys():
                    termination = {
                        key: value.reshape(-1)[keep]
                        for key, value in stats["termination"].items()
                    }
                accumulator.add_episodes(
                    keep,
                    stats["success"].reshape(-1)[keep],
                    stats["episode_len"].reshape(-1)[keep],
                    episode_index[keep],
                    termination,
                )
                episode_index[done_ids] += 1
                # every finished environment starts a fresh episode after reset
                recorder.clear(done_ids)

                while accumulator.completed >= next_report and next_report <= num_episodes:
                    print(f"Evaluation Progress: {min(accumulator.completed, num_episodes)} / {num_episodes}")
                    next_report += progress_every

    if accumulator.completed < num_episodes:
        print(
            f"[WARN] stopped after {step} steps with only {accumulator.completed}/{num_episodes} "
            "episodes completed (max_steps reached)."
        )
    elif accumulator.completed != next_report - progress_every:
        print(f"Evaluation Progress: {accumulator.completed} / {num_episodes}")

    summary = accumulator.summary()
    summary["num_env_steps"] = step
    summary["num_envs"] = base_env.num_envs
    summary["deterministic"] = deterministic
    summary["seed"] = seed
    return summary
