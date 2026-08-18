"""Policy evaluation entry point.

Example
-------
    python scripts/eval.py --task tienkung_pro/hdmi/scenario \
        --num_envs 200 --num_episodes 1000 --headless \
        --algo ppo_roa_adapt_est --checkpoint /path/to/checkpoint_final.pt

The argparse-style flags above are translated into hydra overrides, so any
plain hydra override (``key=value``) can still be mixed in freely, e.g.
``task.max_episode_length=600``.

Evaluation runs the environment until exactly ``--num_episodes`` episodes have
*finished*; environments terminate independently and simply start a new episode
after each reset. No training, no optimizer step, deterministic actions
(``ExplorationType.MODE``).
"""

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import datetime

import hydra
import termcolor
from omegaconf import OmegaConf, DictConfig

from isaaclab.app import AppLauncher

# registers the algo configs (ppo_roa_train / ppo_roa_adapt_est / ...) in hydra's
# ConfigStore; must happen before @hydra.main composes the config
import active_adaptation.learning  # noqa: F401


# argparse-style flag -> hydra override key
_VALUE_FLAGS = {
    "--task": "task",
    "--num_envs": "task.num_envs",
    "--num-envs": "task.num_envs",
    "--num_episodes": "num_episodes",
    "--num-episodes": "num_episodes",
    "--checkpoint": "checkpoint_path",
    "--checkpoint_path": "checkpoint_path",
    "--algo": "algo",
    "--seed": "seed",
    "--max_episode_length": "task.max_episode_length",
}
# argparse-style switch -> (hydra override key, value)
_SWITCH_FLAGS = {
    "--headless": ("headless", "true"),
    "--no_headless": ("headless", "false"),
    "--no-headless": ("headless", "false"),
    "--render": ("eval_render", "true"),
    "--stochastic": ("deterministic", "false"),
    "--print_success": ("print_success_episodes", "true"),
    "--no_print_success": ("print_success_episodes", "false"),
}


def translate_argv(argv):
    """Translate argparse-style flags into hydra overrides.

    Unknown ``--flags`` are passed through untouched so that hydra's own
    command line options (``--help``, ``--multirun``, ...) keep working.
    """
    out = []
    i = 0
    while i < len(argv):
        arg = argv[i]
        key, _, inline_value = arg.partition("=")
        if key in _SWITCH_FLAGS:
            override_key, value = _SWITCH_FLAGS[key]
            out.append(f"{override_key}={inline_value or value}")
        elif key in _VALUE_FLAGS:
            if inline_value:
                value = inline_value
            else:
                i += 1
                if i >= len(argv):
                    raise SystemExit(f"Missing value for {key}")
                value = argv[i]
            out.append(f"{_VALUE_FLAGS[key]}={value}")
        else:
            out.append(arg)
        i += 1
    return out


@hydra.main(config_path="../cfg", config_name="eval", version_base=None)
def main(cfg: DictConfig):
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)

    # tasks with an ego camera (e.g. depth observations) need the renderer even
    # when running headless
    if cfg.task.get("enable_cameras", False):
        cfg.app.enable_cameras = True

    app_launcher = AppLauncher(OmegaConf.to_container(cfg.app))
    simulation_app = app_launcher.app

    from scripts.helpers import make_env_policy, evaluate
    from active_adaptation.utils.eval_metrics import (
        evaluate_episodes, format_summary, format_success_episodes,
        format_failure_breakdown,
    )

    env, agent, vecnorm = make_env_policy(cfg)
    policy_eval = agent.get_rollout_policy("eval")

    if cfg.get("eval_render", False):
        # legacy fixed-horizon rollout, kept for video recording (eval_run.py -v)
        info, trajs, stats, policy_trajs = evaluate(
            env,
            policy_eval,
            render=True,
            render_mode=cfg.get("render_mode", "rgb_array"),
            seed=cfg.get("seed", 0),
        )
        summary = dict(info)
    else:
        summary = evaluate_episodes(
            env,
            policy_eval,
            num_episodes=int(cfg.get("num_episodes", 1000)),
            seed=int(cfg.get("seed", 0)),
            progress_every=cfg.get("progress_every", None),
            max_steps=cfg.get("max_eval_steps", None),
            deterministic=bool(cfg.get("deterministic", True)),
        )
        print()
        print(format_summary(summary))
        print()
        print(format_failure_breakdown(summary))
        if cfg.get("print_success_episodes", True):
            print()
            print(format_success_episodes(summary))

    info = dict(summary)
    info["task"] = cfg.task.name
    info["algo"] = cfg.algo.name
    info["checkpoint_path"] = cfg.checkpoint_path
    info["argv"] = sys.argv

    time_str = datetime.datetime.now().strftime("%m-%d_%H-%M")
    dir_path = os.path.join(os.path.dirname(__file__), "eval", str(cfg.task.name))
    os.makedirs(dir_path, exist_ok=True)
    path = os.path.join(dir_path, f"{cfg.task.name}-{time_str}.yaml")
    with open(path, "w") as f:
        OmegaConf.save(OmegaConf.create(info), f)
    print(termcolor.colored(f"Saved results to: {path}", "green"))

    os._exit(0)
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    sys.argv = [sys.argv[0]] + translate_argv(sys.argv[1:])
    main()
