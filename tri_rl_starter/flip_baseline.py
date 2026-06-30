"""Simple non-learning baselines for the topology-only flip environment."""

from argparse import ArgumentParser

import numpy as np

from .env import MeshCaseConfig, TriFlipEnv, degree_l1_score
from .plotting import plot_mesh
from .topology import degree_l1_after_flip


def random_action(observation, rng):
    """Choose a random candidate flip, or no-op if no flip is available."""
    if observation.n_candidates == 0:
        return 0
    return int(rng.integers(1, observation.n_candidates + 1))


def greedy_degree_l1_action(env, allow_worse=False):
    """Choose the flip with the best immediate degree-L1 score."""
    observation = env.observation()
    if observation.n_candidates == 0:
        return 0

    current = degree_l1_score(env.mesh)
    best_action = 0
    best_score = current
    for index, edge in enumerate(observation.candidate_edges, start=1):
        score = degree_l1_after_flip(env.mesh, edge)
        if score < best_score or (allow_worse and best_action == 0):
            best_action = index
            best_score = score
    return best_action


def rollout(env, policy="greedy", seed=None, allow_worse=False):
    """Run one baseline rollout and return per-step history."""
    rng = np.random.default_rng(seed)
    observation = env.observation()
    history = [{
        "step": 0,
        "score": observation.score,
        "degree_l1": observation.degree_l1,
        "reward": 0.0,
        "action": None,
    }]

    terminated = False
    truncated = False
    while not (terminated or truncated):
        if policy == "random":
            action = random_action(observation, rng)
        elif policy == "greedy":
            action = greedy_degree_l1_action(env, allow_worse=allow_worse)
        else:
            raise ValueError("policy must be 'greedy' or 'random'")
        observation, reward, terminated, truncated, info = env.step(action)
        history.append({
            "step": info["step_count"],
            "score": info["score"],
            "degree_l1": info["degree_l1"],
            "reward": reward,
            "action": action,
        })
    return history


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--policy", choices=["greedy", "random"], default="greedy")
    parser.add_argument("--case", default="random_polygon")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--nvertices", type=int, default=20)
    parser.add_argument("--hmax", type=float, default=0.35)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--max-steps-factor", type=float, default=1.0)
    parser.add_argument("--allow-worse", action="store_true")
    parser.add_argument("--plot-initial", action="store_true")
    parser.add_argument("--plot-final-smoothed", action="store_true")
    parser.add_argument("--plot-block", action="store_true")
    parser.add_argument("--annotate-nodes", action="store_true")
    parser.add_argument(
        "--degree-colors",
        choices=["discrete", "continuous", "none"],
        default="discrete")
    parser.add_argument("--smooth-iterations", type=int, default=10)
    parser.add_argument("--smooth-omega", type=float, default=0.5)
    parser.add_argument("--smooth-no-check", action="store_true")
    return parser.parse_args()


def plot_rollout_meshes(initial_mesh, final_mesh, args):
    """Plot requested rollout meshes in separate Matplotlib windows."""
    if not (args.plot_initial or args.plot_final_smoothed):
        return

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("baseline plotting requires matplotlib") from exc

    if args.plot_initial:
        fig = plt.figure("initial mesh")
        fig.clf()
        ax = fig.add_subplot(111)
        plot_mesh(
            initial_mesh,
            ax=ax,
            show=False,
            annotate_nodes=args.annotate_nodes,
            degree_coloring=args.degree_colors)
        ax.set_title("initial mesh")

    if args.plot_final_smoothed:
        smoothed = final_mesh.smoothed(
            iterations=args.smooth_iterations,
            omega=args.smooth_omega,
            check_inversion=not args.smooth_no_check)
        fig = plt.figure("smoothed final mesh")
        fig.clf()
        ax = fig.add_subplot(111)
        plot_mesh(
            smoothed,
            ax=ax,
            show=False,
            annotate_nodes=args.annotate_nodes,
            degree_coloring=args.degree_colors)
        ax.set_title("smoothed final mesh")

    plt.show(block=args.plot_block)
    if not args.plot_block:
        plt.pause(0.001)


def main():
    args = parse_args()
    config = MeshCaseConfig(
        case=args.case,
        seed=args.seed,
        nvertices=args.nvertices,
        hmax=args.hmax,
    )
    env = TriFlipEnv(
        mesh_config=config,
        max_steps=args.max_steps,
        max_steps_factor=args.max_steps_factor,
    )
    env.reset(seed=args.seed)
    initial_mesh = env.mesh
    history = rollout(
        env,
        policy=args.policy,
        seed=args.seed,
        allow_worse=args.allow_worse)
    final_mesh = env.mesh
    start = history[0]
    end = history[-1]
    improvement = start["degree_l1"] - end["degree_l1"]
    print(
        f"policy={args.policy}, case={args.case}, seed={args.seed}, "
        f"steps={end['step']}")
    print(
        "degree_l1: "
        f"{start['degree_l1']:.6g} -> {end['degree_l1']:.6g} "
        f"(improvement {improvement:.6g})")
    print(f"best_score={env.best_score:.6g}, optimum={env.info()['optimum_score']:.6g}")
    plot_rollout_meshes(initial_mesh, final_mesh, args)


if __name__ == "__main__":
    main()
