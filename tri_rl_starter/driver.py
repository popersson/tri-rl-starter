"""Create and plot one random triangular mesh."""

from argparse import ArgumentParser
import os
import random
import sys

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from tri_rl_starter.geometry import CASE_NAMES, generate_case_mesh
    from tri_rl_starter.plotting import plot_mesh
else:
    from .geometry import CASE_NAMES, generate_case_mesh
    from .plotting import plot_mesh


def _format(value, digits=3):
    value = float(value)
    nearest = round(value)
    if abs(value - nearest) < 1.0e-10:
        return str(int(nearest))
    return f"{value:.{digits}f}".rstrip("0").rstrip(".")


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--case", choices=CASE_NAMES, default="random_polygon")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--nvertices", type=int, default=20)
    parser.add_argument("--nblocks", type=int, default=25)
    parser.add_argument("--np-out", type=int, default=20)
    parser.add_argument("--np-in", type=int, default=8)
    parser.add_argument("--chamfer-probability", type=float, default=0.4)
    parser.add_argument("--inner-scale", type=float, default=1.0 / 3.0)
    parser.add_argument("--hmax", type=float, default=None)
    parser.add_argument("--min-angle", type=float, default=28.6)
    parser.add_argument("--no-quality-refine", action="store_true")
    parser.add_argument("--smooth", action="store_true")
    parser.add_argument("--smooth-iterations", type=int, default=10)
    parser.add_argument("--smooth-omega", type=float, default=0.5)
    parser.add_argument("--smooth-no-check", action="store_true")
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--annotate-nodes", action="store_true")
    parser.add_argument(
        "--degree-colors",
        choices=["discrete", "continuous", "none"],
        default="discrete")
    parser.add_argument("--degree-color-min", type=int, default=-3)
    parser.add_argument("--degree-color-max", type=int, default=5)
    parser.add_argument("--plot-block", action="store_true")
    return parser.parse_args()


def print_summary(mesh):
    summary = mesh.summary()
    print("triangular mesh summary")
    print(f"  points: {summary['points']}")
    print(f"  triangles: {summary['triangles']}")
    print(f"  boundary nodes: {summary['boundary_nodes']}")
    print(f"  corner nodes: {summary['corner_nodes']}")
    print(
        "  degree L1/L2/Linf: "
        f"{_format(summary['degree_l1'])}/"
        f"{_format(summary['degree_l2'])}/"
        f"{_format(summary['degree_linf'])}")
    print(
        "  quality min/mean/p10: "
        f"{_format(summary['quality_min'])}/"
        f"{_format(summary['quality_mean'])}/"
        f"{_format(summary['quality_p10'])}")
    print(
        "  min-degree violations: "
        f"{summary['min_deficit_count']} "
        f"(total deficit {summary['min_deficit_total']})")


def main():
    args = parse_args()
    seed = (
        random.randrange(1, 1_000_000)
        if args.seed is None else args.seed
    )
    mesh = generate_case_mesh(
        case=args.case,
        seed=seed,
        nvertices=args.nvertices,
        nblocks=args.nblocks,
        np_out=args.np_out,
        np_in=args.np_in,
        chamfer_probability=args.chamfer_probability,
        inner_scale=args.inner_scale,
        hmax=args.hmax,
        min_angle=args.min_angle,
        quality=not args.no_quality_refine,
    )
    if args.smooth:
        mesh = mesh.smoothed(
            iterations=args.smooth_iterations,
            omega=args.smooth_omega,
            check_inversion=not args.smooth_no_check)
    print(
        f"case={args.case}, seed={seed}, "
        f"hmax={args.hmax}, min_angle={args.min_angle}")
    if args.smooth:
        print(
            "smoother: "
            f"iterations={args.smooth_iterations}, "
            f"omega={args.smooth_omega}, "
            f"check_inversion={not args.smooth_no_check}")
    print_summary(mesh)
    if not args.no_plot:
        plot_mesh(
            mesh,
            annotate_nodes=args.annotate_nodes,
            degree_coloring=args.degree_colors,
            degree_range=(args.degree_color_min, args.degree_color_max),
            block=args.plot_block)


if __name__ == "__main__":
    main()
