"""Integration test: STL analysis + honeycomb + trajectory + FK + IK.

Tests the full robotics pipeline locally on the provided scaffold STL.
"""

import sys
sys.path.insert(0, ".")

import numpy as np
from pathlib import Path


def test_stl_analysis():
    print("[1/5] Testing STL analysis + void detection...")
    from modules.stl_analysis import analyze_scaffold

    stl_path = "data/scaffold_curved_void.stl"
    if not Path(stl_path).exists():
        print("  SKIP: STL file not found")
        return None

    result = analyze_scaffold(stl_path)

    assert result["vertices"].shape[1] == 3
    assert result["cylinder"]["radius"] > 0
    assert result["void_bounds"]["void_width"] > 0
    assert result["void_bounds"]["void_length"] > 0
    assert result["void_bounds"]["shell_thickness"] > 0

    cyl = result["cylinder"]
    vb = result["void_bounds"]
    print(f"  Cylinder: R={cyl['radius']:.1f}mm, center=({cyl['cy']:.1f}, {cyl['cz']:.1f})")
    print(f"  Void: {vb['void_width']:.1f}mm x {vb['void_length']:.1f}mm, shell={vb['shell_thickness']:.1f}mm")
    print("  -> PASS")
    return result


def test_honeycomb_and_tsp(scaffold_result):
    print("\n[2/5] Testing honeycomb grid + TSP...")
    from modules.honeycomb import create_hex_grid, compute_grid_params
    from modules.tsp_solver import optimize_visitation_order

    vb = scaffold_result["void_bounds"]
    nx, ny, hex_side = compute_grid_params(vb["void_width"], vb["void_length"])
    grid = create_hex_grid(nx, ny, hex_side)

    assert grid.shape == (ny, nx, 2)
    print(f"  Grid: {nx}x{ny} cells, hex_side={hex_side:.1f}mm")

    cell_indices = np.array([[ix, iy] for iy in range(ny) for ix in range(nx)])
    ordered = optimize_visitation_order(grid, cell_indices, rise=20.0, time_limit=30)

    assert len(ordered) == len(cell_indices)
    print("  -> PASS")
    return grid, ordered, hex_side


def test_trajectory_planning(scaffold_result):
    print("\n[3/5] Testing full trajectory planning...")
    from modules.trajectory_planner import plan_full_trajectory

    cyl = scaffold_result["cylinder"]
    vb = scaffold_result["void_bounds"]

    result = plan_full_trajectory(
        void_bounds=vb,
        cyl_radius=cyl["radius"],
        cyl_cy=cyl["cy"],
        cyl_cz=cyl["cz"],
        rise=20.0,
        optimize_tsp=True,
    )

    assert result["traj_m"].shape[0] == 3
    assert result["R_targets"].shape[0] == 3
    assert result["n_points"] > 0
    print(f"  Trajectory: {result['n_points']} points")
    print("  -> PASS")
    return result


def test_forward_kinematics():
    print("\n[4/5] Testing robot model + FK...")
    from modules.robot_model import (
        forward_kinematics_8dof, geometric_jacobian_8dof,
        manipulability, home_configuration,
    )

    q_home = home_configuration()
    T = forward_kinematics_8dof(q_home)
    assert T.shape == (4, 4)
    assert np.allclose(T[3, :], [0, 0, 0, 1])

    J = geometric_jacobian_8dof(q_home)
    assert J.shape == (6, 8)

    mu = manipulability(J)
    assert mu >= 0

    print(f"  Home FK position: [{T[0,3]:.4f}, {T[1,3]:.4f}, {T[2,3]:.4f}] m")
    print(f"  Home manipulability: {mu:.6f}")
    print("  -> PASS")


def test_inverse_kinematics():
    print("\n[5/5] Testing IK solver (5 sample points)...")
    from modules.robot_model import forward_kinematics_8dof
    from modules.inverse_kinematics import InverseKinematicsSolver, IKParams

    params = IKParams()
    params.max_iter = 50  # reduce for speed
    solver = InverseKinematicsSolver(params)

    # Test with random reachable targets
    successes = 0
    for i in range(5):
        np.random.seed(i)
        q_test = np.random.uniform(-0.3, 0.3, size=8)
        T_target = forward_kinematics_8dof(q_test)

        q_sol, info = solver.solve(T_target)
        if info["pos_err"] < 0.01:
            successes += 1

    print(f"  IK success: {successes}/5 (< 10mm error)")
    assert successes >= 3, f"Only {successes}/5 IK solutions converged"
    print("  -> PASS")


if __name__ == "__main__":
    print("=" * 60)
    print("TRAJECTORY PIPELINE INTEGRATION TEST")
    print("=" * 60)

    scaffold = test_stl_analysis()
    if scaffold is not None:
        test_honeycomb_and_tsp(scaffold)
        test_trajectory_planning(scaffold)
    test_forward_kinematics()
    test_inverse_kinematics()

    print("\n" + "=" * 60)
    print("ALL TRAJECTORY PIPELINE TESTS PASSED")
    print("=" * 60)
