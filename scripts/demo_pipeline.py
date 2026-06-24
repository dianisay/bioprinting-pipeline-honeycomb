"""End-to-end pipeline demo: synthetic multi-view images -> trajectory -> IK.

Demonstrates the complete autonomous bioprinting pipeline:
1. Generate synthetic multi-view wound images (8 views)
2. Run volumetric encoder (CT-style fusion)
3. Run polar decoder (boundary + depth + layer fill)
4. Bridge decoder output to physical coordinates
5. Plan honeycomb trajectory on cylinder surface
6. Solve inverse kinematics for the 8-DOF robot

All in-silico with synthetic data. This validates that the modules
communicate correctly end-to-end.
"""

import sys
import time
import numpy as np
import torch

sys.path.insert(0, ".")

from utils.logging_config import get_logger
from models.volumetric_encoder import VolumetricWoundEncoder3D
from models.volumetric_decoder import PolarDecoder3DLayered
from modules.wound_to_trajectory import bridge_decoder_to_planner
from modules.trajectory_planner import plan_full_trajectory
from modules.inverse_kinematics import solve_ik_scipy
from modules.robot_model import forward_kinematics_8dof, home_configuration

logger = get_logger("demo_pipeline")


def generate_synthetic_views(num_views: int = 8, image_size: int = 128) -> torch.Tensor:
    """Generate synthetic multi-view wound images for demo purposes."""
    views = torch.randn(1, num_views, 3, image_size, image_size) * 0.5 + 0.5
    views = views.clamp(0, 1)
    return views


def main():
    print("=" * 70)
    print("  AUTONOMOUS BIOPRINTING PIPELINE - END-TO-END DEMO")
    print("  (In-silico validation with synthetic data)")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    # --- Stage 1: Vision ---
    print("\n" + "-" * 50)
    print("STAGE 1: Multi-View Image Acquisition")
    print("-" * 50)
    views = generate_synthetic_views(num_views=8, image_size=128)
    print(f"  Input: {views.shape} (batch, views, C, H, W)")
    views = views.to(device)

    # --- Stage 2: Volumetric Encoder ---
    print("\n" + "-" * 50)
    print("STAGE 2: CT-Style Volumetric Encoding")
    print("-" * 50)
    t0 = time.time()
    encoder = VolumetricWoundEncoder3D(
        num_views=8, d_model=256, grid_size=8, pretrained=False,
    ).to(device)
    encoder.eval()

    with torch.no_grad():
        enc_output = encoder(views, return_voxel_grid=True)

    t_enc = time.time() - t0
    print(f"  Features: {enc_output['features'].shape}")
    if "voxel_grid" in enc_output:
        print(f"  Voxel grid: {enc_output['voxel_grid'].shape}")
    print(f"  Encoding time: {t_enc*1000:.0f}ms")

    # --- Stage 3: Polar Decoder ---
    print("\n" + "-" * 50)
    print("STAGE 3: Layer-Aware Polar Decoding")
    print("-" * 50)
    t0 = time.time()
    decoder = PolarDecoder3DLayered(
        d_model=256, num_radii=64, num_layers=4,
    ).to(device)
    decoder.eval()

    with torch.no_grad():
        pred = decoder(enc_output["features"])

    t_dec = time.time() - t0
    print(f"  Centroid: {pred['centroid'].squeeze().cpu().numpy()}")
    print(f"  Radii: mean={pred['radii'].mean():.3f}, std={pred['radii'].std():.3f}")
    print(f"  Depth: mean={pred['depth'].mean():.3f}, max={pred['depth'].max():.3f}")
    print(f"  Layer amounts: {pred['layer_amounts'].shape}")
    print(f"  Decoding time: {t_dec*1000:.0f}ms")

    # --- Stage 4: Bridge (Vision -> Robotics) ---
    print("\n" + "-" * 50)
    print("STAGE 4: Vision-to-Robotics Bridge")
    print("-" * 50)
    decoder_out = {
        "centroid": pred["centroid"].squeeze(0),
        "radii": pred["radii"].squeeze(0),
        "depth": pred["depth"].squeeze(0),
        "layer_amounts": pred["layer_amounts"].squeeze(0),
    }

    bridge_result = bridge_decoder_to_planner(
        decoder_out,
        wound_scale_mm=60.0,
        cyl_radius=50.0,
        cyl_cy=0.0,
        cyl_cz=50.0,
    )

    # Clamp extreme values from untrained model for demo feasibility
    vb = bridge_result["void_bounds"]
    vb["shell_thickness"] = min(vb["shell_thickness"], 6.0)
    vb["void_width"] = min(vb["void_width"], 40.0)
    vb["void_length"] = min(vb["void_length"], 40.0)

    print(f"  Void: {vb['void_width']:.1f} x {vb['void_length']:.1f} mm")
    print(f"  Shell thickness: {vb['shell_thickness']:.1f} mm")
    print(f"  (values clamped for untrained model demo)")

    # --- Stage 5: Trajectory Planning ---
    print("\n" + "-" * 50)
    print("STAGE 5: Honeycomb Trajectory Planning")
    print("-" * 50)
    t0 = time.time()
    traj_result = plan_full_trajectory(
        bridge_result["void_bounds"],
        bridge_result["cyl_radius"],
        bridge_result["cyl_cy"],
        bridge_result["cyl_cz"],
        optimize_tsp=True,
    )
    t_traj = time.time() - t0
    print(f"  Grid: {traj_result['nx']}x{traj_result['ny']} cells")
    print(f"  Trajectory points: {traj_result['n_points']}")
    print(f"  Planning time: {t_traj*1000:.0f}ms")

    # --- Stage 6: Inverse Kinematics ---
    print("\n" + "-" * 50)
    print("STAGE 6: 8-DOF Inverse Kinematics")
    print("-" * 50)
    n_ik_points = min(20, traj_result["n_points"])
    step = max(1, traj_result["n_points"] // n_ik_points)
    subset_positions = traj_result["traj_m"][:, ::step].T
    subset_orientations = traj_result["R_targets"][:, :, ::step].transpose(2, 0, 1)

    t0 = time.time()
    q_solutions = []
    position_errors = []
    q_prev = home_configuration()

    for i in range(len(subset_positions)):
        T_target = np.eye(4)
        T_target[:3, 3] = subset_positions[i]
        T_target[:3, :3] = subset_orientations[i]

        q_sol, pos_err = solve_ik_scipy(T_target, q_prev, max_iter=100)
        q_solutions.append(q_sol)
        position_errors.append(pos_err)
        q_prev = q_sol

    t_ik = time.time() - t0
    position_errors = np.array(position_errors)
    converged = position_errors < 0.01  # 10mm threshold

    n_solved = converged.sum()
    print(f"  IK solved: {n_solved}/{len(subset_positions)} points converged (<10mm)")
    print(f"  Mean position error: {position_errors.mean()*1000:.2f} mm")
    print(f"  IK time ({len(subset_positions)} pts): {t_ik*1000:.0f}ms")

    # --- Stage 7: Closed-Loop Printing (Simulated) ---
    print("\n" + "-" * 50)
    print("STAGE 7: Closed-Loop Printing (Depth Sensor Feedback)")
    print("-" * 50)

    from modules.depth_sensor import DepthSensorModel
    from modules.closed_loop_controller import PrintingLoopController

    sensor = DepthSensorModel(
        resolution=(64, 1),
        working_distance_mm=150.0,
        noise_sigma_base_mm=0.3,
        specular_dropout_rate=0.05,
        seed=42,
    )

    controller = PrintingLoopController(
        sensor=sensor,
        layer_height_mm=0.4,
        num_layers=4,
        correction_gain=0.7,
    )

    # Simulate wound depth (use bridge result's depth profile, clamped for demo)
    sim_depth = np.clip(bridge_result["depth_profile_mm"], 0.5, 6.0)

    t0 = time.time()
    cl_result = controller.run_full_cycle(
        true_depth_mm=sim_depth,
        predicted_depth_mm=sim_depth * 0.85,  # simulate 15% prediction error
    )
    t_cl = time.time() - t0

    print(f"  Layers deposited: {cl_result['layers_deposited']}")
    print(f"  Initial depth: {cl_result['initial_depth_mean_mm']:.2f} mm")
    print(f"  Final depth:   {cl_result['final_depth_mean_mm']:.2f} mm")
    print(f"  Fill achieved: {cl_result['fill_percentage']:.1f}%")
    print(f"  Per-layer errors: {['%.2f' % e for e in cl_result['layer_errors_mm']]}")
    print(f"  Closed-loop time: {t_cl*1000:.0f}ms")

    # --- Summary ---
    print("\n" + "=" * 70)
    print("  PIPELINE SUMMARY")
    print("=" * 70)
    print(f"  Input:          8 synthetic RGB views (128x128)")
    print(f"  Encoding:       CT-style volumetric fusion -> 256-d features")
    print(f"  Decoding:       Polar boundary + depth + 4-layer fill")
    print(f"  Wound extent:   {bridge_result['void_bounds']['void_width']:.1f} x "
          f"{bridge_result['void_bounds']['void_length']:.1f} mm")
    print(f"  Trajectory:     {traj_result['n_points']} points on cylinder")
    print(f"  IK success:     {n_solved}/{len(subset_positions)} "
          f"({100*n_solved/len(subset_positions):.0f}%)")
    print(f"  Closed-loop:    {cl_result['fill_percentage']:.0f}% fill in "
          f"{cl_result['layers_deposited']} layers")
    print(f"  Total time:     {(t_enc+t_dec+t_traj+t_ik+t_cl)*1000:.0f}ms")
    print(f"\n  Status: END-TO-END PIPELINE WITH CLOSED-LOOP FEEDBACK FUNCTIONAL")
    print("=" * 70)


if __name__ == "__main__":
    main()
