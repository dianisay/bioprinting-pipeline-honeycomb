"""Modules for the bioprinting trajectory pipeline.

- stl_analysis: STL loading, cylinder fitting, void detection
- honeycomb: Hexagonal grid generation
- conformal_mapping: UV→XYZ mapping on cylindrical surfaces
- tsp_solver: TSP cell visitation optimization via PuLP
- trajectory_planner: Full trajectory generation
- robot_model: 8-DOF FK and Jacobian
- inverse_kinematics: Numerical IK with APF + Super-Twisting
- wound_to_trajectory: Bridge from decoder output to trajectory planner
- depth_sensor: Simulated RealSense D405 + real hardware interface
- depth_fusion: Confidence-weighted fusion of predicted and measured depth
- closed_loop_controller: Scan-plan-deposit-verify-correct printing cycle
- visualization_3d: matplotlib-based 3D plotting
"""
