"""UV → XYZ conformal mapping for cylindrical surfaces.

Maps 2D parameter-space trajectories onto the 3D cylinder surface,
including computation of surface normals for nozzle orientation.

Translates Sections 5-6 of MuffinFresa_ConformalMapping.m.
"""

import numpy as np
from typing import Tuple


def uv_to_xyz(
    traj_uv: np.ndarray,
    cyl_radius: float,
    cyl_cy: float,
    cyl_cz: float,
    u_offset: float,
    v_offset: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert UV trajectory points to 3D positions on cylinder surface.

    Convention: cylinder axis = X, curvature in Y-Z plane.
    - U = arc-length (theta * R) around circumference in Y-Z
    - V = axial coordinate along X
    - h = height above surface (positive = outside, negative = inside shell)

    Args:
        traj_uv: (3, N) trajectory in UV space [u, v, h]
        cyl_radius: cylinder radius
        cyl_cy, cyl_cz: cylinder center in YZ plane
        u_offset, v_offset: offsets to center honeycomb on void

    Returns:
        xyz: (3, N) 3D positions
        normals: (3, N) outward unit normals at each point
    """
    N = traj_uv.shape[1]
    xyz = np.zeros((3, N))
    normals = np.zeros((3, N))

    u = traj_uv[0] + u_offset
    v = traj_uv[1] + v_offset
    h = traj_uv[2]

    theta = u / cyl_radius

    # Surface point
    Sx = v
    Sy = cyl_cy + cyl_radius * np.sin(theta)
    Sz = cyl_cz + cyl_radius * np.cos(theta)

    # Outward radial normal in YZ plane
    nx = np.zeros(N)
    ny = np.sin(theta)
    nz = np.cos(theta)

    # TCP = surface + h * normal
    xyz[0] = Sx + h * nx
    xyz[1] = Sy + h * ny
    xyz[2] = Sz + h * nz

    normals[0] = nx
    normals[1] = ny
    normals[2] = nz

    return xyz, normals


def compute_nozzle_orientations(normals: np.ndarray) -> np.ndarray:
    """Compute rotation matrices for nozzle orientation at each point.

    Z_tool = -normal (nozzle points INTO the surface).
    X_tool perpendicular to Z_tool, preferring world X direction.

    Args:
        normals: (3, N) outward unit normals

    Returns:
        (3, 3, N) rotation matrices
    """
    N = normals.shape[1]
    R_targets = np.zeros((3, 3, N))

    for k in range(N):
        n = normals[:, k]
        z_tool = -n

        x_ref = np.array([1.0, 0.0, 0.0])
        if abs(np.dot(z_tool, x_ref)) > 0.99:
            x_ref = np.array([0.0, 1.0, 0.0])

        x_tool = np.cross(x_ref, z_tool)
        x_tool = x_tool / (np.linalg.norm(x_tool) + 1e-15)
        y_tool = np.cross(z_tool, x_tool)

        R_targets[:, :, k] = np.column_stack([x_tool, y_tool, z_tool])

    return R_targets


def apply_workspace_transform(
    trajectory_xyz: np.ndarray,
    normals: np.ndarray,
    z_offset: float = -0.35,
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply workspace transform: mm→m, 180°Y rotation, Z offset.

    This compensates for the robot base being inverted (pi rotation about Y).

    Args:
        trajectory_xyz: (3, N) positions in mm
        normals: (3, N) unit normals
        z_offset: vertical workspace offset in meters

    Returns:
        traj_m: (3, N) positions in meters (robot frame)
        normals_transformed: (3, N) transformed normals
    """
    traj_m = trajectory_xyz * 0.001

    # 180-degree Y rotation: negate X and Z
    traj_m[0] = -traj_m[0]
    traj_m[2] = -traj_m[2]

    normals_t = normals.copy()
    normals_t[0] = -normals_t[0]
    normals_t[2] = -normals_t[2]

    # Z workspace offset
    traj_m[2] += z_offset

    return traj_m, normals_t
