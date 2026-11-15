#!/usr/bin/env python3
"""
================================================================================
ISN-02 MARK II: OPEN-SOURCE MULTIPHYSICS & HUMAN VASCULATURE SIMULATION
Authors: Kallu Krishna Sumanth & Shiek Zaheer Khalander
Project: Theoretical Framework for Nanobots in Future Biomedical Applications
Target: IEEE Transactions on NanoBioscience / Zenodo Open Science
================================================================================
"""

import numpy as np


def generate_evans_fung_cad(
    filename="isn02_mark_ii_cad_model.obj",
    D0=1.80,
    h_dimple=0.25,
    h_rim=0.70,
    rho_rim=0.68,
    N_r=30,
    N_theta=48,
):
    """
    Exports a watertight, manifold 3D Wavefront CAD mesh (.OBJ) based on the
    Evans-Fung parametric formula calibrated to exact nanobot dimensions:
    D0 = 1.80 um, rim = 0.70 um (at rho = 0.68), dimple = 0.25 um.
    """
    R0 = D0 / 2.0

    # Analytically calibrated Evans-Fung coefficients for custom (D0, h_dimple, h_rim)
    u_m = rho_rim**2
    c0 = h_dimple / D0
    f_m = (h_rim / D0) / np.sqrt(1.0 - u_m)
    df_m = f_m / (2.0 * (1.0 - u_m))
    c2 = (df_m - (f_m - c0) / u_m) / u_m
    c1 = (f_m - c0) / u_m - c2 * u_m

    # Interior radial rings (excluding poles r=0 and shared equator r=R0)
    r_inner = np.linspace(0.0, R0, N_r + 1)[1:-1]
    theta_vals = np.linspace(0.0, 2.0 * np.pi, N_theta, endpoint=False)

    vertices = []
    # 1. Top Pole (index 1) and Bottom Pole (index 2)
    z_pole = 0.5 * D0 * c0
    vertices.append((0.0, 0.0, z_pole))
    vertices.append((0.0, 0.0, -z_pole))

    # 2. Top Shell Interior Rings
    for r in r_inner:
        rho = r / R0
        z = 0.5 * D0 * np.sqrt(max(0.0, 1.0 - rho**2)) * (c0 + c1 * (rho**2) + c2 * (rho**4))
        for theta in theta_vals:
            vertices.append((r * np.cos(theta), r * np.sin(theta), z))

    # 3. Shared Equator Ring (r = R0, z = 0)
    for theta in theta_vals:
        vertices.append((R0 * np.cos(theta), R0 * np.sin(theta), 0.0))

    # 4. Bottom Shell Interior Rings
    for r in r_inner:
        rho = r / R0
        z = -0.5 * D0 * np.sqrt(max(0.0, 1.0 - rho**2)) * (c0 + c1 * (rho**2) + c2 * (rho**4))
        for theta in theta_vals:
            vertices.append((r * np.cos(theta), r * np.sin(theta), z))

    N_inner = len(r_inner)
    top_start = 3
    eq_start = top_start + N_inner * N_theta
    bot_start = eq_start + N_theta

    faces = []
    # Top pole triangle fan (outward normal +Z)
    for j in range(N_theta):
        next_j = (j + 1) % N_theta
        faces.append((1, top_start + j, top_start + next_j))

    # Top shell quads (including connection to shared equator)
    for i in range(N_inner):
        row_curr = top_start + i * N_theta
        row_next = eq_start if (i == N_inner - 1) else (row_curr + N_theta)
        for j in range(N_theta):
            next_j = (j + 1) % N_theta
            faces.append((row_curr + j, row_next + j, row_next + next_j, row_curr + next_j))

    # Bottom pole triangle fan (reversed winding for outward normal -Z)
    for j in range(N_theta):
        next_j = (j + 1) % N_theta
        faces.append((2, bot_start + next_j, bot_start + j))

    # Bottom shell quads (reversed winding for outward normal -Z)
    for i in range(N_inner):
        row_curr = bot_start + i * N_theta
        row_next = eq_start if (i == N_inner - 1) else (row_curr + N_theta)
        for j in range(N_theta):
            next_j = (j + 1) % N_theta
            faces.append((row_curr + j, row_curr + next_j, row_next + next_j, row_next + j))

    with open(filename, "w") as f:
        f.write("# ISN-02 Mark II Neurovascular Edition - Watertight Manifold 3D CAD Mesh\n")
        f.write(f"# Calibrated Evans-Fung Geometry: D0={D0:.2f} um, rim={h_rim:.2f} um, dimple={h_dimple:.2f} um\n")
        f.write(f"# Coefficients: c0={c0:.4f}, c1={c1:.4f}, c2={c2:.4f}\n")
        for v in vertices:
            f.write(f"v {v[0]:.5f} {v[1]:.5f} {v[2]:.5f}\n")
        for face in faces:
            f.write("f " + " ".join(str(idx) for idx in face) + "\n")

    return len(vertices), len(faces)


def simulate_human_circulatory_system():
    """
    Simulates nanobot transit, Skalak strain energy (with glycocalyx-adjusted
    effective lumen & area dilation), and margination across 7 compartments.
    """
    vascular_tree = [
        {"name": "Ascending Aorta", "D_um": 25000.0, "v_mms": 300.0, "shear_rate": 96.0},
        {"name": "Major Arteries", "D_um": 4000.0, "v_mms": 120.0, "shear_rate": 240.0},
        {"name": "Terminal Arterioles", "D_um": 30.0, "v_mms": 8.0, "shear_rate": 1060.0},
        {"name": "Cerebral Microcapillaries", "D_um": 3.2, "v_mms": 0.8, "shear_rate": 1000.0},
        {"name": "Coronary Capillaries", "D_um": 4.5, "v_mms": 1.0, "shear_rate": 890.0},
        {"name": "Post-Capillary Venules", "D_um": 20.0, "v_mms": 3.5, "shear_rate": 700.0},
        {"name": "Vena Cava", "D_um": 28000.0, "v_mms": 180.0, "shear_rate": 50.0},
    ]

    D0_bot = 1.80        # um (unconstrained nanobot diameter)
    B_shear = 2.4e-5     # N/m (Skalak membrane shear modulus)
    C_area = 1.8e-3      # N/m (Skalak area dilation modulus)
    glycocalyx_um = 0.50 # um (endothelial glycocalyx + RBC train confinement thickness)

    print("\n" + "=" * 100)
    print("ISN-02 MARK II: HUMAN VASCULAR CIRCULATION & SKALAK DEFORMATION SIMULATION")
    print("=" * 100)
    print(f"{'Vascular Compartment':<26} | {'Lumen (µm)':<10} | {'Velocity':<10} | {'Profile':<10} | {'Margination & Mechanics'}")
    print("-" * 100)

    for comp in vascular_tree:
        d = comp["D_um"]
        gamma_dot = comp["shear_rate"]

        if d <= 5.0:
            d_eff = max(0.8, (d - 2.0 * glycocalyx_um) * 0.65)
            lambda_2 = min(1.0, d_eff / D0_bot)
            area_dilation = 1.0 + 0.025 * (1.0 - lambda_2)
            lambda_1 = area_dilation / max(0.1, lambda_2)
        else:
            lambda_2 = 1.0 - min(0.08, 4.5e-5 * gamma_dot)
            area_dilation = 1.0 + 0.002 * (gamma_dot / 1000.0)
            lambda_1 = area_dilation / lambda_2

        I1 = lambda_1**2 + lambda_2**2 - 2.0
        I2 = (lambda_1 * lambda_2)**2 - 1.0
        W_skalak = (B_shear / 8.0) * (I1**2 + 2.0 * I1 - 2.0 * I2) + (C_area / 8.0) * (I2**2)

        if gamma_dot > 500.0 and d > 6.0:
            status = f"Margination in CFL (Zero RBC contact, W={W_skalak:.2e} J/m²)"
        elif d <= 5.0:
            status = f"Elastic Squeeze ({D0_bot * lambda_2:.2f} µm transit, W={W_skalak:.2e} J/m²)"
        else:
            status = f"Laminar Bulk Flow (Free circulation, W={W_skalak:.2e} J/m²)"

        print(f"{comp['name']:<26} | {d:>8.1f} µm | {comp['v_mms']:>6.1f} mm/s | {D0_bot * lambda_2:>6.2f} µm | {status}")
    print("=" * 100 + "\n")


if __name__ == "__main__":
    verts, faces = generate_evans_fung_cad()
    print(f"[1] Watertight 3D CAD Mesh exported: 'isn02_mark_ii_cad_model.obj' ({verts} vertices, {faces} faces)")
    simulate_human_circulatory_system()
