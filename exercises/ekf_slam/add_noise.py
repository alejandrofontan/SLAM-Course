"""
add_noise.py

Loads a perfect .g2o file and adds zero-mean Gaussian noise to the
geometric components of EDGE_SE2 and EDGE_SE2_XY edges.

EDGE_SE2   : noise on (ux, uy, delta_theta)
EDGE_SE2_XY: noise on (dx, dy)

All other lines (VERTEX_XY, VERTEX_SE2, info matrices) are passed through unchanged.
"""

import numpy as np
import argparse


def add_noise(
    input_path:  str,
    output_path: str,
    sigma_se2:   np.ndarray,   # 3x3 covariance for (ux, uy, delta_theta)
    sigma_xy:    np.ndarray,   # 2x2 covariance for (dx, dy)
    seed:        int = 42,
) -> None:
    """
    Parameters
    ----------
    input_path  : path to the perfect .g2o file
    output_path : path to write the noisy .g2o file
    sigma_se2   : (3, 3) covariance matrix for EDGE_SE2 noise
    sigma_xy    : (2, 2) covariance matrix for EDGE_SE2_XY noise
    seed        : random seed for reproducibility
    """
    rng = np.random.default_rng(seed)

    # Pre-compute Cholesky factors once
    L_se2 = np.linalg.cholesky(sigma_se2)   # noise = L @ z,  z ~ N(0, I)
    L_xy  = np.linalg.cholesky(sigma_xy)

    n_se2 = 0
    n_xy  = 0
    out_lines = []

    with open(input_path, "r") as f:
        for line in f:
            tokens = line.strip().split()
            if not tokens:
                out_lines.append(line.rstrip())
                continue

            if tokens[0] == "EDGE_SE2":
                id_from     = tokens[1]
                id_to       = tokens[2]
                ux          = float(tokens[3])
                uy          = float(tokens[4])
                delta_theta = float(tokens[5])
                info_matrix = tokens[6:]

                noise = L_se2 @ rng.standard_normal(3)
                ux          += noise[0]
                uy          += noise[1]
                delta_theta += noise[2]

                out_lines.append(
                    f"EDGE_SE2 {id_from} {id_to} "
                    f"{ux:.10f} {uy:.10f} {delta_theta:.10f} "
                    + " ".join(info_matrix)
                )
                n_se2 += 1

            elif tokens[0] == "EDGE_SE2_XY":
                pose_id     = tokens[1]
                landmark_id = tokens[2]
                dx          = float(tokens[3])
                dy          = float(tokens[4])
                info_matrix = tokens[5:]

                noise = L_xy @ rng.standard_normal(2)
                dx += noise[0]
                dy += noise[1]

                out_lines.append(
                    f"EDGE_SE2_XY {pose_id} {landmark_id} "
                    f"{dx:.10f} {dy:.10f} "
                    + " ".join(info_matrix)
                )
                n_xy += 1

            else:
                out_lines.append(line.rstrip())

    with open(output_path, "w") as f:
        f.write("\n".join(out_lines) + "\n")

    print(f"Wrote {output_path}")
    print(f"  Noisy EDGE_SE2    : {n_se2}")
    print(f"  Noisy EDGE_SE2_XY : {n_xy}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Add Gaussian noise to EDGE_SE2 and EDGE_SE2_XY in a .g2o file."
    )
    parser.add_argument("--input",  default="datasets/dataset_point_gt_extended_long2.g2o")
    parser.add_argument("--output", default="datasets/dataset_point.g2o")

    # EDGE_SE2 noise std devs
    parser.add_argument("--std-ux",    type=float, default=0.02,
                        help="Std dev for ux noise (m)")
    parser.add_argument("--std-uy",    type=float, default=0.02,
                        help="Std dev for uy noise (m)")
    parser.add_argument("--std-theta", type=float, default=0.01,
                        help="Std dev for delta_theta noise (rad)")

    # EDGE_SE2_XY noise std devs
    parser.add_argument("--std-dx", type=float, default=0.1,
                        help="Std dev for dx landmark noise (m)")
    parser.add_argument("--std-dy", type=float, default=0.1,
                        help="Std dev for dy landmark noise (m)")

    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    # Diagonal covariances (independent noise per component)
    sigma_se2 = np.diag([args.std_ux**2, args.std_uy**2, args.std_theta**2])
    sigma_xy  = np.diag([args.std_dx**2, args.std_dy**2])

    print(f"EDGE_SE2  noise  σ = diag({args.std_ux}, {args.std_uy}, {args.std_theta})")
    print(f"EDGE_SE2_XY noise σ = diag({args.std_dx}, {args.std_dy})")

    add_noise(
        input_path=args.input,
        output_path=args.output,
        sigma_se2=sigma_se2,
        sigma_xy=sigma_xy,
        seed=args.seed,
    )
