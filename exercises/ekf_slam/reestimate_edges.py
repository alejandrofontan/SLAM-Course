import numpy as np
import matplotlib.pyplot as plt

# ── File paths ─────────────────────────────────────────────────────────────────
G2O_FILE_IN  = "datasets/dataset_point.g2o"
G2O_FILE_OUT = "datasets/dataset_point_gt.g2o"

# ── Parse the .g2o file ────────────────────────────────────────────────────────
vertices_xy  = {}   # landmark id  -> (x, y)
vertices_se2 = {}   # robot pose id -> (x, y, theta)
lines_raw    = []   # all original lines, in order

with open(G2O_FILE_IN, "r") as f:
    for line in f:
        lines_raw.append(line.rstrip())
        tokens = line.strip().split()
        if not tokens:
            continue
        if tokens[0] == "VERTEX_XY":
            vid = int(tokens[1])
            vertices_xy[vid] = (float(tokens[2]), float(tokens[3]))
        elif tokens[0] == "VERTEX_SE2":
            vid = int(tokens[1])
            vertices_se2[vid] = (float(tokens[2]), float(tokens[3]), float(tokens[4]))

# ── Reestimate edges ───────────────────────────────────────────────────────────
# EDGE_SE2   from_id to_id  ux uy delta_theta  <info matrix>
#   ux, uy      = displacement in the robot's LOCAL frame at time t
#   delta_theta = change in heading
#
#   Given perfect poses p_i=(x_i,y_i,θ_i) and p_j=(x_j,y_j,θ_j):
#     dx_world = x_j - x_i
#     dy_world = y_j - y_i
#     ux = cos(θ_i)*dx_world + sin(θ_i)*dy_world   (rotate into body frame)
#     uy = -sin(θ_i)*dx_world + cos(θ_i)*dy_world
#     delta_theta = θ_j - θ_i
#
# EDGE_SE2_XY  pose_id  landmark_id  dx dy  <info matrix>
#   dx = lx - rx   (world-frame x distance, landmark minus robot)
#   dy = ly - ry   (world-frame y distance, landmark minus robot)

out_lines = []

for line in lines_raw:
    tokens = line.strip().split()
    if not tokens:
        out_lines.append(line)
        continue

    if tokens[0] == "EDGE_SE2":
        id_from     = int(tokens[1])
        id_to       = int(tokens[2])
        info_matrix = tokens[6:]          # keep original info matrix

        xi, yi, ti = vertices_se2[id_from]
        xj, yj, tj = vertices_se2[id_to]

        dx_world = xj - xi
        dy_world = yj - yi

        ux          =  np.cos(ti) * dx_world + np.sin(ti) * dy_world
        uy          = -np.sin(ti) * dx_world + np.cos(ti) * dy_world
        delta_theta = tj - ti

        new_line = (f"EDGE_SE2 {id_from} {id_to} "
                    f"{ux:.10f} {uy:.10f} {delta_theta:.10f} "
                    + " ".join(info_matrix))
        out_lines.append(new_line)

    elif tokens[0] == "EDGE_SE2_XY":
        pose_id     = int(tokens[1])
        landmark_id = int(tokens[2])
        info_matrix = tokens[5:]          # keep original info matrix

        rx, ry, rt = vertices_se2[pose_id]
        lx, ly     = vertices_xy[landmark_id]

        # Rotate world-frame delta into robot body frame (consistent with EKF)
        dx_world = lx - rx
        dy_world = ly - ry
        dx =  np.cos(rt) * dx_world + np.sin(rt) * dy_world
        dy = -np.sin(rt) * dx_world + np.cos(rt) * dy_world

        new_line = (f"EDGE_SE2_XY {pose_id} {landmark_id} "
                    f"{dx:.10f} {dy:.10f} "
                    + " ".join(info_matrix))
        out_lines.append(new_line)

    else:
        out_lines.append(line)

# ── Write output file ──────────────────────────────────────────────────────────
with open(G2O_FILE_OUT, "w") as f:
    f.write("\n".join(out_lines) + "\n")

print(f"Saved reestimated file → {G2O_FILE_OUT}")

# ── Build trajectories for plotting ───────────────────────────────────────────
sorted_ids  = sorted(vertices_se2.keys())
pose_xs     = [vertices_se2[i][0] for i in sorted_ids]
pose_ys     = [vertices_se2[i][1] for i in sorted_ids]
pose_thetas = [vertices_se2[i][2] for i in sorted_ids]

# Dead-reckoning from the reestimated EDGE_SE2
# Re-parse the output file for clean edges
edges_new = {}
with open(G2O_FILE_OUT, "r") as f:
    for line in f:
        tokens = line.strip().split()
        if tokens and tokens[0] == "EDGE_SE2":
            id_from     = int(tokens[1])
            id_to       = int(tokens[2])
            ux          = float(tokens[3])
            uy          = float(tokens[4])
            delta_theta = float(tokens[5])
            edges_new[id_from] = (id_to, ux, uy, delta_theta)

start_id = sorted_ids[0]
x, y, theta = vertices_se2[start_id]
dr_xs     = [x]
dr_ys     = [y]
dr_thetas = [theta]

current_id = start_id
while current_id in edges_new:
    id_to, ux, uy, delta_theta = edges_new[current_id]
    x     = x + np.cos(theta)*ux - np.sin(theta)*uy
    y     = y + np.sin(theta)*ux + np.cos(theta)*uy
    theta = theta + delta_theta
    dr_xs.append(x)
    dr_ys.append(y)
    dr_thetas.append(theta)
    current_id = id_to

# ── Plot ───────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 9))
fig.suptitle("Reestimated Edges — Ground-Truth Overlay", fontsize=14, fontweight="bold")

def draw_traj(ax, xs, ys, thetas, label, color, arrow_color, arrow_every=20):
    xs, ys = np.array(xs), np.array(ys)
    ax.plot(xs, ys, color=color, linewidth=2, zorder=2, label=label)
    ax.scatter(xs[0],  ys[0],  c="green", s=90, zorder=6)
    ax.scatter(xs[-1], ys[-1], c="red",   s=90, zorder=6)
    for i in range(0, len(xs), arrow_every):
        dx = 0.25 * np.cos(thetas[i])
        dy = 0.25 * np.sin(thetas[i])
        ax.annotate("", xy=(xs[i]+dx, ys[i]+dy), xytext=(xs[i], ys[i]),
                    arrowprops=dict(arrowstyle="->", color=arrow_color, lw=1.2),
                    zorder=4)

# Robot trajectories
draw_traj(ax, pose_xs, pose_ys, pose_thetas,
          label=f"VERTEX_SE2 (ground truth, n={len(pose_xs)})",
          color="steelblue", arrow_color="navy")

draw_traj(ax, dr_xs, dr_ys, dr_thetas,
          label=f"Dead-reckoning from reestimated EDGE_SE2 (n={len(dr_xs)})",
          color="darkorange", arrow_color="saddlebrown")

# Landmarks
lx = [vertices_xy[i][0] for i in vertices_xy]
ly = [vertices_xy[i][1] for i in vertices_xy]
ax.scatter(lx, ly, c="purple", marker="^", s=100, zorder=5, label="VERTEX_XY landmarks")
for lid, (lxv, lyv) in vertices_xy.items():
    ax.annotate(str(lid), (lxv, lyv), textcoords="offset points",
                xytext=(5, 5), fontsize=7, color="purple")

# Shared start/end in legend
ax.scatter([], [], c="green", s=80, label="Start")
ax.scatter([], [], c="red",   s=80, label="End")

ax.set_xlabel("x  [m]")
ax.set_ylabel("y  [m]")
ax.set_aspect("equal")
ax.grid(True, linestyle="--", alpha=0.5)
ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig("reestimated_trajectory.png", dpi=150, bbox_inches="tight")
print("Plot saved → reestimated_trajectory.png")
plt.show()
