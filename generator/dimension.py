# ------------------------------------------------------------------
# 3‑D Fractal Spiral + Bandage / Sphere Intersection
# ------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# 1. Parameters – tweak these!
# -----------------------------
a = 1.0  # base radial scale
k = 0.1  # exponential decay rate (controls “fractal” scaling)
m = 2 * np.pi  # angular speed – one full turn per l‑unit
n = 0.2  # vertical rise per l‑unit (helix)
scale_factor = 0.5  # each nested spiral is this fraction of the previous
levels = 1  # how many nested copies to plot

# -----------------------------
# 2. Create a range of stripe lengths
# -----------------------------
L_max = 30.0  # how far we go in each direction
l_vals = np.linspace(-L_max, L_max, 500)  # dense sampling → accurate arc‑length

# -----------------------------
# 3. Build nested spirals
# -----------------------------
spiral_points = []  # list of (x, y, z) arrays for each level

scale = scale_factor ** 0
r_vals = a * scale * np.exp(-k * l_vals)  # radial distance
theta = m * l_vals  # polar angle
z_vals = n * l_vals  # vertical offset

x = r_vals * np.cos(theta)
y = r_vals * np.sin(theta)

spiral_points = zip(x, y, z_vals)

# -----------------------------
# 4. Plot the spirals + sphere
# -----------------------------
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Colour‑map for the levels
cmap = plt.cm.viridis
# colors = cmap(np.linspace(0, 1, levels))

# pp = spiral_points[-1]
pp = None
# Plot each level as a tiny scatter (s=1 keeps it crisp)
for i, p in enumerate(spiral_points):
    if i > 0:
        ax.plot3D((pp[0], p[0]), (pp[1], p[1]), (pp[2], p[2]))
    pp = p
    ax.scatter(*p, s=1, alpha=0.7)

plt.tight_layout()
plt.show()
