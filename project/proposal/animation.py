import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle
from shader import compute_points, compute_intensities, NUM_I, T_MAX, compute_global_bbox

# Setup output directory
os.makedirs('.cache/frames', exist_ok=True)

GRID_RES = int(400)

# Animation settings
FPS = 24
DURATION = T_MAX  # seconds
TOTAL_FRAMES = int(FPS * DURATION)

# Get bounding box for visualization
bbox_min_x, bbox_max_x, bbox_min_y, bbox_max_y = compute_global_bbox()
print(f"Animation bbox: [{bbox_min_x:.3f}, {bbox_max_x:.3f}] x [{bbox_min_y:.3f}, {bbox_max_y:.3f}]")

print("Setting up grid...")
xx, yy = np.meshgrid(np.linspace(-1.0, 1.0, GRID_RES), np.linspace(-1.0, 1.0, GRID_RES))
grid_xy = np.stack([xx.ravel(), yy.ravel()], axis=1)

print(f"Pre-computing points for {TOTAL_FRAMES} frames (T_MAX={T_MAX})...")
all_points_array = np.zeros((TOTAL_FRAMES, NUM_I, 2))
for frame_num in range(TOTAL_FRAMES):
    t = (frame_num / TOTAL_FRAMES) * T_MAX
    if frame_num % 10 == 0 or frame_num == TOTAL_FRAMES - 1:
        print(f"  Frame {frame_num + 1}/{TOTAL_FRAMES}, t={t:.2f}")
    px, py = compute_points(t)
    all_points_array[frame_num, :, 0] = px
    all_points_array[frame_num, :, 1] = py

print("Creating figure...")
fig, ax = plt.subplots(figsize=(8, 8))
ax.set_facecolor('black')
im = ax.imshow(
    np.zeros((GRID_RES, GRID_RES)),
    cmap='gray', vmin=0, vmax=1, animated=True,
    extent=[-1.0, 1.0, -1.0, 1.0], origin='lower'
)

# Draw red bounding box
rect = Rectangle((bbox_min_x, bbox_min_y), 
                 bbox_max_x - bbox_min_x, 
                 bbox_max_y - bbox_min_y,
                 linewidth=2, edgecolor='red', facecolor='none')
ax.add_patch(rect)

ax.axis('on')
ax.set_title('Shader Animation (t=0.00)')
ax.set_xlabel('x')
ax.set_ylabel('y')

def update(frame_num):
    t = (frame_num / TOTAL_FRAMES) * T_MAX
    print(f"Frame {frame_num + 1}/{TOTAL_FRAMES}, t={t:.2f}")

    points_norm = all_points_array[frame_num]
    intensities = compute_intensities(points_norm, grid_xy)

    frame = intensities.reshape(GRID_RES, GRID_RES)
    im.set_array(frame)
    ax.set_title(f'Shader Animation (t={t:.2f}, frame {frame_num + 1}/{TOTAL_FRAMES})')
    return [im]

print("Starting animation...")
anim = animation.FuncAnimation(fig, update, frames=TOTAL_FRAMES, interval=1000//FPS, 
                                blit=True, repeat=False)

print("Saving video...")
output_path = '.cache/frames/shader_animation.mp4'
anim.save(output_path, fps=FPS, extra_args=['-vcodec', 'libx264'])
print(f"Animation saved to {output_path}")

plt.show()

