import pygame
import numpy as np
from typing import Optional
from config import EnvConfig, VisConfig


def get_env_image_as_numpy(
    env, env_config: EnvConfig, vis_config: VisConfig
) -> Optional[np.ndarray]:
    """Renders a single environment state to a NumPy array for logging."""
    img_h = 300
    aspect_ratio = (env_config.COLS * 0.75 + 0.25) / max(1, env_config.ROWS)
    img_w = int(img_h * aspect_ratio)
    if img_w <= 0 or img_h <= 0:
        return None
    try:
        temp_surf = pygame.Surface((img_w, img_h))
        cell_w_px = img_w / (env_config.COLS * 0.75 + 0.25)
        cell_h_px = img_h / max(1, env_config.ROWS)
        temp_surf.fill(vis_config.BLACK)
        if hasattr(env, "grid") and hasattr(env.grid, "triangles"):
            for r in range(env.grid.rows):
                for c in range(env.grid.cols):
                    if r < len(env.grid.triangles) and c < len(env.grid.triangles[r]):
                        t = env.grid.triangles[r][c]
                        if t.is_death:
                            continue
                        pts = t.get_points(
                            ox=0, oy=0, cw=int(cell_w_px), ch=int(cell_h_px)
                        )
                        color = vis_config.GRAY
                        if t.is_occupied:
                            color = t.color if t.color else vis_config.RED
                        pygame.draw.polygon(temp_surf, color, pts)
        img_array = pygame.surfarray.array3d(temp_surf)
        return np.transpose(img_array, (1, 0, 2))
    except Exception as e:
        print(f"Error generating environment image for TB: {e}")
        return None
