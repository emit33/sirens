import os
import shutil
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import RegularPolygon


def make_triangle(orientation=0.0, path=None, figsize=(2, 2), dpi=128):
    fig, ax = plt.subplots(1, figsize=figsize, dpi=dpi)

    # base_x_min, base_x_max = -np.sqrt(3) / 2, np.sqrt(3) / 2
    # base_y_min, base_y_max = -1 / 2, 1.
    #
    # ax.set_xlim(base_x_min - 1, base_x_max + 1)
    # ax.set_ylim(base_y_min - 1, base_y_max + 1)

    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)

    ax.set_aspect("equal")

    triangle = RegularPolygon(
        (0, 0), numVertices=3, radius=0.5, orientation=orientation, color="black"
    )

    ax.add_patch(triangle)
    plt.axis("off")
    plt.show()

    if path:
        fig.savefig(path)


def make_triangles(dir, n=10, **kwargs):
    if os.path.exists(dir):
        shutil.rmtree(dir)

    os.makedirs(dir)
    for i in range(n):
        path = Path(dir) / f"triangle{i}.png"
        make_triangle(i * 2 * np.pi / n, path, **kwargs)


if __name__ == "__main__":
    make_triangles("triangles", figsize=(2, 2), dpi=128)
