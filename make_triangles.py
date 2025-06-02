import os
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import RegularPolygon



def make_triangle(orientation = 0, path = None):
    fig, ax = plt.subplots(1)

    # base_x_min, base_x_max = -np.sqrt(3) / 2, np.sqrt(3) / 2
    # base_y_min, base_y_max = -1 / 2, 1.
    #
    # ax.set_xlim(base_x_min - 1, base_x_max + 1)
    # ax.set_ylim(base_y_min - 1, base_y_max + 1)

    ax.set_xlim(-1,1)
    ax.set_ylim(-1,1)

    ax.set_aspect('equal')

    triangle = RegularPolygon((0, 0),
                                numVertices=3,
                                radius=0.5,
                                orientation=orientation,
                                color = 'black')

    ax.add_patch(triangle)
    plt.axis('off')
    plt.show()

    if path:
        fig.savefig(path)


if __name__ == '__main__':
    os.makedirs('images', exist_ok=True)

    make_triangle(path='images/triangle1.png')
    make_triangle(orientation=np.pi/2, path='images/triangle2.png')

