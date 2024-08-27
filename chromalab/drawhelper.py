import numpy as np
import polyscope as ps  # https://polyscope.run/py/
import random

z_offset = -0.2

def draw_vertex_defined_square(center_x, center_y, side, color, id):
    """
    Draw a square with colors defined at the vertices.
    """
    z = id * z_offset
    half = side / 2

    # Generate mesh geometry
    verts = np.array([
        [center_x - half, center_y + half, z],
        [center_x - half, center_y - half, z],
        [center_x + half, center_y - half, z],
        [center_x + half, center_y + half, z]
    ])

    faces = np.array([
        [0, 1, 2],
        [0, 2, 3]
    ])

    square = ps.register_surface_mesh(f'square {str(id)}', verts, faces, enabled=True)
    
    # Assign colors to mesh geometry
    if len(color.shape) == 1:
        color = np.array([color])
        color = np.tile(color, (4, 1))
    else: 
        if color.shape[0] != 4:
            raise Exception(f'A square has 4 corners, please pass in a 4xd color array!')

    if color.shape[1] == 3:
        square.set_material('flat')
        square.add_color_quantity(f'color {str(id)}', color, defined_on='vertices', enabled=True)
    elif color.shape[1] == 4:
        square.add_tetracolor_quantity(f'tetracolor {str(id)}', color, defined_on='vertices', enabled=True)
    elif color.shape[1] == 6:
        even = color[:, 0:3] 
        odd = color[:, 3:6]
        square.add_six_channel_color_quantity("vertices six channel", even, odd, defined_on='vertices', enabled=True)
    else:
        raise Exception(f'{color.shape[0]}-channel color not supported')
    return square

