import matplotlib.colors as mcolors
import numpy as np
import polyscope as ps  # https://polyscope.run/py/
import random

def draw_circle_helper(center_x, center_y, radius, color, id):
    # Generate mesh geometry
    n_faces = 100
    verts, faces = [], []
    z = id * -0.1   # Small offset to prevent shapes from clipping into each other

    verts.append([center_x, center_y, z])
    thetas = np.linspace(0, 2 * np.pi, num=n_faces + 1)
    for theta in thetas[:-1]:
        x = radius * np.cos(theta) + center_x
        y = radius * np.sin(theta) + center_y
        verts.append([x, y, z])
    verts = np.array(verts)

    for i in range(n_faces - 1):
        faces.append([0, i+1, i+2])
    faces.append([0, n_faces, 1])
    faces = np.array(faces)

    circle = ps.register_surface_mesh(f'circle {str(id)}', verts, faces, enabled=True)
    
    # Assign colors to mesh geometry
    if color.shape[0] == 3:
        circle.set_material('flat')
        circle.set_color(color)
    elif color.shape[0] == 4:
        values = np.tile(color, (circle.n_faces(), 1))
        circle.set_material('flat_tetra')
        circle.add_tetracolor_quantity(f'tetracolor {str(id)}', values, defined_on='faces', enabled=True)
    else:
        raise Exception(f'{color.shape[0]}-channel color not supported')
        
    return circle

def draw_square_helper(center_x, center_y, side, color, id):
    z = id * -0.1
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
    if color.shape[0] == 3:
        square.set_material('flat')
        square.set_color(color)
    elif color.shape[0] == 4:
        values = np.tile(color, (square.n_faces(), 1))
        square.set_material('flat_tetra')
        square.add_tetracolor_quantity(f'tetracolor {str(id)}', values, defined_on='faces', enabled=True)
    else:
        raise Exception(f'{color.shape[0]}-channel color not supported')
    return square


def dead_leaves(sigma, color_palette, filename, res=512, 
                max_iters=30, shape_mode='mixed', tetra_save_mode='LMS_Q'):
    """
    https://github.com/mbaradad/learning_with_noise/blob/main/generate_datasets/dead_leaves/generate_dataset.py
    
    sigma           -- determines distribution of radiis
    color_palette   -- a numpy array of colors, of shape (n, 3) or (n, 4)
    res             -- image resolution
    max_iters       -- number of shapes to draw
    shape_mode      -- whether to draw circles, squares, or a mixture
    tetra_save_mode -- if rasterizing a tetracolor scene, mode to save the output
        'LMS_Q' : saves first 3 channels in one RGB .png file, saves 4th channel in one grayscale .png file
        'RG1G2B' : saves 4 color channels in one RGBA .png file
        'four_gray' : saves each channel to a separate grayscale .png file
    """
    r_min = 0.05
    r_max = 0.4

    # Compute distribution of radiis (exponential distribution with lambda = sigma)
    k = 200
    r_list = np.linspace(r_min, r_max, k)
    r_dist = 1.0 / (r_list ** sigma)
    if sigma > 0:
        # Normalize
        r_dist = r_dist - (1 / r_max ** sigma)
    r_dist = np.cumsum(r_dist)
    # Normalize so that cumulative sum is 1
    r_dist = r_dist / r_dist.max()

    available_shapes = ['circle', 'square']
    assert shape_mode in available_shapes or shape_mode == 'mixed'
    for i in range(max_iters):
        if shape_mode == 'mixed':
            shape = random.choice(available_shapes)
        else:
            shape = shape_mode
        
        # TODO: sample a color randomly from color_palette
        color_index = np.random.choice(color_palette.shape[0])
        color = color_palette[color_index]

        r_p = np.random.uniform(0, 1)
        r_i = np.argmin(np.abs(r_dist - r_p))
        radius = max(int(r_list[r_i] * res), 1)
        
        center_x, center_y = np.array(np.random.uniform(0, res, size=2), dtype='int32')
        if shape == 'circle':
            draw_circle_helper(center_x, center_y, radius, color, i)
        elif shape == 'square':
            side = radius * np.sqrt(2)
            draw_square_helper(center_x, center_y, side, color, i)
        else:
            raise Exception(f'Got unsupported shape mode {shape}')

    n_ch = color_palette[0].shape[0]
    if n_ch == 3:
        ps.screenshot(filename)
    elif n_ch == 4:
        ps.rasterize_tetra(filename, tetra_save_mode)
