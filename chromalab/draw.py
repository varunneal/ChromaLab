import numpy as np
import polyscope as ps  # https://polyscope.run/py/
import random
import os
import packcircles
from importlib import resources

from PIL import Image, ImageDraw

z_offset = -0.2

def draw_vertex_defined_square(center_x, center_y, side, color, id):
    """
    Draw a square with colors defined at the vertices.
    """
    z = id*0.01 * z_offset
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
    square.set_material('flat')
    
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
        square.add_six_channel_color_quantity(f"vertices six channel {str(id)}", even, odd, defined_on='vertices', enabled=True)
    else:
        raise Exception(f'{color.shape[0]}-channel color not supported')
    return square

def draw_rectangle(center_x, center_y, width, height, color, id):
    """
    Draw a square with colors defined at the vertices.
    """
    z = id*0.01 * z_offset

    # Generate mesh geometry
    verts = np.array([
        [center_x - width/2, center_y + height/2, z],
        [center_x - width/2, center_y - height/2, z],
        [center_x + width/2, center_y - height/2, z],
        [center_x + width/2, center_y + height/2, z]
    ])

    faces = np.array([
        [0, 1, 2],
        [0, 2, 3]
    ])

    square = ps.register_surface_mesh(f'square {str(id)}', verts, faces, enabled=True)
    square.set_material('flat')
    
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
        square.add_six_channel_color_quantity(f"vertices six channel {str(id)}", even, odd, defined_on='vertices', enabled=True)
    else:
        raise Exception(f'{color.shape[0]}-channel color not supported')
    return square


def draw_circle_helper(center_x, center_y, radius, color, id):
    """
    Draw circles defined by the center coordinates, radius, and d-tuple color
    """
    # Generate mesh geometry
    n_faces = 100
    verts, faces = [], []
    z = id * z_offset  # Small offset to prevent shapes from clipping into each other

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

    name = f'circle {str(id)}'
    circle = ps.register_surface_mesh(name, verts, faces, enabled=True)
    circle.set_material('flat')

    color = np.tile(color, (circle.n_faces(), 1))
    
    if color.shape[1] == 3:
        circle.add_color_quantity(f'color {str(id)}', color, defined_on='faces', enabled=True)
    elif color.shape[1] == 4:
        circle.add_tetracolor_quantity(f'tetracolor {str(id)}', color, defined_on='faces', enabled=True)
    elif color.shape[1] == 6:
        even = color[:, 0:3] 
        odd = color[:, 3:6]
        circle.add_six_channel_color_quantity(f"vertices six channel {str(id)}", even, odd, defined_on='faces', enabled=True)
    else:
        raise Exception(f'{color.shape[0]}-channel color not supported')

    return circle, name

def _create_grid_indices(height, width, num_columns, num_rows):
    grid_indices = []
    step_x = width / num_columns
    step_y = height / num_rows
    step = min(step_x, step_y)
    half_step = step / 2
    
    for i in range(num_rows):
        for j in range(num_columns):
            x = j * step + half_step
            y = i * step + half_step
            grid_indices.append([x, y])
    
    offset = ((width - num_columns*step)/2, (height - num_rows * step)/2)
    return grid_indices, step, offset

def get_auto_sized_grid(lower_left, height, width, num_columns, num_rows, padding_percentage=0.1):
    padding = width * padding_percentage
    lower_left_subimage = lower_left + np.array([padding, padding])
    subimage_h, subimage_w = height - 2 * padding, width - 2 * padding
    positions, max_side_length, offset = _create_grid_indices(subimage_h, subimage_w, num_columns, num_rows)
    positions = lower_left_subimage + offset + positions
    return positions, max_side_length


class IshiharaPlate:
    def __init__(self, inside_color, outside_color, secret,
                 num_samples = 100, dot_sizes = [16,22,28], image_size = 1024,
                 directory = '.', seed = 0, noise = 0, gradient = False):
        """
        :param inside_color:  RG1G2B sequence of length 4; channels can be
                              specified as a float in [0, 1] or int [0, 255]

        :param outside_color: RG1G2B sequence of length 4; channels can be
                              specified as a float in [0, 1] or int [0, 255]

        :param secret:  May be either a string or integer, specifies which
                        secret file to use from the secrets directory.
        """
        self.inside_color = self.__standardize_color(inside_color)
        self.outside_color = self.__standardize_color(outside_color)

        self.num_samples = num_samples
        self.dot_sizes = dot_sizes
        self.image_size = image_size
        self.directory = directory
        self.seed = seed
        self.noise = noise
        self.gradient = gradient

        with resources.path("chromalab.secretImages", f"{str(secret)}.png") as data_path:
            self.secret = Image.open(data_path)
        self.secret = self.secret.transpose(Image.FLIP_TOP_BOTTOM)
        self.secret = self.secret.resize([self.image_size, self.image_size])
        self.secret = np.asarray(self.secret)

        self.object_names = []
        if not ps.is_initialized():
            ps.init()
        configurePolyscopeFor2D(self.image_size, self.image_size)

        self.__reset_plate()


    def generate_plate(self, seed: int = None, inside_color = None, outside_color = None):
        """
        Generate the Ishihara Plate with specified inside/outside colors and secret.
        A new seed can be specified to generate a different plate pattern.
        New inside or outside colors may be specified to recolor the plate
        without modifying the geometry.

        :param seed: A seed for RNG when creating the plate pattern.
        :param inside_color: A 4-tuple RG1G2B color.
        :param outside_color: A 4-tuple RG1G2B color.
        """
        def helper_generate():
            self.__generate_geometry()
            self.__compute_inside_outside()
            self.__draw_plate()

        if inside_color:
            self.inside_color = self.__standardize_color(inside_color)
        if outside_color:
            self.outside_color = self.__standardize_color(outside_color)

        # Plate doesn't exist; set seed and colors and generate whole plate.
        if self.circles is None:
            self.seed = seed or self.seed
            helper_generate()
            return

        # Need to generate new geometry and re-color.
        if seed and seed != self.seed:
            self.seed = seed
            self.__reset_plate()
            helper_generate()
            return

        # Need to re-color, but don't need to re-generate geometry.
        if not seed and (inside_color or outside_color):
            self.__reset_images()
            self.__draw_plate()
            return


    def __standardize_color(self, color):
        """
        :param color: Convert color to np.array and ensure it is a float in [0, 1].
        """
        color = np.asarray(color)
        if np.issubdtype(color.dtype, np.integer):
            color = color.astype(float) / 255.0
        return color


    def __generate_geometry(self):
        """
        :return output_circles: List of [x, y, r] sequences, where (x, y)
                                are the center coordinates of a circle and r
                                is the radius.
        """
        np.random.seed(self.seed)

        # Create packed_circles, a list of (x, y, r) tuples.
        radii = self.dot_sizes * 2000
        np.random.shuffle(radii)
        packed_circles = packcircles.pack(radii)

        # Generate output_circles.
        center = self.image_size // 2
        output_circles = []

        for (x, y, radius) in packed_circles:
            if np.sqrt((x - center) ** 2 + (y - center) ** 2) < center * 0.95:
                r = radius - np.random.randint(2, 5)
                output_circles.append([x,y,r])

        self.circles = output_circles


    def __compute_inside_outside(self):
        """
        For each circle, estimate the proportion of its area that is inside or outside.
        Take num_sample point samples within each circle, generated by rejection sampling.
        """
        # Inside corresponds to numbers; outside corresponds to background
        outside = np.int32(np.sum(self.secret == 255, -1) == 4)
        inside  = np.int32((self.secret[:,:,3] == 255)) - outside

        inside_props = []
        outside_props = []
        n = np.random.rand(len(self.circles))

        for i, [x, y, r] in enumerate(self.circles):
            x, y = int(round(x)), int(round(y))
            inside_count, outside_count = 0, 0

            for _ in range(self.num_samples):
                while True:
                    dx = np.random.uniform(-r, r)
                    dy = np.random.uniform(-r, r)
                    if (dx**2 + dy**2) <= r**2:
                        break

                x_grid = int(np.clip(np.round(x + dx), 0, self.image_size - 1))
                y_grid = int(np.clip(np.round(y + dy), 0, self.image_size - 1))
                if inside[y_grid, x_grid]:
                    inside_count += 1
                elif outside[y_grid, x_grid]:
                    outside_count += 1

            in_p  = np.clip(inside_count  / self.num_samples * (1 - (n[i] * self.noise / 100)), 0, 1)
            out_p = np.clip(outside_count / self.num_samples * (1 - (n[i] * self.noise / 100)), 0, 1)

            inside_props.append(in_p)
            outside_props.append(out_p)

        self.inside_props = inside_props
        self.outside_props = outside_props


    def __draw_plate(self):
        """
        Using generated geometry data and computed inside/outside proportions,
        draw the plate.
        """
        assert None not in [self.circles, self.inside_props, self.outside_props]

        for i, [x, y, r] in enumerate(self.circles):
            in_p, out_p = self.inside_props[i], self.outside_props[i]
            if not self.gradient:
                in_p = 1 if in_p > 0.5 else 0
                out_p = 1 - in_p

            circle_color = in_p * self.inside_color + out_p * self.outside_color
            # self.__draw_ellipse([x-r, y-r, x+r, y+r], circle_color)
            item, name = draw_circle_helper(x, y, r, circle_color, i)
            self.object_names.append(name)


    def __reset_geometry(self):
        """
        Reset plate geometry. Useful if we want to regenerate the plate pattern
        with a different seed.
        """
        self.circles = None
        self.inside_props = None
        self.outside_props = None


    def __reset_images(self):
        """
        Reset plate images. Useful if we want to regenerate the plate with
        different inside/outside colors.
        """
        [ps.remove_surface_mesh(name) for name in self.object_names]
        self.object_names = []
        


    def __reset_plate(self):
        """
        Reset geometry and images.
        """
        self.__reset_geometry()
        self.__reset_images()



def configurePolyscopeFor2D(im_w, im_h):
    if not ps.is_initialized(): 
        ps.init()
    # basic parameters
    ps.set_window_resizable(True)
    ps.set_window_size(im_w, im_h)
    center = [im_w/2, im_h/2]
    fov = 90

    # trig to determine z distance given a particular fov and image height
    z_dist = im_h / (2 * np.tan(np.radians(fov/2)))

    # create camera parameters
    intrinsics = ps.CameraIntrinsics(fov_vertical_deg=fov, aspect=im_w/im_h)
    extrinsics = ps.CameraExtrinsics(root=(center[0], center[1], z_dist), look_dir=(0, 0, -1), up_dir=(0.,1.,0.))
    params = ps.CameraParameters(intrinsics, extrinsics)
    ps.set_view_camera_parameters(params)

    # set the viewport view to be orthographic & 2d
    ps.set_navigation_style("planar")
    ps.set_view_projection_mode("orthographic")

    # set the scaling so the orthographic view is correct for the image
    ps.set_automatically_compute_scene_extents(False)
    # refer to view.cpp:456 in polyscope repository for the scaling required to get the correct ratio of pixels
    ps.set_length_scale(im_h/2/2)
    low = np.array((0,0, -1.))
    high = np.array((im_w, im_h, 1))
    ps.set_bounding_box(low, high)

def configurePolyscopeForEvenOdd(fps=60, target_sleep=95, vsync=False):
    if not ps.is_initialized(): 
        ps.init()
    ps.set_ground_plane_mode("none")
    ps.set_background_color([0, 0, 0])

    # FPS for a full frame (one even and one odd frame)
    ps.set_max_fps(fps)
    # May need to adjust this number in the range of (0, 100) to hit target FPS on a specific devide
    # If the frame rate is lower than the target, decrease the target_sleep value
    ps.set_target_sleep(target_sleep)
    ps.set_enable_vsync(vsync)

    # Set always redraw to True (or else polyscope tries to optimize and won't draw even-odd correctly)
    ps.set_always_redraw(True)

    # Enable the Even-Odd gui panel
    ps.set_build_even_odd_gui_panel(True)