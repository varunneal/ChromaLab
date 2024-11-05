import os
import numpy as np
import packcircles
from importlib import resources

from PIL import Image, ImageDraw


class IshiharaPlate:
    def __init__(self, inside_color, outside_color, secret,
                 num_samples = 100, dot_sizes = [16,22,28], image_size = 1024,
                 directory = '.', seed = 0, lum_noise=0, noise = 0, gradient = False):
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
        self.lum_noise = lum_noise

        # self.secret = Image.open(f'{self.directory}/secretImages/{str(secret)}.png')
        with resources.path("chromalab.secretImages", f"{str(secret)}.png") as data_path:
            self.secret = Image.open(data_path)
        self.secret = self.secret.resize([self.image_size, self.image_size])
        self.secret = np.asarray(self.secret)

        self.__reset_plate()


    def generate_plate(self, seed: int = None, inside_color = None, outside_color = None):
        """
        Generate the Ishihara Plate with specified inside/outside colors and secret.
        A new seed can be specified to generate a different plate pattern.
        New inside or outside colors may be specified to recolor the plate
        without modifying the geometry.

        :param seed: A seed for RNG when creating the plate pattern.
        :param inside_color: A 6-tuple RGBOCV color.
        :param outside_color: A 6-tuple RGBOCV color.
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


    def export_plate(self, save_name: str, ext = None):
        """
        This method saves two images - RGB and OCV encoded image.

        :param save_name: Name of directory to save plate to.
        :param ext: File extension to use, such as 'png' or 'tif'.
        """
        save_dir = f'{self.directory}'
        os.makedirs(save_dir, exist_ok=True)

        # Quick change in order to get the channels in a new order
        # rgb = np.array(self.channels[0])
        # ocv = np.array(self.channels[1])

        # rgb[:, :, 1] = ocv[:, :, 1]
        # ocv[:, :, 1] = np.zeros((self.image_size, self.image_size), dtype=np.uint8)

        # Image.fromarray(rgb).save(f'{save_dir}/{save_name}-rgb.png')
        # Image.fromarray(ocv).save(f'{save_dir}/{save_name}-ocv.png')

        self.channels[0].save(f'{save_dir}/{save_name}_RGB.png')
        self.channels[1].save(f'{save_dir}/{save_name}_OCV.png')


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
            # noise apply to the six channel, scale the entire vector
            lum_noise = np.random.normal(0, self.lum_noise)
            # only apply to vector that are on
            new_color = np.clip(circle_color + (lum_noise * (circle_color > 0)), 0, 1)
            self.__draw_ellipse([x-r, y-r, x+r, y+r], new_color)
            
            # self.__draw_ellipse([x-r, y-r, x+r, y+r], circle_color)


    def __draw_ellipse(self, bounding_box, fill):
        """
        Wrapper function for PIL ImageDraw. Draws to each of the
        R, G1, G2, and B channels; each channel is represented as
        a grayscale image.

        :param bounding_box: Four points to define the bounding box.
            Sequence of either [(x0, y0), (x1, y1)] or [x0, y0, x1, y1].
        :param fill: RG1G2B tuple represented as float [0, 1].
        """
        ellipse_color = (fill * 255).astype(int)
        # for i, ch_draw in enumerate(self.channel_draws):
        #     ch_color = tuple(ellipse_color[i*3: 3*i + 3])
        #     ch_draw.ellipse(bounding_box, ch_color, width=0)
        try:
            self.channel_draws[0].ellipse(bounding_box, fill=tuple(ellipse_color[:3]), width=0)
            self.channel_draws[1].ellipse(bounding_box, fill=tuple(ellipse_color[3:]), width=0)
        except: 
            import pdb; pdb.set_trace()
            
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
        self.channels = [Image.new(mode='RGB', size=(self.image_size, self.image_size)) for _ in range(4)]
        self.channel_draws = [ImageDraw.Draw(ch) for ch in self.channels]


    def __reset_plate(self):
        """
        Reset geometry and images.
        """
        self.__reset_geometry()
        self.__reset_images()