
import numpy as np
import os
from itertools import product

from PIL import Image, ImageDraw

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


def create_weighted_binary_combination_grid(directory, image_size, dim, weights_on_each_primary):
    """
    Creates a grid of binary combinations of the given dimension.
    """
    os.makedirs(directory, exist_ok=True)
    num_columns = dim
    num_rows = dim
    height, width = image_size
    positions, max_side_length = get_auto_sized_grid((0, 0), height, width, num_columns, num_rows)
    combinations = np.flip(np.array(list(product([1, 0], repeat=4))), axis=0).astype(float)
    for i in range(len(weights_on_each_primary)):
        combinations[:, i] = combinations[:, i] * float(weights_on_each_primary[i])
    weights = (np.clip(np.insert(arr=combinations, obj=[4, 4], values=0, axis=1), 0, 1) * 255).astype(int)
    names =['RGB', 'OCV']

    for j in range(2):
        image = Image.new('RGB', (width, height), (0, 0, 0))
        draw = ImageDraw.Draw(image)
        combinations = weights[:, 3*j:3*(j+1)]
        for i, pos in enumerate(positions):
            x, y = pos
            r = (max_side_length / 2) * 0.9
            top_left = (x - r, y - r)
            bottom_right = (x + r, y + r)
            draw.rectangle([top_left, bottom_right], fill= tuple(combinations[i]))
        image.save(f"{directory}/{names[j]}.png")

def create_binary_combination_grid(directory, image_size, dim):
    """
    Creates a grid of binary combinations of the given dimension.
    """
    os.makedirs(directory, exist_ok=True)
    num_columns = dim
    num_rows = dim
    height, width = image_size
    positions, max_side_length = get_auto_sized_grid((0, 0), height, width, num_columns, num_rows)
    combinations = np.flip(np.array(list(product([1, 0], repeat=4))), axis=0)
    weights = (np.clip(np.insert(arr=combinations, obj=[1,1], values=0, axis=1), 0, 1) * 255).astype(int)
    names =['RGB', 'OCV']

    for j in range(2):
        image = Image.new('RGB', (width, height), (0, 0, 0))
        draw = ImageDraw.Draw(image)
        combinations = weights[:, 3*j:3*(j+1)] * 255
        for i, pos in enumerate(positions):
            x, y = pos
            r = (max_side_length / 2) * 0.9
            top_left = (x - r, y - r)
            bottom_right = (x + r, y + r)
            draw.rectangle([top_left, bottom_right], fill= tuple(combinations[i]))
        image.save(f"{directory}/{names[j]}.png")


# Function to create an image gradient from black to a specified RGB color
def create_gradient(draw, lower_left, upper_right, color):
    lower_left_x, lower_left_y = lower_left
    upper_right_x, upper_right_y = upper_right
    width = int(upper_right_x - lower_left_x)
    height = int(upper_right_y - lower_left_y)
    
    for x in range(width):
        for y in range(height):
            # Calculate the gradient ratio based on the y position
            ratio = 1 - (y / height)
            # Apply the ratio to each RGB component to interpolate from black (0,0,0) to the specified color
            r = int(ratio * color[0])
            g = int(ratio * color[1])
            b = int(ratio * color[2])
            draw.point((lower_left_x + x, lower_left_y + y), fill=(r, g, b))


def create_gradient_image(directory, image_size, dim):
    os.makedirs(directory, exist_ok=True)
    num_columns = dim
    num_rows = 1
    height, width = image_size
    positions, max_side_length = get_auto_sized_grid((0, 0), height, width, num_columns, num_rows)
    weights = np.insert(arr=np.eye(4), obj=[4,4], values=0, axis=1)
    names =['RGB', 'OCV']

    for j in range(2):
        image = Image.new('RGB', (width, height), (0, 0, 0))
        draw = ImageDraw.Draw(image)
        colors = weights[:, 3*j:3*(j+1)] * 255
        for i, pos in enumerate(positions):
            if j == 1 and i > 4:
                continue # don't draw in G and B
            x, y = pos
            r = (max_side_length / 2) * 0.9
            h = (height // 2) * 0.9
            top_left = (x - r, y - h)
            bottom_right = (x + r, y + h)
            create_gradient(draw, top_left, bottom_right, colors[i])
        image.save(f"{directory}/{names[j]}.png")

def create_gradient_image_3_channel(directory, image_size, dim):
    os.makedirs(directory, exist_ok=True)
    num_columns = dim
    num_rows = 1
    height, width = image_size
    positions, max_side_length = get_auto_sized_grid((0, 0), height, width, num_columns, num_rows)
    weights = np.insert(arr=np.eye(4), obj=[4,4], values=0, axis=1)
    names =['RGB', 'OCV']

    for j in range(2):
        image = Image.new('RGB', (width, height), (0, 0, 0))
        draw = ImageDraw.Draw(image)
        colors = weights[:, 3*j:3*(j+1)] * 255
        for i, pos in enumerate(positions):
            if j == 1 and i > 4:
                continue # don't draw in G and B
            x, y = pos
            r = (max_side_length / 2) * 0.9
            h = (height // 2) * 0.9
            top_left = (x - r, y - h)
            bottom_right = (x + r, y + h)
            create_gradient(draw, top_left, bottom_right, colors[i])
        image.save(f"{directory}/{names[j]}.png")