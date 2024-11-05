import numpy as np
import tetrapolyscope as ps

from chromalab.draw import configurePolyscopeFor2D, configurePolyscopeForEvenOdd, get_auto_sized_grid
from chromalab.observer import Cone, Observer, getSampledHyperCube
from chromalab.spectra import Illuminant
from chromalab.inks import get_metamer_buckets
from chromalab.maxdisplaybasis import TetraDisplayGamut
from chromalab.ishihara import IshiharaPlate
from chromalab.draw import get_auto_sized_grid

import os
from PIL import Image
from tqdm import tqdm


class ScreeningTest:
    plates = [27, 35, 39, 64, 67, 68, 72, 73, 85, 87, 89, 96]
    def __init__(self, name, wavelengths, peaks=None, noise_range=[0, 0.05, 0.1], observers=None, hypercube_sample=0.05, template='neitz', randomIndex=False) -> None:
        self.dirname = name
        self.wavelengths = wavelengths
        self.peaks = peaks
        self.noise_range = noise_range
        self.randomIndex = randomIndex
        np.random.seed(8)

        print("Generating hypercube")
        self.hypercube_sample = hypercube_sample
        self.hypercube = getSampledHyperCube(hypercube_sample, 4) # takes 40 seconds at 0.01 step, fine if we only run it once
        print("Done generating hypercube")
        self.observers = observers if observers is not None else self.__get_observers(self.peaks, template=template)
        self.metamers_per_observer = self.__get_metamers(self.observers)
        self.num_noise_levels = len(noise_range)
        self.plate_numbers = np.random.choice(ScreeningTest.plates, len(self.observers) * self.num_noise_levels, replace=True)
        # self.plate_numbers = [27, 27, 27, 27, 39, 39, 39, 39, 72, 72, 72, 72]
       

    def __get_observers(self, peaks, template="neitz"):
        obs = []
        for peak in peaks:
            l_cone = Cone.cone(int(peak[1]), wavelengths=self.wavelengths, template=template, od=0.35)
            q_cone = Cone.cone(547, wavelengths=self.wavelengths, template=template, od=0.35)
            m_cone = Cone.cone(int(peak[0]), wavelengths=self.wavelengths, template=template, od=0.35)
            s_cone = Cone.s_cone(wavelengths=self.wavelengths)
            obs += [Observer([s_cone, m_cone, q_cone, l_cone], verbose=False)]
        return obs
    

    def __get_refined_hypercube(self, metamer_led_weights, previous_step):
        outer_range = [ [metamer_led_weights[i] - previous_step, metamer_led_weights[i] + previous_step ]for i in range(4)]
        outer_range = [[round(max(0, min(1, outer_range[i][0])) * 255) / 255, 
                round(max(0, min(1, outer_range[i][1])) * 255) / 255] 
                   for i in range(4)]
        return getSampledHyperCube(0.004, 4, outer_range)
    
    def __refine_metamers(self, weights_1, weights_2, intensities):
        hypercube1 = self.__get_refined_hypercube(weights_1, self.hypercube_sample * 2)
        hypercube2 = self.__get_refined_hypercube(weights_2, self.hypercube_sample * 2)
        hypercube = np.vstack([hypercube1, hypercube2])
        return self.__get_top_metamer(intensities, hypercube, prec=0.0005, exponent=11)


    def __get_top_metamer(self, intensities, hypercube, prec=0.005, exponent=8):
        all_lms_intensities = (intensities@hypercube.T).T # multiply all possible led combinations with the intensities
        buckets = get_metamer_buckets(all_lms_intensities, axis=2, prec=prec, exponent=exponent)
        if self.randomIndex:
            random_index = np.random.randint(int(len(buckets)* 0.3))
        else:
            random_index = 0
        dst, (metamer_1, metamer_2) = buckets[random_index]
        return metamer_1, metamer_2
    

    def __get_metamers(self, observers):
        factor = 10000
        metamers_per_observer = []
        for observer in tqdm(observers):
            d = TetraDisplayGamut.loadTutenLabDisplay(observer, led_indices=[1, 3, 4, 5])
            # d = TetraDisplayGamut.loadTutenLabDisplay(observer, led_indices=[0, 3, 4, 5])
            # d = TetraDisplayGamut.loadTutenLabDisplay(observer, led_indices=[0, 1, 2, 3])
            intensities = d.primary_intensities.T * factor # columns are the ratios, also 10000 is the factor in order to get the get buckets function to work
            metamer_1, metamer_2 = self.__get_top_metamer(intensities, self.hypercube)
            weights_4tup = d.convertActivationsToIntensities(np.array([np.array(metamer_1)/factor, np.array(metamer_2)/factor]).T)
            # print("Actual first metamers, ", metamer_1, metamer_2)
            # print("Initial Metamers", (d.primary_intensities.T@weights_4tup.T).T * factor)

            metamer_1, metamer_2 = self.__refine_metamers(weights_4tup[0], weights_4tup[1], intensities)
            # print("Actual finetuned metamers, ", metamer_1, metamer_2)
            weights_4tup = d.convertActivationsToIntensities(np.array([np.array(metamer_1)/factor, np.array(metamer_2)/factor]).T)
            print("Finetuned Metamers\n", (d.primary_intensities.T@weights_4tup.T).T * factor)
            print("LED Weights\n", weights_4tup)

            # fix here as well
            weights = np.insert(arr=weights_4tup, obj=[0, 1], values=0, axis=1)
            metamers_per_observer+= [weights]
        return metamers_per_observer
    
    def __create_pseudoisochromatic_plate(self, dirname, observer_idx, noise_idx, inside_color, outside_color):
        grid_idx = len(self.observers) * noise_idx + observer_idx
        ip = IshiharaPlate(inside_color, outside_color, self.plate_numbers[grid_idx], lum_noise=self.noise_range[noise_idx], image_size=912, directory=dirname + "/images")
        ip.generate_plate()
        ip.export_plate(f"{grid_idx}")

    def create_image_grid(self, save_dir, ext, grid_size=(4, 3), canvas_size=(912, 912), padding=10):
        # load all images
        images = []
        for i in range(grid_size[0] * grid_size[1]):
            img_path = f"{save_dir}/images/{i}_{ext}"
            if os.path.exists(img_path):
                images.append(Image.open(img_path))
            else:
                raise ValueError(f"Image {img_path} not found")
            
        
        # paste images into the equally spaced, centered, grid
        image_size = (canvas_size[0] - padding, canvas_size[1] - padding)
        factor = max(np.ceil(images[0].width * grid_size[0] / image_size[0]), np.ceil(images[0].height * grid_size[1] / image_size[1]))
        grid_width = int(image_size[0] * factor)
        grid_height = int(image_size[1] * factor)
        grid_img = Image.new('RGB', (grid_width, grid_height))
        
        x_width = grid_width // grid_size[0]
        y_height = grid_height // grid_size[1]

        x_offset = x_width - images[0].width
        y_offset = y_height - images[0].height
        for idx, img in enumerate(images):
            x = idx % grid_size[0] * x_width + x_offset // 2 + padding//2
            y = idx // grid_size[0] * y_height + y_offset // 2 + padding//2
            grid_img.paste(img, (x, y))

        grid_img.resize(canvas_size)
        grid_img.save(f"{save_dir}/{save_dir}_{ext}")
        

    def create_observers_vs_noise(self):
        for i, (metamer_1, metamer_2) in enumerate(self.metamers_per_observer):
            for j in range(self.num_noise_levels):
                # insert 0 for G and B channels
                self.__create_pseudoisochromatic_plate(self.dirname, i, j, metamer_1, metamer_2)

        self.create_image_grid(self.dirname, ext='RGB.png', grid_size=(len(self.observers), self.num_noise_levels), canvas_size=(1140, 912))
        self.create_image_grid(self.dirname, ext='OCV.png', grid_size=(len(self.observers), self.num_noise_levels), canvas_size=(1140, 912))
