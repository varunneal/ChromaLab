from tqdm import tqdm
from enum import Enum
from typing import List, Optional
import numpy.typing as npt

import numpy as np

from .observer import Observer
from .spectra import Spectra, Illuminant

class LEDS(Enum):
    RED = 0 # 660nm
    GREEN = 1 # 550nm
    BLUE = 2 # 451nm
    YELLOW = 3 # 590nm
    CYAN = 4 # 510nm
    SBLUE = 5 # 410nm

LED_WAVELENGTHS = [660, 550, 451, 590, 510, 410]

class LEDSequence:
    HDMI_BANDWIDTH = 24

    def __init__(self, leds: List[LEDS], sequence: List[LEDS], observer: Optional[Observer]=None):
        self.primaries = leds
        if len(leds) > 6: 
            raise ValueError("Only up to 6 LEDs are supported")
        elif len(leds) != len(set(leds)):
            raise ValueError("LEDs Must Be Unique")
        elif len(sequence) % 2 != 0 and len(sequence) % 3 != 0:
            raise ValueError("Only supporting k that divides 24")
        self.k = len(leds)
        self.b = self.HDMI_BANDWIDTH // self.k

        if len(sequence) != self.HDMI_BANDWIDTH:# check if 24 bits are used
            raise ValueError("Sequence Must be 24 Bits")
        elif not np.all(np.isin(sequence, self.primaries)):
            raise ValueError("Sequence Must Only Contain LEDs in primary array")
        
        self.seq_idx = []
        curr_leds = [0] * len(leds)
        for i in range(24):
            led_idx = sequence[i].value
            self.seq_idx.append(curr_leds[led_idx])
            curr_leds[led_idx]+=1
        self.sequence = sequence
        if observer is not None:
            self.observer = observer
            if len(leds) != observer.dimension:
                raise NotImplementedError("Observer Dimensionality Must Match LED Count")
            self.primary_intensities = observer.get_wavelength_sensitivity(np.array([LED_WAVELENGTHS[v.value] for v in leds]))

    @staticmethod
    def TetrachromaticDisplay(observer=None):
        seq = [LEDS.RED, LEDS.GREEN, LEDS.BLUE, LEDS.YELLOW]
        return LEDSequence(seq, seq * 6, observer)

    # getters
    def get_primaries(self):
        return self.primaries

    def get_sequence(self):
        return self.sequence
    
    def activations_to_intensities(self, activations: npt.ArrayLike):
        return np.matmul(np.linalg.inv(self.primary_intensities), activations)
    
    def encode_intensities_to_seq(self, intensities: npt.ArrayLike):
        if type(intensities) != np.ndarray and len(intensities) == self.k:
            intensities = np.array([intensities])
        elif type(intensities) == np.ndarray and intensities.shape[1] != self.k:
            raise ValueError("Intensities Need to Be Same Length as Primaries")
        elif not np.logical_and(np.all(intensities >=0), np.all(intensities <= 1)):
            raise ValueError("Intensities Need to be Between 0 and 1")

        num_colors = intensities.shape[0]
        casted_intensities = np.array(intensities * (2 **self.b - 1), dtype=np.uint8) # discretization here, ask will or ren what's acceptable?
        casted_intensities = np.unpackbits(casted_intensities, axis=1).reshape(num_colors, self.k, 8)
        bin_seq = np.zeros((num_colors, 24), dtype=bool)
        for i in range(24):
            led_type = self.sequence[i].value
            led_idx = self.seq_idx[i]
            bin_seq[:, i] = casted_intensities[:, led_type, (8 - self.b) + led_idx]
        bin_seq = bin_seq.reshape(num_colors, 3, 8)
        encoded_rgb = np.packbits(bin_seq, axis=2).reshape(num_colors, 3)
        return encoded_rgb
    
    def decode_seq_to_intensities(self, encoded_rgb: npt.ArrayLike):
        if type(encoded_rgb) != np.ndarray and len(encoded_rgb) == 3:
            encoded_rgb = np.array([encoded_rgb], dtype=np.uint8)
        elif type(encoded_rgb) != np.ndarray and encoded_rgb.shape[0] != 3:
            raise ValueError("Encoded RGB must be 24 bits")
        elif type(encoded_rgb) == np.ndarray and encoded_rgb.shape[1] != 3:
            raise ValueError("Encoded RGB Must be 24 bits")
        
        if encoded_rgb.dtype != np.uint8:
            raise ValueError("Encoded RGB Must be a uint8")
        
        num_colors = encoded_rgb.shape[0]
        casted_rgb = np.unpackbits(encoded_rgb, axis=1)
        intensities = np.zeros((num_colors, self.k, 8), dtype=bool)
        for i in range(24):
            led_type = self.sequence[i].value
            led_idx = self.seq_idx[i]
            intensities[:, led_type, (8 - self.b) + led_idx] = casted_rgb[:, i]
        intensities = np.packbits(intensities, axis=2).reshape(num_colors, self.k)
        return intensities/(2 **self.b - 1)