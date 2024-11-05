from screeninfo import get_monitors

import numpy as np
import tkinter as tk
import time
import platform
import matplotlib.pyplot as plt
import os

from chromalab.pr650 import PR650

class MeasureSpectraDisplay:
    def __init__(self, pr650, save_directory=None, debug=False):
        self.debug = debug
        # Create the main screen window
        self.main_window = tk.Tk()
        self.main_window.title("Main Window")
        self.main_window.geometry("400x300+100+100")
        

        # Create the second fullscreen window
        self.second_window = tk.Toplevel(self.main_window)
        self.second_window.withdraw()  # Start hidden until positioned

        self.current_color = [255, 0, 0]

        # Get monitor information and setup the second window
        # self.setup_second_window()
        # self.show_second_window()
        self.save_directory_var = tk.StringVar()
        self.save_directory_var.set(save_directory)

        self.new_dir_button = tk.Button(self.main_window, text="New Directory", command=self.create_new_directory)
        self.new_dir_button.pack(pady=10)
        
        # Add buttons to the main screen window
        # self.next_button = tk.Button(self.main_window, text="Next", command=self.show_next_color)
        # self.next_button.pack(pady=10)

        self.measure_button = tk.Button(self.main_window, text="Measure", command=self.measure_spectra)
        self.measure_button.pack(pady=10)

        self.quit_button = tk.Button(self.main_window, text="Quit", command=self.quit_app)
        self.quit_button.pack(pady=10)

        # need measuremnt device
        self.pr650 = None if debug else pr650
        # result tables
        self.spectras =[]
        self.luminances=[]

        self.save_directory = save_directory
        self.save_directory_label = tk.Label(self.main_window,textvariable=self.save_directory_var)
        self.save_directory_label.pack(pady=10)

    def create_new_directory(self):
        def submit_text():
            text = entry.get()
            self.save_directory = text
            self.save_directory_var.set(text)
            os.makedirs(self.save_directory, exist_ok=True)
            print(f"New directory created: {self.save_directory}")
            entry.delete(0, tk.END)  # Clear the entry field
            window.destroy()

        window = tk.Tk()
        window.title("Directory Input")

        label = tk.Label(window, text="Enter Text:")
        label.pack(pady=10)

        entry = tk.Entry(window)
        entry.pack()

        submit_button = tk.Button(window, text="Submit", command=submit_text)
        submit_button.pack(pady=10)
    
    def setup_second_window(self):
        # Get monitor information
        monitors = get_monitors()

        first_monitor = monitors[0]
        first_x = first_monitor.x

        if len(monitors) > 1:
            second_monitor = monitors[1]  # Assuming the second monitor is at index 1
            width = second_monitor.width
            height = second_monitor.height
            x = second_monitor.x
            y = second_monitor.y
            self.second_window.geometry(f"{width}x{height}+{x}+{y}")
            
            # Set fullscreen based on platform
            current_platform = platform.system()
            if current_platform == "Windows":
                self.second_window.overrideredirect(True)
                self.second_window.state("zoomed")
                self.second_window.bind("<F11>", lambda event: self.second_window.attributes("-zoomed",
                                        not self.second_window.attributes("-zoomed")))
                self.second_window.bind("<Escape>", lambda event: self.second_window.attributes("-zoomed", False))
            elif current_platform == "Darwin":  # macOS
                self.second_window.attributes("-fullscreen", True)
                self.second_window.bind("<F11>", lambda event: self.second_window.attributes("-fullscreen",
                                        not self.second_window.attributes("-fullscreen")))
                self.second_window.bind("<Escape>", lambda event: self.second_window.attributes("-fullscreen", False))
        else:
            print("Only one monitor detected. The second window will open on the main screen.")
            self.second_window.geometry("800x600")
        self.change_background_color(self.second_window, self.current_color)

    def show_second_window(self):
        # Show the second window
        self.second_window.deiconify()

    def quit_app(self):
        def submit_text():
            text = entry.get()
            if text:
                with open(os.path.join(self.save_directory, "description.txt"), "w") as f:
                    f.write(text)
            entry.delete(0, tk.END)  # Clear the entry field
            window.destroy()

        window = tk.Tk()
        window.title("Summary of Measurements")

        label = tk.Label(window, text="Enter Text:")
        label.pack(pady=10)

        entry = tk.Entry(window)
        entry.pack()

        submit_button = tk.Button(window, text="Submit", command=submit_text)
        submit_button.pack(pady=10)        
        
        np.save(os.path.join(self.save_directory, "spectras.npy"), self.spectras)
        np.save(os.path.join(self.save_directory, "luminances.npy"), self.luminances)
        save_spectra_to_six_channel(os.path.join(self.save_directory, "spectras.npy"), os.path.join(self.save_directory, "final_spectras.csv"))
        # Close both windows
        self.second_window.destroy()
        self.main_window.destroy()

    def rgb_to_hex(self, rgb):
        # Convert RGB tuple to hex color code
        return "#{:02x}{:02x}{:02x}".format(*rgb)

    def change_background_color(self, window, rgb):
        # Change the background color of the given window using RGB input
        hex_color = self.rgb_to_hex(rgb)
        window.configure(bg=hex_color)

    def measure_spectra(self):
        if not hasattr(self, 'save_directory'):
            self.save_directory = tk.filedialog.askdirectory(title="Select Save Directory")
            if not self.save_directory:
                print("No directory selected. Measurement aborted.")
                return
            self.create_new_directory()

        if not self.debug:
            print("Measuring Spectra...")
            spectra, lum = self.pr650.measureSpectrum()
            print(f"Done Measuring, Lum: {lum}")
            self.spectras.append(spectra)
            self.luminances.append(lum)
            plot_resulting_spectras(self.spectras)

    def show_next_color(self):
        self.current_color = self.current_color[2:] + self.current_color[:2]
        print(f"Current Color Showing is {self.current_color}")
        self.change_background_color(self.second_window, self.current_color)

    def run(self):
        self.main_window.mainloop()


def plot_resulting_spectras(spectras):
    for i, spectra in enumerate(spectras):
        plt.plot(spectra[0], spectra[1], label=f'Spectra {i+1}')

    plt.xlabel('Wavelength')
    plt.ylabel('Intensity')
    plt.title('Spectra Measurements')
    plt.legend()
    plt.show()

def save_spectrum_to_csv(filename):
    spectras = np.load(filename)
    nm, power = spectras[:, 0], spectras[:, 1]
    data = np.row_stack((nm[:1], power))
    np.savetxt('spectras.csv', data.T, delimiter=',', header='Wavelength,R,O,C,V', comments='')

def load_spectrum_from_npy(filename):
    data = np.load(filename)
    nm, power = data[:, 0], data[:, 1]
    return nm, power

def save_spectra_to_six_channel(spectra_filename, save_filename):
    spectras = np.load(spectra_filename)
    lols = np.array([spectras[0, 0]] + [spectras[i, 1] for i in range(spectras.shape[0])]).T # wavelengths + 4 spectra
    # new_csv = np.insert(arr=lols, obj=[5, 5], values=0, axis=1)
    np.savetxt(save_filename, lols, delimiter=',', header='Wavelength,R,G,B,O,C,V')

# Example usage:
if __name__ == "__main__":
    # # save_spectrum_to_csv("spectras.npy")
    mac_port_name = '/dev/cu.usbserial-A104D0XS'
    pr650 = PR650(mac_port_name)
    # import pdb; pdb.set_trace() # keep trying until we connect to the PR650 - the key is the start the program and turn on the PR650 at the same time

    app = MeasureSpectraDisplay(pr650, save_directory='tmp', debug=False)
    app.run()

    # spectras = np.load('./metamer-set/spectras.npy')
    # save_spectra_to_six_channel('./metamer-set/spectras.npy', './metamer-set/final_spectras.csv')
    # plot_resulting_spectras(app.spectras)