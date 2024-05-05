import os
from PIL import Image
import numpy as np
import h5py
import plot_utils as pu

class InitialConditions:
    def __init__(self, image_path, N, MASS=1, R_CM=[0, 0, 0], V_CM=[0, 0, 0], invert=True):
        """
        Initializes the class with image path, particle number N, default mass, center of mass position and velocity, and an invert flag. Loads image, processes to grayscale, normalizes, optionally inverts, and samples initial positions and colors from the image data, including a z-coordinate based on image intensity.
        """
        self.image_path = image_path
        self.N = int(N)
        self.MASS = MASS
        self.R_CM = np.array(R_CM)
        self.V_CM = np.array(V_CM)

        img_grey = Image.open(image_path).convert("L")
        color_array = np.array(Image.open(image_path))[::-1, :]

        self.img_array = np.array(img_grey)[::-1, :]
        self.img_array = self.img_array / np.sum(self.img_array)  #normalising the sum to 1
        if invert:
            self.img_array = np.max(self.img_array) - self.img_array

        self.cum_sum = np.cumsum(self.img_array.ravel())
        self.cum_sum /= self.cum_sum[-1]

        x, y, z = self._sample_positions()
        self.colors = color_array[y, x] / 255

        self.POS = np.array([x, y, z]).T
        self.POS = self.POS / np.max(self.POS.max(axis=0))
        self.POS = self.POS - np.mean(self.POS, axis=0) 
        self.POS = self.POS + self.R_CM

        self.VEL = np.zeros((self.N, 3))
        self.M = self.MASS * np.ones(self.N) / self.N

    def _sample_positions(self):
        random_values = np.random.rand(self.N)
        max_index = len(self.cum_sum) - 1
        indices = np.searchsorted(self.cum_sum, random_values)
        indices = np.clip(indices, 0, max_index)
        y, x = np.unravel_index(indices, self.img_array.shape)
        x, y = np.array(x), np.array(y)
        z = self.img_array[y, x]  #used normalized image intensity as depth

        z = np.interp(z, (z.min(), z.max()), (0, 10)) #exxagerated z axis
        z = np.power(z, 3)
        return x, y, z

    def set_circular_velocity(self, factor=1):
        radii = np.linalg.norm(self.POS - self.R_CM, axis=1)
        velocity_magnitude = factor * np.sqrt(self.MASS / radii)
        velocity_direction = np.cross(self.POS - self.R_CM, [0, 0, 1])  #making z-axis as rotation axis
        velocity_direction = velocity_direction / np.linalg.norm(velocity_direction, axis=1)[:, np.newaxis]
        self.VEL = velocity_magnitude[:, np.newaxis] * velocity_direction

    def generate_ic_file(self, savepath):
        savefold = "/".join(savepath.split("/")[:-1])
        if not os.path.exists(savefold):
            os.makedirs(savefold)

        with h5py.File(savepath, "w") as f:
            header_grp = f.create_group("Header")
            header_grp.attrs["Dimensions"] = 3
            header_grp.attrs["N"] = self.N

            header_grp.attrs["RCM"] = self.R_CM
            header_grp.attrs["VCM"] = self.V_CM

            part_type_grp = f.create_group("Bodies")
            part_type_grp.create_dataset("Positions", data=self.POS)
            part_type_grp.create_dataset("Velocities", data=self.VEL)
            part_type_grp.create_dataset("Masses", data=self.M)

            if self.colors is not None:
                part_type_grp.create_dataset("Colors", data=self.colors)

    def set_plot(self, marker_size=0.1, facecolor="#fffff0", ax_color="k", lim=0.65, greyscale=False, ax_spines=True, show=True):
        if greyscale:
            colors = 'grey'
        else:
            colors = self.colors

        if show:
            Fig = pu.Figure(fig_size=540) 
            fs = Fig.fs
            Fig.facecolor = facecolor
            Fig.ax_color = ax_color

            ax = Fig.get_axes(projection='3d')
            ax.scatter(self.POS[:, 0], self.POS[:, 1], self.POS[:, 2], lw=0, s=fs * marker_size, c=colors)

            ax.set_xlim([-lim, lim])
            ax.set_ylim([-lim, lim])
            ax.set_zlim([-lim, lim])
            ax.set_aspect('auto')

            Fig.show()
