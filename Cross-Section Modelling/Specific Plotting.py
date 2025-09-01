from skimage.color import xyz2lab

import discrete_density_plot as dens
import matplotlib.pyplot as plt
from kuibit.simdir import SimDir
import kuibit.unitconv as uc
import numpy as np

# initialized geometrized units for 1 solar mass object
# Computational units
CU = uc.geom_umass_msun(1)

#nuclear saturation density
saturation = 0.16 #fm^-3

#neutron mass in grams
m_N = 1.675e-24  # grams
number_density_conv = m_N * (1e39)  #g fm^3

side_length = 20
x0 = [-side_length, -side_length]
x1 = [side_length, side_length]
shape = [800, 800]

# Empty array of maximum densities with time steps
maximum_densities = []
time_steps = []
n = 0

# Directory where images will be saved
save_dir = r"C:\Users\hunte\PycharmProjects\INTURN\INTURN\Cross-Section Modelling\Plots"

for i in range(5):

    sim = SimDir(f"C:/Users/hunte/PycharmProjects/INTURN/rho_files/data_0{i}")
    sim = sim.gf.xyz
    rho = sim.fields.press

    for iter in rho.iterations:
        rho_frame = rho[iter]
        rho_slice = rho_frame.sliced([None, None, 1])
        refined_data = rho_slice.get_level(4)

        #unit conversion to n/n_0
        refined_data = refined_data * CU.density / 1000  # g/cm^3
        number_density_data = (refined_data / number_density_conv)  # 1/fm^3
        number_density_data = number_density_data / saturation  # n/n_0
        density = refined_data.data_xyz
        saturation_values = number_density_data.data_xyz


        maximum = max(max(arr) for arr in number_density_data)
        maximum_densities.append(maximum)
        time_steps.append(n)


        # Create plot
        fig, ax = plt.subplots(figsize=(5, 5), frameon=False)
        dens.plot_contourf(number_density_data, x0=x0, x1=x1, shape=shape,
                           colorbar = "True")

        plt.axis("off")

        save_path = f"{save_dir}\\iteration{n}.png"
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0, dpi=300, transparent=True)
        plt.close(fig)
        print(f"{iter} plot produced")


        n = n + 1



max_density = np.array(maximum_densities)
iterations = np.array(time_steps)

plt.plot(iterations, max_density)
plt.show()
