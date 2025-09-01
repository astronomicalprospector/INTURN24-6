import kuibit.visualize_matplotlib as viz
import matplotlib.pyplot as plt
from kuibit.simdir import SimDir
import discrete_plottings as tplot
import numpy as np
import kuibit.unitconv as uc
from scipy.interpolate import interp1d
import kuibit.grid_data as gd

#Visual plot set-up
side_length = 25
x0 = [-side_length, -side_length]
x1 = [side_length, side_length]
shape = [1000, 1000]

# Directory where images will be saved
save_dir = r"C:\Users\hunte\PycharmProjects\INTURN\INTURN\BNS Remnant Temperature Plot\Remnant Plots"

# Interpolation of cold EOS data
cold_eos_data = np.loadtxt(open('NO_PT_12_5_24.pizza', 'rt').readlines(), skiprows = 5)
cold_eos_data = cold_eos_data.T

eos_rest_mass_density = cold_eos_data[0, :]  # kg/m^3
eos_pressure = cold_eos_data[2, :]           # pascals

interpolation = interp1d(eos_rest_mass_density, eos_pressure, kind='cubic', fill_value = 'extrapolate')

# Unit Conversion step

# initialized geometrized units for 1 solar mass object
# Computational units
CU = uc.geom_umass_msun(1)

t = 0

#Looping for iterations
for i in range(5):
    sim = SimDir(f"C:/Users/hunte/PycharmProjects/INTURN/INTURN/BNS Remnant Temperature Plot/Press_Data/output-000{i}").gf
    sim_data = sim.xy

    # Assignment of simulation data to interpolate
    press = sim_data.fields.press
    # rho = sim_data.fields.rho

    # Iterations are the same for both pressure grid data and density grid data
    iterations = press.iterations

    pressure_data = press[iterations[0]]

    # density = rho[iterations[0]].get_level(5)

    for j in range(len(iterations)):
        pressure_data = press[iterations[j]]
        # density = rho[iterations[j]].get_level(5)

        total_pressure = pressure_data * CU.pressure  # Pa

        # sim_density = density * CU.density  # kg/m^3

        # cold_pressure = (interpolation(sim_density.data_xyz)).T

        # converting to UniformGridData
        # box = gd.UniformGrid(
        #     [total_pressure.shape[0], total_pressure.shape[1]],
        #     total_pressure.x0,
        #     x1=total_pressure.x1
        # )

        # cold_pressure = gd.UniformGridData(box, cold_pressure)

        # thermal_pressure = total_pressure - cold_pressure

        fig, ax = plt.subplots(figsize=(5, 5), frameon=False)

        plt.title(r'Thermal Pressure [$\mathrm{Pa}$] of Remnant at t = ' + f'{t}')

        viz.plot_color(total_pressure, x0=x0, x1=x1, shape=shape,
                       colorbar='True', vmin = 0, vmax = 4e34)

        save_path = f"{save_dir}\\iteration{t}.png"
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0, dpi=300, transparent=True)
        plt.close(fig)

        t = t + 1
