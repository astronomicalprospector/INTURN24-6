import kuibit.visualize_matplotlib as viz
import matplotlib.pyplot as plt
from kuibit.simdir import SimDir

sim = SimDir("C:/Users/hunte/PycharmProjects/INTURN/rho_files/data_00")
sim = sim.gf.xyz
rho = sim.fields.press

print(rho.iterations)

side_length = 30
x0 = [-side_length, -side_length]
x1 = [side_length, side_length]
shape = [800, 800]

# Directory where images will be saved
save_dir = r"C:\Users\hunte\PycharmProjects\INTURN\INTURN\Cross-Section Modelling\Plots"

iteration = rho.iterations[1]

z_slice = 1

while True:
    try:
        # Get the frame and slice it
        rho_frame = rho[iteration]
        rho_slice = rho_frame.sliced([None, None, z_slice])

        # Check if the slice exists
        if rho_slice is None:
            break  # Stop if no more slices

        refined_data = rho_slice.get_level(4)
        print(f"Plotting z_slice {z_slice}")

        # Create plot
        fig, ax = plt.subplots(figsize=(5, 5), frameon=False)
        viz.plot_color(refined_data, x0=x0, x1=x1, shape=shape)

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_frame_on(False)
        plt.axis("off")

        save_path = f"{save_dir}\\output{z_slice}iteration{iteration}.png"
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0, dpi=300, transparent=True)
        plt.close(fig)

        z_slice += 1

    except Exception as e:
        break  # Stop loop if an error occurs