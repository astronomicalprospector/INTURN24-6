import numpy as np
import matplotlib.pyplot as plt

# Number density imported as neutron number density
with open(f'eos.nb') as number_density:
    input_data = number_density.readlines()

baryon_number_density = input_data[2:]
baryon_number_density = np.array(baryon_number_density, dtype=float)

# Establishing constants (MeV)
m_e = 0.511
m_p = 938.272
m_n = 939.56

# Setting neutron momentum range
k_fn = (3 * np.pi**2 * baryon_number_density)**(1/3) * 197.33

# k_F for protons function (MeV)
k_fp = (np.sqrt((k_fn**2 + m_n**2 - m_e**2)**2 - 2 * m_p**2 * (k_fn**2 + m_n**2 + m_e**2) + m_p**4)
        / (2 * np.sqrt(k_fn**2 + m_n**2)))

#charge neutrality
k_fe = k_fp

#Relativistic Parameters
x_p = k_fp/m_p
x_e = k_fe/m_e
x_n = k_fn/m_n

# Solving Pressure and Energy Density
e_0 = m_n**4 / (3 * np.pi**2)  # MeV^4

def pressure_func(x):
    return x * (2 * x**2 + 1) * np.sqrt(1 + x**2) - np.arcsinh(x)

def edensity_func(x):
    return x * (2 * x**2 - 3) * np.sqrt(1 + x**2) + 3 * np.arcsinh(x)

pressure = np.zeros(len(x_n))
energy_density = np.zeros(len(x_n))

# Building total pressure and energy density

param = np.vstack([x_p, x_n, x_e])

n = 0
while n < 2:
    x = param[n, :]
    p = np.zeros(len(x_n))
    ed = np.zeros(len(x_n))

    for j, i in enumerate(x):
        p[j] = pressure_func(i)
        ed[j] = edensity_func(i)

    pressure = pressure + p
    energy_density = energy_density + ed
    n += 1

# Adding dimension to pressure and energy density
pressure = e_0 * pressure / (197.33)**3
energy_density = e_0 * energy_density / (197.33)**3

plt.title(r"Neutron Star Equation of State $MeV/fm^3$")
plt.xlabel(r"$\epsilon$")
plt.ylabel(r"$P$")
plt.plot(energy_density, pressure, color = 'black')
plt.show()

# Back into MeV^4
pressure = pressure * (197.33)**3
energy_density = energy_density * (197.33)**3

# Converting to erg/cm^3
pressure = pressure / (197.33e-13)**3 * 1.602e-6
energy_density = energy_density/ (197.33e-13)**3 * 1.602e-6

plt.title(r"Neutron Star Equation of State $ergs/cm^3$")
plt.xlabel(r"$\epsilon$")
plt.ylabel(r"$P$")
plt.plot(energy_density, pressure, color = 'red')
plt.show()
