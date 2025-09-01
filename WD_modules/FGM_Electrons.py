from scipy import constants
import numpy as np
import matplotlib.pyplot as plt

# All constants here are in cgs units

m_e = constants.m_e * 1000  # grams
m_n = constants.m_n * 1000  # grams
c = constants.c * 100  # cm/s
pi = constants.pi
hbar = constants.hbar * 1e+7  # erg seconds

e_0 = m_e**4 * c**5 / (pi**2 * hbar**3)  # erg/cm^3

# Conversion factors for natural unit usage
mass_conv = 1.783e-27  # g/MeV
momentum_conv = 5.345e-17  # g cm/s /Mev
erg_conv = 1.602e-6  # erg/Mev

# Fermi Momentum
electron_mass = m_e / mass_conv  # converts grams to MeV
k_f = np.linspace(0, 2 * electron_mass, 100)
k_f = k_f * momentum_conv  # converts MeV to g cm/s

# Dimensionless variable x
x = k_f / (m_e * c)

# number density n
n = k_f**3 / (3 * pi**2 * hbar**3)  # 1/cm^3

# Pressure
p = np.zeros(len(k_f))
p = (2 * x**3 - 3 * x) * np.sqrt(1 + x**2) + 3 * np.log(x + np.sqrt(x**2 + 1))  # dimensionless
p = p * (e_0 / 24)  # e_0 returns the necessary energy per volume units (erg/cm^3)
print(p)

# Energy Density
e = np.zeros(len(k_f))
e = n * m_n * 2  # g/cm^3
conversion = erg_conv / mass_conv
e = e * conversion

print(e)

plt.title(r"Fermi Gas Model of Electrons $erg / cm^3$")
plt.xlabel("Energy Density")
plt.ylabel("Pressure")
plt.plot(e, p, color= 'black')
plt.show()


