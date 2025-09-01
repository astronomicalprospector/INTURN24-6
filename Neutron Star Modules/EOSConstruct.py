import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator

class EOSConstruct:

    # MeV (for all fermion masses)
    M_E = 0.511
    M_N = 939.5653
    M_P = 938.2720

    def __init__(self, nb):
        nb = np.array(nb)
        self.nb = nb  # fm^-1

        # dimension factor
        self.e_0 = self.M_N**4 / (3*np.pi**2)  # MeV^4

        # energy density and pressure arrays
        self.energy_density = np.zeros(len(self.nb))
        self.pressure = np.zeros(len(self.nb))

    def construct_eosfg(self):
        # fermi momentum of neutron based on nb
        k_fn = (3*np.pi**2 * self.nb)**(1/3) * 197.33  # MeV

        # proton fermi momentum determined by weak interaction equilibrium or beta equilibrium
        k_fp = (np.sqrt((k_fn ** 2 + self.M_N ** 2 - self.M_E ** 2) ** 2 - 2 * self.M_P ** 2 *
                        (k_fn ** 2 + self.M_N ** 2 + self.M_E ** 2) + self.M_P ** 4)
                        / (2 * np.sqrt(k_fn ** 2 + self.M_N ** 2)))

        # Charge neutrality makes any instance of electron fermi momentum equal to proton fermi momentum

        # setting parameters
        # All x and y parameters are dimensionless quantities
        x_p = k_fp / self.M_P
        x_n = k_fn / self.M_N
        x_e = k_fp / self.M_E

        # Energy Density and Pressure solving

        def pressure_func(x):
            return x * (2 * x ** 2 + 1) * np.sqrt(1 + x ** 2) - np.arcsinh(x)

        def edensity_func(x):
            return x * (2 * x ** 2 - 3) * np.sqrt(1 + x ** 2) + 3 * np.arcsinh(x)

        # stacking parameter x to loop over for each fermion
        x = np.vstack([x_p, x_n, x_e])

        n = 0
        while n < 2:
            param = x[n, :]
            p = np.zeros(len(x_n))
            ed = np.zeros(len(x_n))

            for j, i in enumerate(param):
                p[j] = pressure_func(i)
                ed[j] = edensity_func(i)

            self.energy_density = self.energy_density + ed
            self.pressure = self.pressure + p
            n += 1

        # MeV/fm^3
        energy_density = self.energy_density * self.e_0 / 197.33**3
        pressure = self.pressure * self.e_0 / 197.33**3

        return energy_density, pressure

    def getED(self, p, energy_density, pressure):
        eosinterp = PchipInterpolator(energy_density, pressure)
        return eosinterp(p)
