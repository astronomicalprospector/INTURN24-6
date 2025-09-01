from EOSConstruct import EOSConstruct
# from TOV import TOVSolver
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.integrate import solve_ivp


with open(f'eos.nb') as number_density:
    input_data = number_density.readlines()

baryon_number_density = input_data[2:]
baryon_number_density = np.array(baryon_number_density, dtype=float)


n_b = EOSConstruct(baryon_number_density)

energy_density, pressure = n_b.construct_eosfg()  # Returns energy density and pressure in Mev/fm^3
plt.plot(energy_density, pressure, 'black')
plt.xlabel('ϵ')
plt.ylabel('p')
plt.title(f'Pressure vs. Energy Density (MeV $fm^{{-3}}$)')

# plt.show()

# Bonus: Polytrope equation developed, for use in TOV

def db_poly_f(p, a1, a2, g1, g2):
    return (a1 * p**g1) + (a2 * p**g2)

e_0 = 939.5653**4 / (3*np.pi**2 * 197.33**3) # Constant which has units of MeV/fm^3

popt, pcov = curve_fit(db_poly_f, energy_density/e_0, pressure/e_0, p0 = [1, 2, 3/5, 1])
a1, a2, g1, g2 = popt

# finds energy density for a given pressure, DIMENSIONLESS UNITS ONLY
def poly(p):
    eps = a1 * p**g1 + a2 * p**g2
    return eps

# TOV Code Block Below (Will be changed to something else)

#constants
e_sol = e_0 * 8.968e-7  # M_sol / km^3
r0 = 1.47  # km
beta_sol = 4 * np.pi * e_sol  # 1/km^3

#Differential Equations
def dtdr(r, T):
    p, m = T
    ed = poly(p)

    if p < 0:
        return[0,0]

    dp_dr = ((-r0 * ed * m /r**2) * (1 + p/ed) *
             (1 + ((beta_sol * r**3 * p) / m)) * (1 - ((2 * r0 * m) / r))**(-1))
    dm_dr = beta_sol * ed * r**2

    #print(f"r: {r}, p: {p}, m: {m}, ed: {ed}, dp/dr: {dp_dr}, dm/dr: {dm_dr}")
    return[dp_dr, dm_dr]

def stop_condition(r, T):
    p, _ = T
    return p

stop_condition.terminal = True
stop_condition.direction = -1

# Initial conditions setup
dr = 0.1
r_max = 50

#Setting up variables
p_c = np.logspace(-6, 2, 100) # dimensionless
mass = np.zeros(len(p_c))  # dimensionless
radii = np.zeros(len(p_c))  # km

for j, n in enumerate(p_c):
    e = poly(n)
    m_init = (1 / 3) * beta_sol * (dr ** 3) * e
    p = n - ((r0 * e * m_init / dr ** 2) * (1 + n / e) *
                                 (1 + (beta_sol * dr ** 3 * n) / m_init) * (
                                             1 - (2 * r0 * m_init) / dr) ** (-1)) * dr
    T_0 = (p , m_init)
    sol = solve_ivp(dtdr, (dr, r_max), T_0,
                   events = stop_condition, dense_output=True, max_step = dr)

    radii[j] = sol.t[-1]
    mass[j] = sol.y[1][-1]


# Testing for errors!
central_pressure = 1e-3 # dimensionless quantity

e_init = poly(central_pressure)
m_init = (1/3) * beta_sol * (dr**3) * e_init
p_init = central_pressure - ((r0 * e_init * m_init /dr**2) * (1 + central_pressure/e_init) *
             (1 + (beta_sol * dr**3 * central_pressure) / m_init) * (1 - (2 * r0 * m_init) / dr)**(-1))*dr

#print(p_init, m_init)

init_cond = (p_init, m_init)
tovsol = solve_ivp(dtdr, (dr, r_max), init_cond,
                   events = stop_condition, dense_output=True, max_step = dr)

pressure_test = tovsol.y[0]
mass_test = tovsol.y[1]
radii_test = tovsol.t

plt.figure()
plt.plot(radii_test, mass_test, 'black')
plt.xlabel('Radius (km)')
plt.ylabel(r'Mass (M $\odot$)')
plt.title('Mass vs. Radius')
plt.show()

plt.figure()
plt.plot(radii_test, pressure_test, 'black')
plt.xlabel('Radius (km)')
plt.ylabel(r'Pressure')
plt.title('Pressure vs. Radius')
plt.show()

plt.figure()
plt.plot(radii, mass, 'black')
plt.xlabel('Radius (km)')
plt.ylabel(r'Mass (M $\odot$)')
plt.title('Pressure vs. Radius')
plt.show()