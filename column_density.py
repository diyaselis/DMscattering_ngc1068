import numpy as np
from scipy import integrate

# natural units
GeV = 1.0e9
MeV = 1.0e6
keV = 1.0e3
second = 1.5192669e-15 # s to eV
meter = 5.06773093741e6 # m to eV (hbar c)
cm = 1.0e-2*meter # cm to eV
km = 1.0e3*meter # km to eV
kg = 5.62e35 # kg to eV
gr = 1e-3*kg  # kg to eV
Na = 6.0221415e+23
parsec = 3.085678e13 # pc to km
kpc = 1.0e3*parsec # kpc to km
Mpc = 1.0e6*parsec

M_sun = 2e30 #kg

# NGC 1068: Basic Details
ra_ngc = np.radians(40.6696292) # 2.7113056
dec_ngc = np.radians(-0.0132806) # -0.0133333
z_ngc = 0.003793
M_ngc = 1e8 * M_sun
# =============== Part 1: Galactic contribution =================


# Values based on Iooco paper https://arxiv.org/pdf/1901.02460.pdf
RS = 26*(kpc*1e5)  # kpc -> cm
gamma=1 # standard NFW
rho_s=1e7*(M_sun*1e3)/(kpc*1e5)**3 # 1e7 M sun/kpc^3 -> g/cm^3
R=8.3*(kpc*1e5) # Distance from sun to galactic centre kpc -> cm


def get_angle(ra,dec):
    # ra radians from 0 to 2pi
    # dec radians from -pi/2 to pi/2
    dec_G=27.12825*np.pi/180
    ra_G=192.85948*np.pi/180
    l_NCP=122.93192*np.pi/180

    b = np.arcsin(np.sin(dec)*np.sin(dec_G)+np.cos(dec)*np.cos(dec_G)*np.cos(ra-ra_G))
    if dec<0:
        b += np.pi

    l = l_NCP-np.arcsin(np.cos(dec)*np.sin(ra-ra_G)/np.cos(b))
    theta=np.arccos(np.cos(b)*np.cos(l))
    return theta


def rho_NFW(theta, x):
    """ Returns the NFW density in gr/cm^3.

    Args:
	theta: angle to galactic centre (radians)
	x: distance from Earth to source (km)
	r: distance from galactic centre to source (km)

    Returns:
        rho: density in gr/cm^3
    """
    r = np.sqrt(x ** 2 + R ** 2 - 2 * x * R * np.cos(theta))
    rho=rho_s/((r/RS)**gamma*(1+r/RS)**(3-gamma))
    return rho


def get_t_NFW(ra,dec):
    """ Returns the NFW column density for a given zenith angle.

    Args:
        theta: zenith angle in radians.

    Returns:
        t: column density in g/cm^2
    """
    # x: distance from Earth to source

    xmax =  RS #np.sqrt((RS - d) ** 2 * np.cos(theta) ** 2 + d * (2 * RS - d)) - (RS - d) * np.cos(theta)
    theta = get_angle(ra,dec)

    n = lambda x: rho_NFW(theta, x)  # mass density
    t=integrate.quad(lambda x: n(xmax - x), 0, xmax, epsrel=1.0e-3, epsabs=1.0e-18)[0] # g/cm^2
    return t

column_dens_1 = get_t_NFW(ra_ngc,dec_ngc) * gr * Na /cm**2  # g/ cm^2 -> eV^3
print(column_dens_1)


# ================= Part 2: Cosmological background ================
# cosmological constants taken from best-fit: https://arxiv.org/pdf/1303.5076.pdf
# inspo from HESE paper: https://arxiv.org/pdf/1706.05746.pdf
hubble_const = 67.11/Mpc #km s^-1 Mpc^-1 -> s^-1
omega_m = 0.3175
omega_lambda = 0.6825
omega_dm = 0.2685
crit_dens = 5.6e-6 # GeV cm^-3
G = 6.67*1e-20 # km^3 s^-2 kg^-1
crit_dens = 3*hubble_const**2 / (8*np.pi*G) * gr * Na /km**3 # kg km^-3 -> eV^4

hubble_param = lambda z: hubble_const * (omega_m*(1+z)**3 + omega_lambda)**(1/2) # removed radiation term
column_dens_2 = omega_dm*crit_dens * integrate.quad(lambda z: 1/hubble_param(z),0,np.infty)[0] / second # eV^4 s^-1 to eV^3
print(column_dens_2)

# ================= Part 3: NGC 1068 DM Halo density ==================

# h =1
# omega_0 = 1.0
# omega_z0 = lambda z: 1 + z
# r_vir = 1.63e-2 * (M_ngc*h/M_sun)**(1/3) * (omega_0/omega_z0(z_ngc))**(-1/3) * (1+z_ngc)
# M_vir = (4*np.pi*200/3)*crit_dens_ngc*r_vir**3
#
# c = 11.7*(M_vir/(M_sun*1e11))**(-0.075)
# delta_c = (200*c**3)/(3*(np.log(1+c)-c/(1+c)))
#
# r_s = (3*M_vir/(4*np.pi*200*crit_dens_ngc))**(1/3) / c
# rho_ngc = lambda r: delta_c * crit_dens_ngc / ((r/r_s)*(1+r/r_s)**2)
#
# column_dens_3 = integrate.quad(lambda r: rho_ngc,0,np.infty)[0] # infinity or radius of galaxy
# print(column_dens_3)
column_dens_3 = 0.

# ================
total_col_dens = column_dens_1 + column_dens_2 + column_dens_3
print(total_col_dens)
