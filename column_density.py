import numpy as np
from scipy import integrate

from astropy import units as u
from astropy.coordinates import SkyCoord,EarthLocation, Galactic, AltAz, ICRS, get_icrs_coordinates
import astropy.coordinates as coord
from astropy.time import Time

# natural units
c = 1.0 # 3 * 1e10 # cm / s
eV = 1.0
GeV = 1.0e9
MeV = 1.0e6
keV = 1.0e3
second = 1.5192669e-17 # s to eV
meter = 5.06773093741e6 # m to 1/eV
cm = 1.0e-2*meter # cm to 1/eV
kg = 5.609588e35/c**2 # kg to eV/c^2
gr = 1e-3*kg # g to eV
Na = 6.0221415e+23
hbarc = 3.16152677e-26
parsec = 3.0856776e13 # pc to km
kpc = 1e3 * parsec * 1e5 # kpc to cm
Mpc = 1e6 * parsec
M_sun = 2e30 #kg


# NGC 1068: Basic Details
ra_ngc = np.radians(40.6696292) # 2.7113056
dec_ngc = np.radians(-0.0132806) # -0.0133333
z_ngc = 0.003793
M_ngc = 2.7*1e10 * M_sun # kg (M_halo)


# =============== Part 1: Galactic contribution =================

# definition of constants
r0 =  8.127 * kpc # distance from the Sun to GC [kpc]; convert kpc to cm
rs = 26 * kpc     # scale radius [kpc]; convert kpc to cm
r_halo = 30 * kpc # maximum Halo radius [kpc]; convert kpc to cm;
rho_0 = 0.4             # local densitity [GeV/cm^-3]
gamma = 1.0             # slope parameter

# calculating scale densitity rho_s [GeV/cm^-3]
rho_s = rho_0 / ((2**(3 - gamma))/(((r0/rs)**gamma)*(1 + (r0/rs))**(3 - gamma))) # [GeV/cm^-3]

# calculating upper limit of line of sight (los)
psi = 0                 # angle between GC and los [rad]
xmax = np.sqrt(r_halo**2 - (r0**2)*(np.sin(psi)**2)) + r0*np.cos(psi) # upper limit of los integration or change to inf

# galactic center definition
gc = get_icrs_coordinates('Galactic Center')
los_ngc = SkyCoord(ra_ngc * u.rad, dec_ngc * u.rad, frame='icrs')
psi_ngc = gc.separation(los_ngc).radian

def rho_DM(x):
    '''
    Parameters
    ----------
    r: galactocentric distance [cm]

    Returns
    -------
    rho_DM: DM densitity in equatorial coordinates [GeV/cm^-3]
    '''
    # calculating DM densitity rho_DM [GeV/cm^-3] based on eq. 5
    r = np.sqrt(r0**2 - 2*x*r0*np.cos(psi_ngc) + x**2)
    rho_DM = rho_s * (2**(3 - gamma))/(((r/rs)**gamma)*(1 + (r/rs))**(3 - gamma)) # [GeV/cm^-3]
    return rho_DM

def get_t_NFW():
    """ Returns the NFW column density.

    Returns
    -------
        t: column density in GeV/cm^2
    """
    n = lambda x: rho_DM(x) # mass density
    t = integrate.quad(lambda x: n(xmax - x), 0, xmax, epsrel=1.0e-3, epsabs=1.0e-18)[0]   # GeV/cm^2
    return t

column_dens_1 = get_t_NFW()  # GeV/cm^2
print('CD #1: {:.4e} GeV / cm^2'.format(column_dens_1))


# ================= Part 2: Cosmological background ================
# cosmological constants taken from best-fit: https://arxiv.org/pdf/1303.5076.pdf
# inspo from HESE paper: https://arxiv.org/pdf/1706.05746.pdf
h = 0.75
hubble_const = 100 * h / Mpc #km s^-1 Mpc^-1 -> s^-1
# hubble_const = 67.11/Mpc #km s^-1 Mpc^-1 -> s^-1
omega_m = 0.3175
omega_lambda = 0.6825
omega_dm = 0.2685

G = 6.673*1e-11 * (1e6) / (kg/GeV) # m^3 s^-2 kg^-1 -> cm^3 s^-2 GeV^-1
crit_dens = 3*hubble_const**2 / (8*np.pi*G) # GeV / cm^3   crit_dens ~ 5.6e-6 # GeV cm^-3

hubble_param = lambda z: hubble_const * (omega_m*(1+z)**3 + omega_lambda)**(1/2) # removed radiation term
column_dens_2 = omega_dm*crit_dens * integrate.quad(lambda z: 1/hubble_param(z),0,z_ngc)[0] / (1e2) # GeV * s / cm^3 to GeV / cm^2 taking c=1
print('CD #2: {:.4e} GeV / cm^2'.format(column_dens_2))

# ================= Part 3: NGC 1068 DM Halo density ==================
v_vir = 1068 * (1e5) # km / s -> cm / s # https://www.aanda.org/articles/aa/pdf/2006/30/aa4883-06.pdf
v_rot = 310 * (1e5) # km / s -> cm / s
# v_rot = 410 * (1e5) # km / s -> cm / s

# planck_length = 1.616 * 1e-35 # m
# lambda_const = 2.888 * 1e-122 /(planck_length)**2 #
# omega_0 = 1.0 - lambda_const
omega_0 = 0.25

omega_z = lambda z: hubble_const * omega_0 * (1 + z)**3 / hubble_param(z)

# NGC 1068
r_vir = 1.63e-2 * (M_ngc*h/M_sun)**(1/3) * (omega_0/omega_z(z_ngc))**(-1/3) * (1+z_ngc)* kpc / h # cm
# M_vir = 2.7*1e10 * M_sun # kg (M_halo)
# r_vir = G*M_ngc*(kg/GeV) / (v_vir**2)  # cm

M_vir = (4*np.pi*r_vir/(kpc*3))*crit_dens*(r_vir)**3 # GeV

# Milky Way
# r_vir  = 200 * kpc # cm
# M_vir = (4*np.pi*200/3)*crit_dens*(r_vir)**3 # GeV

c_const = 11.7*(M_vir/(M_sun*1e11))**(-0.075)
delta_c = (200*c_const**3)/(3*(np.log(1+c_const)-c_const/(1+c_const)))

r_s = r_vir / c_const # cm

rho_ngc = lambda r: delta_c * crit_dens / ((r/r_s)*(1+r/r_s)**2) # GeV / cm^3
column_dens_3 = integrate.quad(lambda r: rho_ngc(r),0,np.infty, epsrel=1.0e-3, epsabs=1.0e-18)[0] # infinity or radius of galaxy
print('CD #3: {:.4e} GeV / cm^2'.format(column_dens_3))
# ================
total_col_dens = column_dens_1 + column_dens_2 + column_dens_3
print('total_col_dens = {:.4e} GeV/ cm^2'.format(total_col_dens))
