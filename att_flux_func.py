import numpy as np
from decimal import Decimal as D
from numpy import linalg as LA

logemin = 3 #log GeV
logemax = 7 #log GeV
gamma = 2.67

# natural units
GeV = 1.0e9
MeV = 1.0e6
keV = 1.0e3
meter = 5.06773093741e6
cm = 1.0e-2*meter
kg = 5.62e35
gr = 1e-3*kg
Na = 6.0221415e+23
parsec = 3.0857e16*meter
kpc = 1.0e3*parsec

# column_dens=np.load('column_dens.npy')
column_dens=1e23 # should

# ================ Cross-Sections ===================
# Enu_i: Initial Energy before scattering
# Enu_f: Final Energy after scattering
# theta: 1-x

def theta(Enu_i, Enu_f, mx):
    return mx*(1./Enu_f-1/Enu_i)

def dthetadE(Enu_f, mx):
    return mx/Enu_f**2

def dxsdE_f_scalar(Enu_i, Enu_f, g, mx, mphi):
    th = theta(Enu_i, Enu_f, mx)
    return dthetadE(Enu_f, mx)*g**2/(16*np.pi)*(th*Enu_i**2*mx)/((th*Enu_i+mx)*\
                                                              (th*Enu_i*mphi**2+\
                                                               mx*(mphi**2+2*th*Enu_i**2))**2) #corrected dsign mistake here

def SSHeavyMediator(Enu,gs,mphi,mx): #Scalar DM, Scalar mediator
    ## THIS FORMULA IS IN THE LAB FRAME (eV)
    Es=D(Enu)
    dm=D(mx)
    p=D(mphi)
    g=D(gs)
    E2=Es**D(2.0)
    m2=dm**D(2.0)
    p2=p**D(2.0)
    g2=g**D(2.0)

    t1=p2*(D(2.0)*Es+dm)
    t2=D(2.0)*Es*p2+D(4.0)*E2*dm+p2*dm
    logs=t1.ln()-t2.ln()

    num=-g2*(D(4.0)*E2*dm+t2*logs)
    den=D(64.0)*D(np.pi)*E2*m2*t2
    sig=num/den

    return float(sig)

# ================ Cascade Equation for Attenuation ===================
def get_RHS_matrices(g,mphi,mx,interaction,energy_nodes):
    NumNodes = energy_nodes.shape[0]
    # auxiliary functions
    if interaction == 'scalar':
        sigma = lambda E: SSHeavyMediator(E,g,mphi,mx)
        DiffXS = lambda Ei,Ef: dxsdE_f_scalar(Ei, Ef, g, mx, mphi) if (Ei > Ef) else 0

    sigma_array = np.array(list(map(sigma,energy_nodes))) # this is for python 3

    # matrices and arrays
    dsigmadE = np.array([[DiffXS(Ei,Ef) for Ei in energy_nodes] for Ef in energy_nodes])
    DeltaE = np.diff(np.log(energy_nodes))
    RHSMatrix = np.zeros((NumNodes,NumNodes))
    # fill in diagonal terms
    for i in range(NumNodes):
        for j in range(i+1,NumNodes):
            RHSMatrix[i][j] = DeltaE[j-1]*dsigmadE[i][j]*energy_nodes[j]**-1*energy_nodes[i]**2
    return RHSMatrix, sigma_array # in eV

def get_eigs(g,mphi,mx,interaction,logemin,logemax):
    """ Returns the eigenvalues and vectors of matrix M in eqn 6

        Args:
            mp: class for DM-nu scenario containing the coupling (mp.g), DM mass (mp.mx) in eV and mediator mass (mp.mphi) in eV
            interaction: interaction between DM and nu
            gamma: power law index of isotropic flux E^-gamma
            logemin: min nu energy log of GeV
            logemax: max nu energy log of GeV

        Returns:
            w: eigenvalues of M matrix in eV
            v: eigenvectors of M matrix in eV
            ci:coefficients in eqn 7
            energy_nodes: neutrino energy in eV
            phi0: initial neutrino flux
        """
    #Note that the solution is scaled by E^2; if you want to modify the incoming spectrum a lot, you'll need to change this here, as well as in the definition of RHS.
    NumNodes = 5 #120
    energy_nodes = np.logspace(logemin,logemax,NumNodes)*GeV # eV
    RHSMatrix, sigma_array = get_RHS_matrices(g,mphi,mx,interaction,energy_nodes)
    phi_0 = energy_nodes**(2-gamma) #eV^(2-gamma)

    w,v = LA.eig(-np.diag(sigma_array)+RHSMatrix)
    ci = LA.lstsq(v,phi_0,rcond=None)[0]
    return w,v,ci,energy_nodes


# ================ Attenuated Flux ===================

def get_att_value_theta(w, v, ci, energy_nodes,gamma,t):
    w=np.tile(w,[len(energy_nodes),1])
    phisol = np.inner(v,ci*np.exp(w.T*t).T).T*energy_nodes**(-gamma) #attenuated flux
    return phisol

def attenuated_flux(g,mphi,mx, interaction='scalar'):
    w, v, ci, energy_nodes = get_eigs(g,mphi,mx,interaction,logemin,logemax)
    flux_astro=np.ones(len(energy_nodes))
    t = column_dens # one value
    flux_astro = get_att_value_theta(w, v, ci, energy_nodes, gamma, t)
    return flux_astro

# attenuated_flux(3e-1,1e7,1e8)
