from astropy.cosmology import FlatLambdaCDM
import pandas as pd
import numpy as np
from scipy.interpolate import griddata
from hmf import MassFunction, cosmo

def interp(logM, z):
    '''interpolates dndlog10m from input logM and z using hmf mass 
    functions'''
    return griddata(np.array([arr_logM, arr_z]).T, dndlog10m, [logM, z])

def dN_dlog10Mdz(logM, z):
    dvdz = cosmo.WMAP9.differential_comoving_volume(z).value
    dN_dlog10MdV = interp(logM, z)
    dN_dlog10Mdz = dN_dlog10MdV * dvdz
    return np.log(dN_dlog10Mdz)

def mcmc(steps, burn_in, start_point):
    n = steps+burn_in
    
    x_cur = start_point
    x_next = np.random.uniform(low=[Mmin, zmin], high=[Mmax, zmax], size=(n,2))
    u = np.random.uniform(size=n)
    
    post_cur = dN_dlog10Mdz(x_cur[0], x_cur[1])
    
    posterior = np.zeros((n, 2), dtype=np.ndarray)

    for ii in range(n):
        x_prop = x_next[ii]
        post_prop = dN_dlog10Mdz(x_prop[0], x_prop[1])
        acceptance = min(post_prop/post_cur,1)
        if u[ii] <= acceptance:
            x_cur = x_prop
            post_cur = post_prop
        
        posterior[ii] = x_cur
    
    return posterior[burn_in:]

# generate mass functions at z=0.0, 0.25, 0.5, 0.75 and 1.00
# using masses within R500
mf = MassFunction(z=0.0, cosmo_model=cosmo.WMAP9, Mmin=13.5, Mmax=15.8, mdef_model="SOCritical", mdef_params={"overdensity": 500})
mf2 = MassFunction(z=0.25, cosmo_model=cosmo.WMAP9, Mmin=13.5, Mmax=15.8, mdef_model="SOCritical", mdef_params={"overdensity": 500})
mf3 = MassFunction(z=0.5, cosmo_model=cosmo.WMAP9, Mmin=13.5, Mmax=15.8, mdef_model="SOCritical", mdef_params={"overdensity": 500})
mf4 = MassFunction(z=0.75, cosmo_model=cosmo.WMAP9, Mmin=13.5, Mmax=15.8, mdef_model="SOCritical", mdef_params={"overdensity": 500})
mf5 = MassFunction(z=1.0, cosmo_model=cosmo.WMAP9, Mmin=13.5, Mmax=15.8, mdef_model="SOCritical", mdef_params={"overdensity": 500})

# load cosmological parameters
H = cosmo.WMAP9.H0.value
Om0 = cosmo.WMAP9.Om0
h = H/100

# concatenate M, z and dndlog10m arrays to feed into scipy's griddata function for 2d interpolation
arr_logM = np.concatenate((np.log10(mf.m/h), np.log10(mf2.m/h), np.log10(mf3.m/h), np.log10(mf4.m/h), np.log10(mf5.m/h)))
arr_z = np.concatenate(([0.0 for ii in np.array(mf.m)], [0.25 for ii in np.array(mf2.m)], [0.5 for ii in np.array(mf3.m)], [0.75 for ii in np.array(mf4.m)], [1.0 for ii in np.array(mf5.m)]))
dndlog10m = np.concatenate((mf.dndlog10m*h**3, mf2.dndlog10m*h**3, mf3.dndlog10m*h**3, mf4.dndlog10m*h**3, mf5.dndlog10m*h**3))

# set mass and redshift range for sampling
Mmin = 13.5 - np.log10(h)
Mmax = 15.8 - np.log10(h)
zmin = 0.05
zmax = 1.0

mcmc = mcmc(100000, 10000, [14.7, 0.5])
df = pd.DataFrame(data={'logm': mcmc.T[0], 'z': mcmc.T[1]})
df.to_csv('clusters.csv', index=False)

#mcmc = pd.read_csv('clusters_py.csv')