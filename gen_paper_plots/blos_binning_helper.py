import os
import numpy as np
import time
import subprocess
from astropy.io import fits
from skimage.transform import resize
from skimage.transform import downscale_local_mean
import scipy.integrate as spi
import pymilos as pym
import pymilos_5250 as pym5
import copy

def load_data(folder: str,angle: str,snapshot: str,default_dir: str ="/path/to/spinor_files") -> np.ndarray:
    """load stokes profiles and reorder

    Parameters
    ----------
    folder: str
        path to folder
    angle: str
        angle of the inclination
    snapshot: str
        snapshot number of the MURaM simulation
    default_dir: str
        default path to the folders

    Returns
    -------
    data: numpy ndarray
        Stokes profiles [y,x,stokes,wavelength] in the order IQUV
    """
    data = fits.getdata(f"{default_dir}/{folder}/{snapshot}/6173_masi_theta{angle}/inverted_profs.1.fits")
    return data[:,:,[0,2,3,1],:] #SPINOR creates IVQU by default


def parse_angle(angle:str) -> float:
    """parse the angle string to a float (in degrees)

    Parameters
    ----------
    angle: str
        angle of the inclination
    snapshot: str
        snapshot number of the MURaM simulation
    default_dir: str
        default path to the folders

    Returns
    -------
    _ : float
       angle (in degrees) in a float
    """
    if 'n' in angle or '-' in angle:
        angle = angle[1:]
    else:
        pass
    return float(angle.replace('_','.'))


def get_inst_shape(inst, angle, remainder_mode, dshape=288, pixel_res=20.833333):
    """calculate the shape of the data for a given instrument resolution

    Parameters
    ----------
    inst: str
        instrument name
    angle: str
        angle of the inclination
    remainder_mode: str
        mode for rounding non-integer pixel counts, can be 'floor', 'ceil' or 'round'
    dshape: int
        default shape of the data (288) of the MURaM simulation
    pixel_res: float
        pixel resolution of the original MURaM simulation (20.833333) km/pixel

    Returns
    -------
    _ : tuple
        shape of the data for the given instrument resolution
    """
    ang = parse_angle(angle)
    physical_y = dshape*np.cos(ang/180*np.pi)
    if inst == 'MURaM':
        fac = 1
    elif inst == 'HRT':
        fac = 101.5/pixel_res
    elif inst == 'FDT':
        fac = 761.5/pixel_res
    elif inst == 'HMI':
        fac = 362.5/pixel_res
    elif inst == 'onepix':
        new_x = 1
        physical_y = 1.0
        fac = 1.0
    if remainder_mode == 'floor':
        new_x = np.floor(dshape/fac)
        inst_y = np.floor(physical_y/fac)
    elif remainder_mode == 'ceil':
        new_x = np.ceil(dshape/fac)
        inst_y = np.ceil(physical_y/fac)
    elif remainder_mode == 'round':
        new_x = round(dshape/fac,0)
        inst_y = round(physical_y/fac,0)
    if inst_y < 1.0:
        inst_y = 1.0
    print("Remainder mode: ", remainder_mode)
    print("inst_y: ", inst_y)
    print("inst_x: :", new_x)
    return (int(inst_y),int(new_x),4,251)


def interp_data_to_nearest_divisible_integer(data, new_shape):
    """interpolate the data to the nearest integer divisible by the desired new shape

    Parameters
    ----------
    data: numpy ndarray
        Stokes profiles [y,x,stokes,wavelength] in the order IQUV
    new_shape: tuple
        shape of the data for the given instrument resolution
    
    Returns
    -------
    _ : numpy ndarray
        interpolated data
    """
    inst_y = new_shape[0]
    inst_x = new_shape[1]
    dshape = data.shape[0]
    if dshape % inst_y == 0 and dshape % inst_x == 0:
        return data
    else:
        div = round(dshape/inst_y, 0)
        interp_y = int(div*inst_y)
        print("interp_y: ", interp_y)
        
        divx = round(dshape/inst_x, 0)
        interp_x = int(divx*inst_x)
        print("interp_x: ", interp_x)
        
        if interp_y < dshape or interp_x < dshape:
            anti_aa = True
        else:
            anti_aa = False
        return resize(data, (interp_y, interp_x, new_shape[2], new_shape[3]), anti_aliasing=anti_aa)

    
def rebin(a, shape, downscale_mean=False):
    """Rebin an array to a new shape.

    Parameters
    ----------
    a : array_like
        Array to be rebinned.
    shape : tuple
        Shape of the new array.
    downscale_mean : bool, optional
        If True, then downscale the array using skimage.transform.downscale_local_mean
        (default is False and uses numpy).
    
    Returns
    -------
    _ : numpy ndarray
        rebinned array
    """
    if not downscale_mean:
        sh = shape[0],a.shape[0]//shape[0],shape[1],a.shape[1]//shape[1],shape[2],shape[3]
        return a.reshape(sh).mean(-3).mean(1)
    else:
        factors = (a.shape[0]//shape[0],a.shape[1]//shape[1],1,1)
        return downscale_local_mean(a, (factors))
    

def run_cmilos(data,wave_axis,rte,options=[6,15],out_dir='./', loc='cmilos/', synthesis=0):
    """RTE inversion using CMILOS

    Parameters
    ----------
    data: numpy ndarray
        Stokes profiles [y,x,stokes,wavelength] in the order IQUV
    wave_axis: numpy ndarray
        wavelength axis of the data in Angstrom
    rte: str
        RTE or CE or CE+RTE
    options: list
        options for CMILOS [num_wavelength_points, num_iterations] (default num_iterations = 15)
    out_dir: str
        output directory for intermediate .txt files
    loc: str
        location of the CMILOS executable

    Returns
    -------
    result: numpy ndarray
        inverted physical quantities [y,x,12] 
    """
    try:
        CMILOS_LOC = "../milos_codes/" + loc
        if os.path.isfile(CMILOS_LOC+'milos'):
            print("Cmilos executable located at:", CMILOS_LOC)
        else:
            raise ValueError('Cannot find cmilos:', CMILOS_LOC)

    except ValueError as err:
        print(err.args[0])
        print(err.args[1])
        return        

    if data.ndim == 4:
        sdata = data
    elif data.ndim != 4:
        print("Incorrect dimensions of 'data' array")
        exit()
    y,x,_,l = sdata.shape
    filename = out_dir + 'dummy_in.txt'
    with open(filename,"w") as f:
        for i in range(x):
            for j in range(y):
                for k in range(l):
                    f.write('%e %e %e %e %e \n' % (wave_axis[k],sdata[j,i,0,k],sdata[j,i,1,k],sdata[j,i,2,k],sdata[j,i,3,k])) #wv, I, Q, U, V

    cmd = CMILOS_LOC+"./milos"

    if rte == 'RTE':
        _ = subprocess.call(cmd+" "+str(options[0])+" "+str(options[1])+f" 0 {synthesis} {out_dir+'dummy_in.txt'}  >  {out_dir+'dummy_out.txt'}",shell=True)
    if rte == 'CE':
        _ = subprocess.call(cmd+" "+str(options[0])+" "+str(options[1])+f" 2 {synthesis} {out_dir+'dummy_in.txt'}  >  {out_dir+'dummy_out.txt'}",shell=True)
    if rte == 'CE+RTE':
        _ = subprocess.call(cmd+" "+str(options[0])+" "+str(options[1])+f" 1 {synthesis} {out_dir+'dummy_in.txt'}  >  {out_dir+'dummy_out.txt'}",shell=True)

    _ = subprocess.call(f"rm {out_dir + 'dummy_in.txt'}",shell=True)

    res = np.loadtxt(out_dir+'dummy_out.txt',skiprows=1)
    _ = subprocess.call(f"rm {out_dir+'dummy_out.txt'}",shell=True)
    
    if synthesis==0:
        result = np.zeros((12,y*x)).astype(float)
        for i in range(y*x):
            result[:,i] = res[i*12:(i+1)*12]
        result = result.reshape(12,x,y)
        result = np.einsum('ijk->ikj', result)
        return result
    
    elif synthesis==1:
        return res


def prep_and_run_cmilos(data, folder, angle, snapshot, dlambda, refwv, return_mean, niter, q_u_0 = True, ext=''):
    """prepare the data for CMILOS and run it
    
    Parameters
    ----------
    data: numpy ndarray
        Stokes profiles [y,x,stokes,wavelength] in the order IQUV
    folder: str
        path to folder
    angle: str
        angle of the inclination
    snapshot: str
        snapshot number of the MURaM simulation
    dlambda: float
        wavelength range in Angstrom (one half side, must be divisible by 0.014)
    refwv: float
        reference wavelength in Angstrom: 6173.341 or 5250.208
    return_mean: bool
        if True, return the mean of the BLOS, else return the BLOS map
    
    Returns
    -------
    result: numpy ndarray
        BLOS map or mean BLOS 
    """
    if q_u_0:
        data[:,:,1,:] = 0 #set Q=0
        data[:,:,2,:] = 0 #set U=0
    Ic = data[:,:,0,:50].mean()
    data = data/Ic
    num_points = int(2*1000*dlambda/(14) + 1.0)
    data = data[:,:,:,125-int((num_points-1)/2):126+int((num_points-1)/2)]
    print(f"Start wv index: {125-int((num_points-1)/2)}")
    print(f"End wv index: {126+int((num_points-1)/2)}")
    wavelengths = np.linspace(refwv-dlambda,refwv+dlambda,num_points)
    
    # if ext is not None:
    #     pass
    # # elif int(refwv) == 5250:
    # #     ext = '_5250_eqweights'
    # else:
    #     ext = ''
    print(ext)
    print(data.shape)
    result = run_cmilos(data, wavelengths, 'CE+RTE', options = [num_points,niter], out_dir = f'./{folder}_{angle}_{snapshot}_', loc = f'cmilos{ext}/')
    if return_mean:
        return np.mean(result[2,:,:]*np.cos(result[3,:,:]*np.pi/180.))
    else:
        return result[2,:,:]*np.cos(result[3,:,:]*np.pi/180.)
    
    
def prep_and_run_cmilos_OLD(data, folder, angle, snapshot, dlambda=0.21, refwv=6173.341, return_mean=True):
    """prepare the data for CMILOS and run it (DEPRECATED DUE TO ERROR- use prep_and_run_cmilos instead)
    kept due to backwards compatibility for reproducing incorrect results
    """
    data[:,:,1,:] = 0 #set Q=0
    data[:,:,2,:] = 0 #set U=0
    Ic = data[:,:,0,:50].mean()
    data = data/Ic
    num_points = int(2*1000*dlambda/(14) + 1.0)
    data = data[:,:,:,110:141] #-210,+210 - was 110:141 hardcoded even when dlambda = 0.35 requested
    wavelengths = np.linspace(refwv-dlambda,refwv+dlambda,num_points)
    if int(refwv) == 5250:
        ext = '_5250'
    else:
        ext = ''
    print(ext)
    print(data.shape)
    result = run_cmilos(data, wavelengths, 'CE+RTE', options = [31,15], out_dir = f'./{folder}_{angle}_{snapshot}_', loc = f'cmilos{ext}/')
    if return_mean:
        return np.mean(result[2,:,:]*np.cos(result[3,:,:]*np.pi/180.))
    else:
        return result[2,:,:]*np.cos(result[3,:,:]*np.pi/180.)
    

def run_mdi_blos(data, dlambda, refwv, return_mean, second_fourier=None):
    """calculate BLOS using MDI-like algorithm
    
    Parameters
    ----------
    data: numpy ndarray
        Stokes profiles [y,x,stokes,wavelength] in the order IQUV
    dlambda: float
        wavelength range in Angstrom (one half side, must be divisible by 0.014)
    refwv: float
        reference wavelength in Angstrom: 6173.341 or 5250.208
    return_mean: bool
        if True, return the mean of the BLOS, else return the BLOS map
    second_fourier: bool
        if True, use second Fourier component, else use first Fourier component
    
    Returns
    -------
    BLOS: numpy ndarray
        BLOS map or mean BLOS
    """
    I = data[:,:,0,:]
    stokesv = data[:,:,3,:]
    Ic = I[:50].mean()
    stokesv /= Ic
    I /= Ic
    
    RCP = (I + stokesv)/2
    LCP = (I - stokesv)/2

    T = dlambda*2
    T_width = T*1000/2/0.014 #T in AA - *1000 to avoid floating point errors
    assert T_width.is_integer()
    T_width = int(T_width/1000)
    
    num_samples = int(T_width*2 + 1)
    x = np.linspace(-T/2,T/2,num_samples)
    central_idx = 125
    
    if second_fourier is None:
        PI_FAC = 2
    elif second_fourier is True:
        PI_FAC = 4
    else:
        print("second_fourier KWARG error")

    ya_LCP = LCP[...,central_idx-T_width:central_idx+T_width+1]*np.cos(PI_FAC*np.pi*x/T)
    yb_LCP = LCP[...,central_idx-T_width:central_idx+T_width+1]*np.sin(PI_FAC*np.pi*x/T)

    ya_RCP = RCP[...,central_idx-T_width:central_idx+T_width+1]*np.cos(PI_FAC*np.pi*x/T)
    yb_RCP = RCP[...,central_idx-T_width:central_idx+T_width+1]*np.sin(PI_FAC*np.pi*x/T)

    a_1l = 2/T * spi.simpson(ya_LCP, axis=-1)
    b_1l = 2/T * spi.simpson(yb_LCP, axis=-1)

    a_1r = 2/T * spi.simpson(ya_RCP, axis=-1)
    b_1r = 2/T * spi.simpson(yb_RCP, axis=-1)
    
    if int(refwv) == 6173:
        dv_dx = 48562.4 #m/s/AA
        K_m = 0.231 #Gs/m 
    elif int(refwv) == 5250:
        dv_dx = 299792458/5250.208 #1/(2*4.67e-5*5250.2*3*299792458/1e10) - missing some factor of 1e-3
        K_m = 0.226
    
    V_RCP = dv_dx * T / (PI_FAC*np.pi) * np.arctan(b_1r/a_1r)
    V_LCP = dv_dx * T / (PI_FAC*np.pi) * np.arctan(b_1l/a_1l)
    
    V = ( V_RCP + V_LCP ) / 2
    BLOS = ( V_LCP - V_RCP ) * K_m
    
    if return_mean:
        return np.mean(-BLOS)
    else:
        return -BLOS
    

def get_ic(folder, snapshot):
    """get continuum intensity at disk centre for a given snapshot
    
    Parameters
    ----------
    folder: str
        path to folder
    snapshot: str
        snapshot number of the MURaM simulation
    
    Returns
    -------
    _: float
        continuum intensity at disk centre"""
    data = load_data(folder,'00',snapshot)
    return data[:,:,0,:50].mean()


def run_cog_blos(data, dlambda, refwv, return_mean):
    """calculate BLOS using Centre-of-gravity algorithm
    
    Parameters
    ----------
    data: numpy ndarray
        Stokes profiles [y,x,stokes,wavelength] in the order IQUV
    dlambda: float
        wavelength range in Angstrom (one half side, must be divisible by 0.014)
    refwv: float
        reference wavelength in Angstrom: 6173.341 or 5250.208
    return_mean: bool
        if True, return the mean of the BLOS, else return the BLOS map
    
    Returns
    -------
    BLOS: numpy ndarray
        BLOS map or mean BLOS
    """
    I = data[:,:,0,:]
    stokesv = data[:,:,3,:]
    RCP = (I + stokesv)
    LCP = (I - stokesv)

    num_points = 2*1000*dlambda/(14) + 1.0
    assert 1000*dlambda % 14 == 0.0 
    num_points = int(num_points)
    wavelength_range = np.linspace(refwv - dlambda, refwv + dlambda, num_points)
    idx = int((num_points-1)/2)
    
    Ic = I[:,:,:50].mean(axis=-1)
    Ic = np.repeat(Ic[:, :, np.newaxis], num_points, axis=2) #algorithm needs local Ic at each pixel location
    
    nominator_RCP = wavelength_range * (Ic - RCP[:,:,125-idx:125+idx+1])
    denominator_RCP =  (Ic - RCP[:,:,125-idx:125+idx+1])
    RCP_cog = spi.simpson(nominator_RCP, dx=0.014, axis=-1)/spi.simpson(denominator_RCP, dx=0.014, axis=-1)
    
    nominator_LCP = wavelength_range * (Ic - LCP[:,:,125-idx:125+idx+1])
    denominator_LCP =  (Ic - LCP[:,:,125-idx:125+idx+1])
    LCP_cog = spi.simpson(nominator_LCP, dx=0.014, axis=-1)/spi.simpson(denominator_LCP, dx=0.014, axis=-1)
    
    if int(refwv) == 6173:
        g_Lande = 2.5
    elif int(refwv) == 5250:
        g_Lande = 3.0
    else:
        print("KWARG Error: wavelength ")
    
    BLOS = (RCP_cog - LCP_cog)/2/(4.67e-13*g_Lande*(refwv**2))
    if return_mean:
        return np.mean(BLOS)
    else:
        return BLOS,
    

def run_wfa_blos(data, dlambda, refwv, return_mean):
    """calculate BLOS using Weak field approximation via Linear Least Squares
    
    Parameters
    ----------
    data: numpy ndarray
        Stokes profiles [y,x,stokes,wavelength] in the order IQUV
    dlambda: float
        wavelength range in Angstrom (one half side, must be divisible by 0.014)
    refwv: float
        reference wavelength in Angstrom: 6173.341 or 5250.208
    return_mean: bool
        if True, return the mean of the BLOS, else return the BLOS map
    
    Returns
    -------
    BLOS: numpy ndarray
        BLOS map or mean BLOS
    """
    I = data[:,:,0,:]
    stokesv = data[:,:,3,:]
    dIdlambda = np.gradient(I, 0.014, axis=-1)
    
    num_points = 2*1000*dlambda/(14) + 1.0
    assert 1000*dlambda % 14 == 0.0 
    num_points = int(num_points)
    idx = int((num_points-1)/2)
    
    print(125-idx)
    print(125+idx+1)
    
    nominator = np.sum(dIdlambda[...,125-idx:125+idx+1]*stokesv[...,125-idx:125+idx+1], axis = -1)
    denominator =  np.sum(dIdlambda[...,125-idx:125+idx+1]**2, axis = -1)
    
    if int(refwv) == 6173:
        g_Lande = 2.5
    elif int(refwv) == 5250:
        g_Lande = 3.0
    else:
        print("KWARG Error: refwv ")
    
    BLOS = - (nominator/denominator) / (4.67e-13*g_Lande*(refwv**2))
    if return_mean:
        return np.mean(BLOS)
    else:
        return BLOS


def wfa_blos_diff(data, dlambda, refwv, return_mean):
    """calculate BLOS using Weak field approximation via Linear Least Squares using np.diff for gradient
    
    Parameters
    ----------
    data: numpy ndarray
        Stokes profiles [y,x,stokes,wavelength] in the order IQUV
    dlambda: float
        wavelength range in Angstrom (one half side, must be divisible by 0.014)
    refwv: float
        reference wavelength in Angstrom: 6173.341 or 5250.208
    return_mean: bool
        if True, return the mean of the BLOS, else return the BLOS map
    
    Returns
    -------
    BLOS: numpy ndarray
        BLOS map or mean BLOS
    """
    I = data[:,:,0,:]
    stokesv = data[:,:,3,:]
    dIdlambda = np.diff(I, axis=-1)/0.014 #instead of np.gradient
    
    num_points = 2*1000*dlambda/(14) + 1.0
    assert 1000*dlambda % 14 == 0.0 
    num_points = int(num_points)
    idx = int((num_points-1)/2)
    
    print(125-idx)
    print(125+idx+1)
    
    nominator = np.sum(dIdlambda[...,125-idx:125+idx+1]*stokesv[...,125-idx:125+idx+1], axis = -1)
    denominator =  np.sum(dIdlambda[...,125-idx:125+idx+1]**2, axis = -1)
    
    if int(refwv) == 6173:
        g_Lande = 2.5
    elif int(refwv) == 5250:
        g_Lande = 3.0
    else:
        print("KWARG Error: refwv ")
    
    BLOS = - (nominator/denominator) / (4.67e-13*g_Lande*(refwv**2))
    if return_mean:
        return np.mean(BLOS)
    else:
        return BLOS
    
    
def cmilos_synthesis(data, folder, angle, snapshot, dlambda, refwv, return_mean, niter):
    """prepare the data for CMILOS and run it
    
    Parameters
    ----------
    data: numpy ndarray
        Stokes profiles [y,x,stokes,wavelength] in the order IQUV
    folder: str
        path to folder
    angle: str
        angle of the inclination
    snapshot: str
        snapshot number of the MURaM simulation
    dlambda: float
        wavelength range in Angstrom (one half side, must be divisible by 0.014)
    refwv: float
        reference wavelength in Angstrom: 6173.341 or 5250.208
    return_mean: bool
        if True, return the mean of the BLOS, else return the BLOS map
    
    Returns
    -------
    result: numpy ndarray
        BLOS map or mean BLOS 
    """
    data[:,:,1,:] = 0 #set Q=0
    data[:,:,2,:] = 0 #set U=0
    Ic = data[:,:,0,:50].mean()
    data = data/Ic
    num_points = int(2*1000*dlambda/(14) + 1.0)
    data = data[:,:,:,125-int((num_points-1)/2):126+int((num_points-1)/2)]
    print(f"Start wv index: {125-int((num_points-1)/2)}")
    print(f"End wv index: {126+int((num_points-1)/2)}")
    wavelengths = np.linspace(refwv-dlambda,refwv+dlambda,num_points)
    if int(refwv) == 5250:
        ext = '_5250'
    else:
        ext = ''
    print(ext)
    print(data.shape)
    result = run_cmilos(data, wavelengths, 'CE+RTE', options = [num_points,niter], out_dir = f'./{folder}_{angle}_{snapshot}_', loc = f'cmilos{ext}/', synthesis=1)
    if return_mean:
        return np.mean(result[2,:,:]*np.cos(result[3,:,:]*np.pi/180.))
    else:
        return result[2,:,:]*np.cos(result[3,:,:]*np.pi/180.)


def run_pymilos(data, dlambda, refwv, return_mean, weight=[1,1,1,1], initial_model=[400.,30.,120.,3.,0.05,1.5,0.01,0.22,0.85], niter=15, norm=True):
    """calculate BLOS using pymilos
       reruns the inversion for pixels where BLOS > 1500 with new parameters to better fit profiles

    Parameters
    ----------
    data: numpy ndarray
        Stokes profiles [y,x,stokes,wavelength] in the order IQUV
    dlambda: float
        wavelength range in Angstrom (one half side, must be divisible by 0.014)
    refwv: float
        reference wavelength in Angstrom: 6173.341 or 5250.208
    return_mean: bool
        if True, return the mean of the BLOS, else return the BLOS map
    weight: list
        weights for the inversion
    initial_model: list
        initial model for the inversion
    niter: int
        number of iterations for the inversion
    norm: bool
        if True, normalize the data by continuum intensity
    
    Returns
    -------
    BLOS: numpy ndarray
        BLOS map or mean BLOS
    """
    if norm:
        Ic = data[:,:,0,100].mean()
    num_points = int(2*1000*dlambda/(14) + 1.0)
    data_copy = copy.deepcopy(data)
    data = data[:,:,:,125-int((num_points-1)/2):126+int((num_points-1)/2)]
    wavelengths = np.linspace(refwv-dlambda,refwv+dlambda,num_points)

    ny, nx, npol, nwave = data.shape
    data = data.reshape(nx*ny, npol, nwave)

    if norm:
        data = data/Ic

    if int(refwv) == 6173:
        out = pym.pymilos(np.array([wavelengths.size,niter,1,0,0,0,0,0,0]), data, wavelengths, weight=weight, \
                       initial_model=initial_model)
    elif int(refwv) == 5250:
        out = pym5.pymilos(np.array([wavelengths.size,niter,1,0,0,0,0,0,0]), data, wavelengths, weight=weight, \
                       initial_model=initial_model)   

    out = out.reshape(ny,nx,12)
    out = np.moveaxis(out,-1,0)

    blos = out[2,:,:]*np.cos(out[3,:,:]*np.pi/180.)

    idx_strong = np.where(np.abs(blos)>1500)

    #print("mean before ",np.mean(blos))
    
    if idx_strong[0].size:
        strong_model = initial_model.copy()
        strong_model[3] = 50.
        niter = 30
        #print(idx_strong[0].shape,idx_strong[1].shape)
        data_strong = data_copy[idx_strong[0],idx_strong[1],:,125-int((num_points-1)/2):126+int((num_points-1)/2)]

        if norm:
            data_strong = data_strong/Ic
        #print(data_strong.shape)
        ny_strong, nx_strong = idx_strong[0].size, idx_strong[1].size
        
        #reinvert pixels where blos > 1500, and set eta0 to 10 and niter=30
        #print('reinverting strong pixels with eta0 = 10 and niter=30')
        if int(refwv) == 6173:
            out_strong = pym.pymilos(np.array([wavelengths.size,niter,1,0,0,0,0,0,0]), data_strong, \
                                 wavelengths, weight=weight, initial_model=strong_model)
        elif int(refwv) == 5250:
            out_strong = pym5.pymilos(np.array([wavelengths.size,niter,1,0,0,0,0,0,0]), data_strong, \
                                 wavelengths, weight=weight, initial_model=strong_model)        
        #out_strong = out.reshape(ny_strong,nx_strong,12)
        out_strong = np.moveaxis(out_strong,-1,0)
        blos[idx_strong] = out_strong[2,:]*np.cos(out_strong[3,:]*np.pi/180.)

    #print("mean after ",np.mean(blos))
    
    if return_mean:
        return np.mean(blos)
    else:
        return blos
    

def print_console(folder, angle, snapshot):
    """print the folder,snapshot and angle for the given simulation
    
    Parameters
    ----------
    folder: str
        path to folder
    angle: str
        angle of the inclination (degrees)
    snapshot: str
        snapshot number of the MURaM simulation
        
    Returns
    -------
    None  
    """
    print("--------START---------")
    print("Folder: ", folder)
    print("Snapshot: ", snapshot)
    print("Angle: ", angle)


def get_cmilos_blos_inst_pixel(folder, angle, snapshot, dlambda, inst=None, downscale_mean=False, remainder_mode='round', \
                               refwv=6173.341, dshape=288, pixel_res=20.833333, return_mean=True, \
                                default_dir="/path/to/spinor_files"):
    """get BLOS for a given instrument resolution via CMILOS (DEPRECATED - use get_blos_inst_pixel)"""
    print_console(folder, angle, snapshot)
    data = load_data(folder, angle, snapshot, default_dir=default_dir)
    if inst == 'onepix':
        data = rebin(data,(1,1,4,251),downscale_mean)
    else:
        new_shape = get_inst_shape(inst, angle, remainder_mode, dshape, pixel_res)
        data = interp_data_to_nearest_divisible_integer(data, new_shape)
        data = rebin(data, new_shape, downscale_mean)
    print("---------END----------")
    return prep_and_run_cmilos(data, folder, angle, snapshot, dlambda, refwv, return_mean)


def create_stokes_profiles(folder, angle, snapshot, inst=None, downscale_mean=False, remainder_mode='round', dshape=288, \
                           pixel_res=20.833333, default_dir="/path/to/spinor_files"):
    """create stokes profiles for a given instrument resolution
    Parameters
    ----------
    folder: str
        path to folder
    angle: str
        angle of the inclination
    snapshot: str
        snapshot number of the MURaM simulation
    inst: str
        instrument name
    downscale_mean: bool
        if True, downscale the data using skimage.transform.downscale_local_mean
    remainder_mode: str
        mode for rounding non-integer pixel counts, can be 'floor', 'ceil' or 'round'
    dshape: int
        horiztonal dimension of the data (288) of the MURaM simulation
    pixel_res: float
        pixel resolution of the original MURaM simulation (20.833333) km/pixel
    default_dir: str
        default path to the folders
    
    Returns
    -------
    data: numpy ndarray
        Stokes profiles [y,x,stokes,wavelength] in the order IQUV
    """
    print_console(folder, angle, snapshot)
    data = load_data(folder, angle, snapshot, default_dir=default_dir)
    if inst == 'onepix':
        data = rebin(data,(1,1,4,251),downscale_mean)
    else:
        new_shape = get_inst_shape(inst, angle, remainder_mode, dshape, pixel_res)
        data = interp_data_to_nearest_divisible_integer(data, new_shape)
        data = rebin(data, new_shape, downscale_mean)
    print("-------PREP END--------")
    return data


def get_blos(data, dlambda, refwv, blos_method, return_mean, folder, angle, snapshot, q_u_0=False, ext='',niter=15,\
             init_model=[400.,30.,120.,10.,0.05,1.5,0.01,0.22,0.85], wghts=[1,1,1,1]):
    """get BLOS for a given blos method
    
    Parameters
    ----------
    data: numpy ndarray
        Stokes profiles [y,x,stokes,wavelength] in the order IQUV
    dlambda: float
        wavelength range in Angstrom
    refwv: float
        reference wavelength in Angstrom
    blos_method: str
        method for calculating BLOS, can be 'MDI', 'COG', 'WFA' or 'CMILOS'
    return_mean: bool
        if True, return the mean of the BLOS, else return the BLOS map
    
    Returns
    -------
    BLOS: numpy ndarray
        BLOS map or mean BLOS
    """
    if blos_method == 'MDI':
        print("------MDI BLOS-------")
        return run_mdi_blos(data, dlambda, refwv, return_mean, second_fourier=None)
    elif blos_method == 'COG':
        print("------COG BLOS-------")
        return run_cog_blos(data, dlambda, refwv, return_mean)
    elif blos_method == 'WFA':
        print("------WFA BLOS-------")
        return run_wfa_blos(data, dlambda, refwv, return_mean)
    elif blos_method == 'WFA-diff':
        print("------WFA BLOS-------")
        return wfa_blos_diff(data, dlambda, refwv, return_mean)
    elif blos_method == 'CMILOS':
        print("----CMILOS BLOS------")
        return prep_and_run_cmilos(data, folder, angle, snapshot, dlambda, refwv, return_mean, q_u_0 = q_u_0, ext=ext, niter=niter)
    elif blos_method == 'pymilos':
        print("----PYMILOS BLOS------")
        return run_pymilos(data, dlambda, refwv, return_mean, weight=wghts,\
                                   initial_model=init_model, niter=niter)


def get_blos_inst_pixel(folder, angle, snapshot, dlambda=0.35, inst=None, downscale_mean=False, remainder_mode='round', \
                        refwv=6173.341, dshape=288, pixel_res=20.833333, blos_method='CMILOS', return_mean=True, \
                        default_dir="/path/to/spinor_files", q_u_0=False, ext='',niter=15, \
                        initial_model=[400.,30.,120.,10.,0.05,1.5,0.01,0.22,0.85], weights=[1,1,1,1]):
    """get BLOS for a given instrument resolution and method from a given snaspshot and inclination

    Parameters
    ----------
    folder: str
        path to folder
    angle: str
        angle of the inclination
    snapshot: str
        snapshot number of the MURaM simulation
    dlambda: float
        wavelength range in Angstrom (one half side, default 0.35, must be divisible by 0.014)
    inst: str
        instrument name (default None) can be 'MURaM', 'HRT', 'FDT', 'HMI' or 'onepix'
    downscale_mean: bool
        if True, downscale the data using skimage.transform.downscale_local_mean
    remainder_mode: str
        mode for rounding non-integer pixel counts, can be 'floor', 'ceil' or 'round'
    refwv: float
        reference wavelength in Angstrom: 6173.341 (default) or 5250.208
    dshape: int
        horiztonal dimension of the data (288) of the MURaM simulation
    pixel_res: float
        pixel resolution of the original MURaM simulation (20.833333) km/pixel
    blos_method: str
        method for calculating BLOS, can be 'MDI', 'COG', 'WFA' or 'CMILOS' (default)
    return_mean: bool
        if True, return the mean of the BLOS, else return the BLOS map
    default_dir: str
        default path to the folders 

    Returns
    -------
    BLOS: numpy ndarray
        BLOS map or mean BLOS
    """
    data = create_stokes_profiles(folder, angle, snapshot, inst, downscale_mean, remainder_mode, dshape, pixel_res, \
                                  default_dir=default_dir)
    return get_blos(data, dlambda, refwv, blos_method, return_mean, folder, angle, snapshot, q_u_0, ext, niter, initial_model, weights)


def get_total_polarisation(data, folder, snapshot):
    """calculate total polarisation from stokes V profiles (absolute)

    Parameters
    ----------
    data: numpy ndarray
        Stokes profiles [y,x,stokes,wavelength] in the order IQUV
    folder: str
        path to folder
    snapshot: str
        snapshot number of the MURaM simulation
    
    Returns
    -------
    _ : float
        total polarisation normalised to Ic
    """
    Ic = get_ic(folder, snapshot)
    stokesv = data[:,:,3,:]/Ic
    return np.sum(spi.simpson(abs(stokesv), dx=0.014, axis=-1))