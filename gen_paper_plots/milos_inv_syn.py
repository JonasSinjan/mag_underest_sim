import os
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
import subprocess

def run_cmilos(data,wave_axis,rte, OLD_CMILOS_LOC, options=[6,15],out_dir='./', synthesis=0):
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
        if os.path.isfile(OLD_CMILOS_LOC+'milos'):
            pass #print("Cmilos executable located at:", CMILOS_LOC)
        else:
            raise ValueError('Cannot find cmilos:', OLD_CMILOS_LOC)

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

    cmd = OLD_CMILOS_LOC+"./milos"

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
    
    
def run_new_cmilos_syn(res, wave_axis, NEW_MILOS_LOC, out_dir='./', niter=15):
    """Synthesis using new MILOS

    Parameters
    ----------
    res: numpy ndarray
        inverted physical quantities [y,x,12]
    wave_axis: numpy ndarray
        wavelength axis of the data in Angstrom
    NEW_MILOS_LOC: str
        location of the new MILOS.x executable
    out_dir: str
        output directory for intermediate .txt files
    niter: int
        number of iterations
    
    Returns
    -------
    res_out: numpy ndarray
        fitted profiles [y,x,4,wavelength]
    """
    res1 = res[2:-1,:,:]
    res2 = np.moveaxis(res1, 0, -1)
    res3 = np.reshape(res2, (res.shape[1]*res.shape[2],9))
    
    file_dummy_in = os.path.join(out_dir, 'dummy_in.txt')
    file_dummy_out = os.path.join(out_dir, 'dummy_out.txt')

    data = res3.flatten(order='C')
    nmodels = len(data)//9

    filename = file_dummy_in
    with open(filename,"w") as f:
        #loop in wavelength axis
        for waves in wave_axis:
            f.write('%.10f \n' % (waves) )
        #loop in input model
        iter = 0
        for model in data:
            if not(iter % 9):
                f.write('%d \n' % (iter // 9) )
            f.write('%.10f \n' % (model) )
            iter += 1
  
    try:
        if os.path.isfile(NEW_MILOS_LOC+'milos.x'):
            pass
        else:
            raise ValueError('Cannot find cmilos:', NEW_MILOS_LOC)

    except ValueError as err:
        print(err.args[0])
        print(err.args[1])
        return 
    
    cmd = NEW_MILOS_LOC+"milos.x"

    trozo = f" {int(wave_axis.size)} {niter} {1} {1} {0} {0} {0}"
    cmd = cmd + trozo + " " + file_dummy_in + " > " + file_dummy_out

    rte_on = subprocess.call(cmd,shell=True)
    res_out = np.loadtxt(file_dummy_out)

    _ = subprocess.call(f"rm {out_dir+'dummy_out.txt'}",shell=True)
    _ = subprocess.call(f"rm {out_dir+'dummy_in.txt'}",shell=True)
    res_out = np.reshape(res_out,(nmodels,wave_axis.size,5))
    res_out = np.einsum('kij->kji',res_out)
    res_out = res_out[:,1:,:]
    return res_out.reshape(res.shape[1],res.shape[2],4,wave_axis.size)


def get_fitted_profiles_from_stokes(stokes, wave_axis, old_cmilos_loc, new_cmilos_loc, niter=15):
    """Inversion and synthesis of the input Stokes profiles

    Parameters
    ----------
    stokes: numpy ndarray
        input Stokes profiles [y,x,stokes,wavelength] in the order IQUV (normalised)
    wave_axis: numpy ndarray
        wavelength axis of the data in Angstrom
    old_cmilos_loc: str
        location of the old CMILOS executable for inversion
    new_cmilos_loc: str
        location of the new MILOS.x executable for synthesis
    niter: int
        number of iterations
    
    Returns
    -------
    res_out: numpy ndarray
        fitted profiles [y,x,4,wavelength] 
    res: numpy ndarray
        inverted physical quantities [y,x,12]
    """
    num_points = wave_axis.size
    res = run_cmilos(stokes, wave_axis, "CE+RTE", OLD_CMILOS_LOC=old_cmilos_loc, options=[num_points,niter], \
                     out_dir='./', synthesis=0)
    
    return run_new_cmilos_syn(res, wave_axis, NEW_MILOS_LOC=new_cmilos_loc,out_dir='./', niter=15), res
