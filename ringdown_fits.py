import glob
import numpy as np
import re
import matplotlib.pyplot as plt
import qnm
from scipy.optimize import minimize,curve_fit


def rd_cte(theta,times = None,**args):
    """RD model parametrized with the damping time tau and with the QNM spectrum fixd to GR. 
    """ 

    ansatz = theta[0]*np.exp(1j*theta[1])
    
    if times is None:
        times = np.linspace(0,100,1000)
        
    ansatz = np.array(ansatz*len(times))
    return ansatz

def rd_model_wtau_fixed(theta,times = None,**args):
    """RD model parametrized with the damping time tau and with the QNM spectrum fixd to GR. 
    """ 
  
    dim=int(len(theta)/2)
        
    xvars = theta[ : (dim)]
    yvars = theta[(dim) : 2*(dim)]
    tau=args['qnms'][:,1].flatten()
    w=args['qnms'][:,0].flatten()
    ansatz = 0
    
    if times is None:
        times = np.linspace(0,100,1000)
    
    for i in range (0,dim):
        ansatz += (xvars[i]*np.exp(1j*yvars[i]))*np.exp(-times/tau[i]) * (np.cos(w[i]*times)-1j*np.sin(w[i]*times))
        #ansatz += (xvars[i]+1j*yvars[i])*np.exp(-times/tau[i]) * (np.cos(w[i]*times)-1j*np.sin(w[i]*times))

    return ansatz

def rd_model_wtau_m_af(theta,times = None,**args):
    """RD model parametrized with the damping time tau and with the QNM spectrum fixd to GR. The QNM spectrum is given from the mass and spin.
    """ 
    dim=int((len(theta)-2)/2) 
    xvars = theta[ : (dim)]
    yvars = theta[(dim) : 2*(dim)]
    mass_vars = theta[-2]
    spin_vars = theta[-1]
        
    if dim == 1:
        w, tau = np.array([QNM_spectrum(2,2,n,mass_vars,spin_vars) for n in range(1)])[0] 
    else:
        w, tau = np.array([QNM_spectrum(2,2,n,mass_vars,spin_vars) for n in range(dim)])
    
    if times is None:
        times = np.linspace(0,100,1000)

    ansatz = 0
    for i in range (0,dim):
        ansatz += (xvars[i]*np.exp(1j*yvars[i]))*np.exp(-times/tau[i]) * (np.cos(w[i]*times)-1j*np.sin(w[i]*times))
    return ansatz

def rd_model_wtau_a_b(theta,times = None,real = True, **args):
    """RD model parametrized with the damping time tau. The values for the amplitude, phase, frequency and damping times are left free.
    """ 
    
    dim = int(len(theta)/4)

    xvars = theta[ : (dim)]
    yvars = theta[(dim) : 2*(dim)]
    avars = theta[2*(dim) : 3*(dim)]
    bvars = theta[3*(dim) : ]
    tau=args['qnms'][:,1].flatten()
    w=args['qnms'][:,0].flatten()

    if times is None:
        times = np.linspace(0,100,1000)
        
    ansatz = 0
    for i in range (0,dim):
        if not real:
            ansatz += (xvars[i]*np.exp(1j*yvars[i]))*np.exp(-times/(tau[i]*(1+bvars[i]))) * (np.cos(w[i]*(1+avars[i])*times)-1j*np.sin(w[i]*times))
        # -1j to agree with SXS convention
        else:
            ansatz += xvars[i]*np.exp(-times/(tau[i]*(1+bvars[i]))) * (np.cos(w[i]*(1+avars[i])*times+yvars[i]))
    return ansatz
                                      
def rd_model_wtau(theta,times = None,real = True,**args):
    """RD model parametrized with the damping time tau. The values for the amplitude, phase, frequency and damping times are left free.
    """ 
    
    dim = int(len(theta)/4)

    xvars = theta[ : (dim)]
    yvars = theta[(dim) : 2*(dim)]
    wvars = theta[2*(dim) : 3*(dim)]
    tvars = theta[3*(dim) : ]
    

    if times is None:
        times = np.linspace(0,100,1000)
        
    ansatz = 0
    for i in range (0,dim):
        if not real:
            ansatz += (xvars[i]*np.exp(1j*yvars[i]))*np.exp(-times/tvars[i]) * (np.cos(wvars[i]*times)-1j*np.sin(wvars[i]*times))
        else:
            ansatz += xvars[i]*np.exp(-times/tvars[i]) * (np.cos(wvars[i]*times+yvars[i]))

    return ansatz


def rd_model_wtau_cte(theta,times = None,**args):
    """ RD model generated from rd_model_wtau + constant.
    """ 
    dim = int(len(theta)/4)

    theta_cte = np.array([theta[dim],theta[2*dim]])
    theta_qnm = np.delete(theta, [dim, 2*dim])
    
    ansatz = rd_model_wtau(theta_qnm,times=times) + rd_cte(theta_cte,times = times)
        
    return ansatz

def rd_model_wtau_m_af_cte(theta,times = None,**args):
    """RD model parametrized with the damping time tau and with the QNM spectrum fixd to GR. 
    """ 
    ind_mid = int(len(theta-2)/2)
    theta_cte = np.array([theta[ind_mid],theta[-3]])
    theta_qnm = np.delete(theta, [ind_mid, -3])
    
    ansatz = rd_model_wtau_m_af(theta_qnm,times=times) + rd_cte(theta_cte,times = times)
        
    return ansatz

def rd_model_wtau_fixed_cte(theta,times = None,**args):
    """RD model parametrized with the damping time tau and with the QNM spectrum fixd to GR. 
    """ 
    ind_mid = int(len(theta)/2)
    theta_cte = np.array([theta[ind_mid-1],theta[-1]])
    theta_qnm = np.delete(theta, [ind_mid-1, -1])

    ansatz = rd_model_wtau_fixed(theta_qnm,times=times) + rd_cte(theta_cte,times = times)
        
    return ansatz

def rd_cte(theta,times = None,**args):
    """RD model parametrized with the damping time tau and with the QNM spectrum fixd to GR. 
    """ 
    if times is None:
        times = np.linspace(0,100,1000)
        
    ansatz = theta[-2]*np.exp(1j*theta[-1])*np.ones(len(times))
        
    return ansatz

def rd_model_wtau_a_b_last(theta,times = None,real = True, **args):
    """RD model parametrized with the damping time tau. The values for the amplitude, phase, frequency and damping times are left free.
    """ 
    
    dim = int((len(theta)+2)/4)

    xvars = theta[ : (dim)]
    yvars = theta[(dim) : 2*(dim)]
    avars = 0
    bvars = 0
    
    tau=args['qnms'][:,1].flatten()
    w=args['qnms'][:,0].flatten()

    if times is None:
        times = np.linspace(0,100,1000)
        
    ansatz = 0
    for i in range (0,dim):
        if i == dim-1:
            avars = theta[-2]
            bvars = theta[-1]
        if not real:
            ansatz += (xvars[i]*np.exp(1j*yvars[i]))*np.exp(-times/(tau[i]*(1+bvars))) * (np.cos(w[i]*(1+avars)*times)-1j*np.sin(w[i]*times))
        # -1j to agree with SXS convention
        else:
            ansatz += xvars[i]*np.exp(-times/(tau[i]*(1+bvars))) * (np.cos(w[i]*(1+avars)*times+yvars[i]))
    return ansatz

dic = {'w-tau-fixed':rd_model_wtau_fixed, 'cte':rd_cte,'w-tau-fixed-cte':rd_model_wtau_fixed_cte,'rd-cte':rd_cte,
      'w-tau-m-af':rd_model_wtau_m_af,'w-tau-m-af-cte':rd_model_wtau_m_af_cte,
      'w-tau':rd_model_wtau,'w-tau-a-b':rd_model_wtau_a_b,'w-tau-a-b-last':rd_model_wtau_a_b_last,
       'w-tau-cte':rd_model_wtau_cte}


def QNM_spectrum(l,m,n,mass,spin):
    """ It computes the RD frequencies and damping times in NR units.
    """  
    omegas_new=modes_spec[-2,l,m,n](spin)[0]
    w_m_a = (np.real(omegas_new))/mass
    tau_m_a=-1/(np.imag(omegas_new))*mass

    return (np.array([w_m_a]), np.array([tau_m_a]))

def QNM_spectrum_w(l,m,n,mass,spin):
    """ It computes the RD frequencies and damping times in NR units.
    """  
    omegas_new=modes_spec[-2,l,m,n](spin)[0]/mass

    return omegas_new

def model_amplitude(theta):
    b0,c0,e0,a1,a2=theta
    return ((-b0 - c0 - e0) +b0/values[:,2]+c0/values[:,2]**2+e0/values[:,2]**3)*(1 + a1*values[:,3]+a2*values[:,3]**2)


def fit_qnm(time_data,initial_parameters,t0 = 0,linear = False, model = 'w-tau-fixed-cte', likelihood = 'chi_2',min_method = 'SLSQP', tol=1e-6,
            bounds = None,**args):
    '''
        """
    Find the complex frequency that minimizes the mismatch for a given 
    ringdown start time. a Set of "fixed" frequencies can also be included in 
    the fit, specified through the modes, Mf, and chif arguments.
    Parameters
    ----------
    time_data : array_like
        The times and data associated with the data to be fitted.
    
    initial_parameters: array_like
        List of initial input parameters used for the mininimization of the log_likelihood 
        
    linear: boolean, optional.
        Default: False. If True it finds the exact solution to the linear system A x = b. The list of initial parameters is
        in this case qnm_freqs. For fitting the free constant, qnm_freqs(cte) = 0. 
        
    t0 : float, optional 
        The start time of the ringdown model.
        
    modes : string, optional
            QNM mode used for the fitting. Models are:
        
        -- w-tau-fixed-cte: (Default). QNM spectrum is fixed to GR. We fit for the amplitude and phase of each mode. 
                            It corrects for the extra SXS amplitude cte. 
        
    min_method : str, optional
        The method used to find the mismatch minimum in the complex-frequency 
        space. This can be any method available to scipy.optimize.minimize. 
        This includes None, in which case the method is automatically chosen. 
        The default is 'SLSQP'. 
    
    tol: float, optional.
        Default tol = 1e-6. Tolerance used to stop the fitting algorithm.
    
    bounds: array_like, optional. 
            Bounds of the parameters one wants to fit for. The default option is bounds = None.
    
            
    t_offset: float, optional. 
            Offset time used for correcting the correct_cte = True case. Default is t_offset = 100, thus ensuring
            that the fundamental tones has faint out.
   
        
    min_method : str, optional
        The method used to find the mismatch minimum in the complex-frequency 
        space. This can be any method available to scipy.optimize.minimize. 
        This includes None, in which case the method is automatically chosen. 
        The default is 'Nelder-Mead'.
    Returns
    -------
    omega_bestfit : complex
        The complex frequency that minimizes the mismatch.
    """
    
    '''    
    
    def log_likelihood(theta,model=model,times=None,args = None):
        
        data_mod = dic[model](theta,times=times,**args) 
        result = 0.5*np.sum((data_mod.real - data.real)**2 +  (data_mod.imag - data.imag)**2)  
        
        if np.isnan(result):
            return np.inf
        return result
    
    data = time_data[:,1][time_data[:,0] >= t0]
    times = time_data[:,0][time_data[:,0] >= t0] 
    
    if not linear: 
        soln = minimize(log_likelihood,
                    list(initial_parameters),
                    args=(model,times,args),
                    method = min_method,
                    bounds = bounds)
        C = soln.x
    else:
        # If linear, the initial_parameters are the list of frequencies and damping times. In case of wanting to add the free cte, 
        # the corresponding frequency must be fixed to 0.
    
        mat = np.array([
        np.exp(-1j*initial_parameters[i]*(times)) for i in range(len(initial_parameters))
        ]).T
        C, res, rank, s = np.linalg.lstsq(mat, data, rcond=None)

    return C  

def sxs_nr_residuals(data_1,data_2):
    mlength=min(len(data_1),len(data_2))
    ph_diff = np.angle(data_1[0])-np.angle(data_2[0])
    
    return abs(data_1[:mlength]-data_2[:mlength]*np.exp(1j*ph_diff))

def sxs_nr_time_errors(time1,time2):
    mtime=min(len(time1),len(time2))    
    return time1[:mtime]

def mismatch_t(wave1,wave2,t0,tend):
    
    """
    Calculates the match between two complex waveforms.
    Parameters
    ----------
        
    wave1, wave2 : array_like
        The two waveforms to calculate the mismatch between. The first column of the wave arrays is the time axis.
        
    RETURNS
    -------
    M : float
        The mismatch between the two waveforms.
    """
    boolean_sel = np.logical_and(wave1[:,0]>=t0,wave1[:,0]<= tend)
    times = wave1[:,0][boolean_sel]
    data1 = wave1[:,1][boolean_sel]
    data2 = wave2[:,1][boolean_sel]
    norm1 = np.sum(data1*np.conjugate(data1))
    norm2 = np.sum(data2*np.conjugate(data2))
   
   
    match = np.sum(data1*np.conjugate(data2))/np.sqrt(norm1*norm2)
   
    return 1-abs(match)

def align_at_t0(time,data,t0=0):
    index_max = np.argmax(data)
    time_aligned = time[index_max:]-time[index_max]
    data_aligned = data[index_max:]

    boolean = time_aligned >=t0    
    return time_aligned[boolean],data_aligned[boolean]



def fit_qnm_varying_f(time_data,initial_parameters,ini_freqs = [],t0 = 0,t_end=100,min_method = 'SLSQP', tol=1e-6,
            bounds = None,**args):
    '''
        """
    Find the complex frequency that minimizes the mismatch for a given 
    ringdown start time. a Set of "fixed" frequencies can also be included in 
    the fit, specified through the modes, Mf, and chif arguments.
    Parameters
    ----------
    time_data : array_like
        The times and data associated with the data to be fitted.
    
    initial_parameters: array_like
        List of initial input parameters used for the mininimization of the log_likelihood 
        
    linear: boolean, optional.
        Default: False. If True it finds the exact solution to the linear system A x = b. The list of initial parameters is
        in this case qnm_freqs. For fitting the free constant, qnm_freqs(cte) = 0. 
        
    t0 : float, optional 
        The start time of the ringdown model.

    t_end : float, optional 
        The end time of the ringdown model.
        
    modes : string, optional
            QNM mode used for the fitting. Models are:
        
        -- w-tau-fixed-cte: (Default). QNM spectrum is fixed to GR. We fit for the amplitude and phase of each mode. 
                            It corrects for the extra SXS amplitude cte. 
        
    min_method : str, optional
        The method used to find the mismatch minimum in the complex-frequency 
        space. This can be any method available to scipy.optimize.minimize. 
        This includes None, in which case the method is automatically chosen. 
        The default is 'SLSQP'. 
    
    tol: float, optional.
        Default tol = 1e-6. Tolerance used to stop the fitting algorithm.
    
    bounds: array_like, optional. 
            Bounds of the parameters one wants to fit for. The default option is bounds = None.
    
            
    t_offset: float, optional. 
            Offset time used for correcting the correct_cte = True case. Default is t_offset = 100, thus ensuring
            that the fundamental tones has faint out.
   
        
    min_method : str, optional
        The method used to find the mismatch minimum in the complex-frequency 
        space. This can be any method available to scipy.optimize.minimize. 
        This includes None, in which case the method is automatically chosen. 
        The default is 'Nelder-Mead'.
    Returns
    -------
    omega_bestfit : complex
        The complex frequency that minimizes the mismatch.
    """
    
    '''    
    data_sel = (time_data[:,0]>=t0) & (time_data[:,0]<t0+t_end)

    t_data = time_data[data_sel]
    times = t_data[:,0]
    x0 = ini_freqs

    def mismatch_t_f(x,t_data,t0,tend):
        
        global C_lin
        # Extract the value of the free frequency
        omega_free = x[0] + 1j*x[1]
        
        # Combine the free frequency with the fixed frequencies
        frequencies = np.hstack([initial_parameters, omega_free])    
        # Construct the coefficient matrix
        a = np.array([
            np.exp(-1j*frequencies[i]*(times)) for i in range(len(frequencies))
            ]).T
    
        # Solve for the complex amplitudes, C. Also returns the sum of 
        # residuals, the rank of a, and singular values of a.
        C_lin, res, rank, s = np.linalg.lstsq(a, t_data[:,1], rcond=None)
    
        # Evaluate the model
        model = np.einsum('ij,j->i', a, C_lin)
        t_model = np.stack((t_data[:,0],model)).T
        
        # Calculate the mismatch for the fit
        mm = mismatch_t(t_data, t_model, t0,tend)
        return mm
    
    soln = minimize(mismatch_t_f,
                x0,
                args=(t_data,t0,t0+200),
                method = min_method,
                bounds = bounds)
    
    omega_sol = soln.x

    return np.hstack([abs(C_lin),np.angle(C_lin), omega_sol])


def QNM_spectrum_re_im(l,m,n,mass,spin):
    """ It computes the RD frequencies and damping times in NR units.
    """  
    omegas_new=modes_spec[-2,l,m,n](spin)[0]
    w_re = (np.real(omegas_new))/mass
    w_im = (np.imag(omegas_new))/mass

    return w_re+1j*w_im


def mismatch(times, wf_1, wf_2):
    """
    Calculates the mismatch between two complex waveforms.
    Parameters
    ----------
    times : array_like
        The times at which the waveforms are evaluated.
        
    wf_1, wf_2 : array_like
        The two waveforms to calculate the mismatch between.
        
    RETURNS
    -------
    M : float
        The mismatch between the two waveforms.
    """
    numerator = np.real(np.trapz(wf_1 * np.conjugate(wf_2), x=times))
    
    denominator = np.sqrt(np.trapz(np.real(wf_1 * np.conjugate(wf_1)), x=times)
                         *np.trapz(np.real(wf_2 * np.conjugate(wf_2)), x=times))
    
    return 1 - (numerator/denominator)