#!/usr/bin/env python
"""
=============== 1D cartesian functions ===============

list of functions called by particle filter 
'main_MPI.py' when dim = 1.

1D analysis on a single, pre-triangulated 
trajectory file with X_geo, Y_geo and Z_geo 
information.

although this is a 1D filter, for outputs to main particle 
filter code to remain consistent, y and z component columns 
remain, but are left unfilled.
------------------------------------------------------------------------------
As scatter/gather is done on numpy arrays, each table is in the format:
column index KEY:
 0 : 'X_geo'       - position along the straight line trajectory 
                     from a given initial position (m)
 1 : 'Y_geo'       - (unused)
 2 : 'Z_geo'       - (unused)
 3 : 'X_geo_DT'    - velocity (dx/dt) in ECEF (m/s) 
 4 : 'Y_geo_DT'    - (unused)
 5 : 'Z_geo_DT'    - (unused)
 36: 'X_eci'       - (unused)
 37: 'Y_eci'       - (unused)
 38: 'Z_eci'       - (unused)
 39: 'X_eci_DT'    - (unused)
 40: 'Y_eci_DT'    - (unused)
 41: 'Z_eci_DT'    - (unused)
 6 : 'mass'        - mass (kg)
 7 : 'cd'          - drag coefficient (aerodynamic => 2 * gamma; see Bronshten 1976)
 8 : 'A'           - shape coefficient,
                       A = cross sectional surface area / volume^(2/3)
 9 : 'kappa'       - shape-density coefficient, 
                       kappa = cd * A / density^(2/3)
 10: 'sigma'       - ablation coefficient
 11: 'mag_v'       - absolute visual magnitude
 12: 'tau'         - luminous efficiency parameter
 13: 'Q_x'         - variance of process noise for X position
 14: 'Q_y'         - (unused)
 15: 'Q_z'         - (unused)
 16: 'Q_v_x'       - variance of process noise for X velocity
 17: 'Q_v_y'       - (unused)
 18: 'Q_v_z'       - (unused)
 19: 'Q_m'         - variance of process noise for mass
 20: 'Q_cd'        - (unused)
 21: 'Q_cl'        - (unused)
 22: 'Q_k'         - variance of process noise for kappa
 23: 'Q_s'         - variance of process noise for sigma
 24: 'Q_tau'       - variance of process noise for luminous efficiency
 25: 'brightness'  - luminous intensiy
 26: 'rho'         - initial density.
 27: 'parent_index'- index of parent particle (t-1)
 28: 'orig_index'  - index of original particle assigned in dim.Initialise() (t0)
 29: 'weight'      - combined position and luminous weighting (assigned in main)
 30: 'D_DT'        - magnitude of velocity vector
 31: 'latitude'    - latitude (radians)
 32: 'longitude'   - longitude (radians)
 33: 'height'      - height (m)
 34: 'lum_weight'  - luminous weighting
 35: 'pos_weight'  - position weighting

------------------------------------------------------------------------------
"""
# import modules
# general
from math import *
import copy, random

# science
import numpy as np
import scipy 
import scipy.integrate
from numpy.linalg import norm

# Astropy
from astropy.table import Table, Column, join, hstack
import astropy.units as u
from astropy.time import TimeDelta

# own
import bf_functions_3d as bf
import trajectory_utilities as tu

def Initialise(x0, v0, index, oindex, N, P, params, alpha, date_info, mass_opt=3, m0_max=2000., gamma= 0.7854, eci_bool=False, uv_ECEF=[], x0_ECEF=[]):
    """ create a random particle to represent a meteoroid - 
        random distance along path, mass, velocity, ablation parameter 
        and shape-density parameter are created with Q_c noise
        according to global set ranges

    input: x0: [1x3] ARRAY : initial triangulated position guess
           v0: [1x3] ARRAY : initial triangulated velocity guess
           index:    INT   : particle index
           oindex:   INT   : particle index
           P: [1x6]  ARRAY : initial state uncertainty
           params:   ARRAY : min and max ranges to initialise parameters -
                            [A_min, A_max, pm_mean, pm_std, random_meteor_type, tau_min, tau_max]
           alpha:    DOUBLE: ballistic alpha parameter for mass_opt=1 (see Gritsevich 2007)
           date_info:ARRAY : t0 of fireball in y, d, s
           mass_opt: INT   : option of how to initiate masses
                            1- ballistic coefficient 
                            2- using random logarithmic distribution 
                            3- using random uniform distribution 
           m0_max:   DOUBLE: maximum mass to use for mass_opt=2,3
           gamma:    DOUBLE: incoming flight angle in radians from local horizontal
           eci_bool: BOOL  : tells function if X, Y, Z positions are in ECI (true) or ECEF (false). 

    output: numpy table following above index key 
    
    note: although this is a 1D filter, the number of columns remains the same
          so that outputs to main particle filter code are consistent. 
          y and z component columns are left unfilled.
    """

    # print(mass_opt)
    X = np.zeros(42)
    [A_min, A_max, pm_mean, pm_std, random_meteor_type, tau_min, tau_max] = params

    ## default filter creates a spread of initial particle state values
    X[0] = random.gauss(x0, P[0])     # initial position in x using Gaussian pdf. P values are already sqrt(cov)
    X[3] = random.gauss(v0, P[3])     # initial velocity in x using Gaussian pdf. P values are already sqrt(cov)
    
    # TODO: try and correlate shape and drag
    X[8] = random.random() * (A_max - A_min) + A_min
    X[7] = 1.3  #0.98182 * X[8]**2 - 1.7846 * X[8] + 1.6418

    # choose a random meteor type
    particle_choices = random.choice(random_meteor_type)

    # use corresponding density range for given meteoroid body type
    X[26] = random.gauss(pm_mean[particle_choices], pm_std[particle_choices])

    #calculate shape-density coefficient (kappa = A * cd / density^(2/3.)
    X[9]  = X[8] * X[7]/pow(X[26], 2/3.)  
    
    ## TODO: abalation coefficient
    ## values for sigma are supposedly correlated to density
    ## ranges here are taken from table 17 of Ceplecha 1998
    # if X[26] > 5000:
    #     X[10] = random.gauss(0.07e-6, 0.01e-6)
    # elif X[26] > 2500:
    #     X[10] = random.gauss(0.014e-6, 0.005e-6)
    # elif X[26] > 1500:
    #     X[10] = random.gauss(0.042e-6, 0.005e-6)
    # else:
    #     X[10] = random.gauss(0.1e-6, 0.05e-6)
    X[10] = 2* 0.014e-6

   # curently luminous efficiency is set to be randomised between typical ranges
    # TODO: investigate calculations for luminous efficiency and determine applicaility
    X[12] = random.random() * (tau_max - tau_min) + tau_min

    # masses    
    if mass_opt == 1:   
    # use ballistic mass from metadata
        ho=7160         # scale height of atmosphere in m
        X[6] = pow(0.5 * ho * 1.29 * X[9] / (np.sin(gamma) * alpha), 3)

    elif mass_opt ==2:  
    # random log sampling from 1g to m0_max    
        X[6] = 10**random.uniform(np.log10(0.001), np.log10(m0_max))  

    else:               
    # random initial mass from 0 to m_0 (uniform distribution)
        X[6] = random.random() * m0_max 

    
    ## extra info...
    X[27] = index
    X[28] = oindex
    X[29] = 1./N

    if len(uv_ECEF) > 0:
        new_pos_ECEF = X[0] * uv_ECEF.transpose() + np.asarray(x0_ECEF)
        grav, lat, lon, alt = tu.Gravity(np.vstack((new_pos_ECEF[0][0], new_pos_ECEF[0][1], new_pos_ECEF[0][2])))
        X[31] = lat
        X[32] = lon
        X[33] = alt   

    # print(X[0], X[3])

    return X

def particle_propagation(X, mu, tkm1, tk, fireball_info, obs_info, lum_info, index, N, frag, t_end, Q_c, m0_max, reverse, eci_bool, uv_ECEF=[], x0_ECEF=[]):
    """ performs non linear integration from tk to tk+1
        Inputs:
        X:       ARRAY : [1x42] Particle array
        mu:      DOUBLE: spin parameter
        tkm1:    DOUBLE: relative start integration time 
        tk:      DOUBLE: relative end integration time
        fireball_info:ARRAY: 
        obs_info:ARRAY : [2 x number of observers] list of observations 
                         [distance from start along triangulated trajectory, uncertainty]
        lum_info:ARRAY : [1 x 2 * number of observations] list of brightness observations
                         [magnitude_1, uncertainty_1,  ... magnitude_n, uncertainty_n] for n number of observations
        index:   INT   : index of parent
        N:       INT   : total number of particles
        frag:    BOOL  : is this a fragmentation timestep?
        t_end:   BOOL  : is this the final timestep? mass is allowed to become 0 if t_end
        Q_c:     ARRAY : [1x13] continuous process noise as a vector
        m0_max:  DOUBLE: maximum mass from initiation step (prediction cant give higher masses)
        reverse: BOOL  : is this filter being performed from relative t_end to t0?
        eci_bool:BOOL  : unused in this dim option
    """

    if X[6] < 0:
        X[29] = np.exp(-5000)
        return X
        # raise ValueError('cannot have a negative mass')

    ###### Prediction #####

    ## extract state vector
    init_x = copy.deepcopy(X[0:13]) 

    ## atmospheric parameters:
    [temp, po, atm_pres] = bf.Atm_nrlmsise_00(fireball_info)

    ## parameters to be passed to integration:
    param = [mu, po, fireball_info[6]]

    ## definite integral limits
    calc_time=[tkm1, tk]

    ## integration:
    with bf.stdout_redirected():
        ode_output = scipy.integrate.odeint(bf.NL_state_eqn_2d, init_x, calc_time, args = (param,)) 

    ## set new particle
    X[0:13] = ode_output[1]

   ## discretisation of process noise covariance:
    Qc = copy.deepcopy(Q_c) 
    Q_d = bf.Q_mx_1d(tkm1, tk,  init_x, mu, po,  Qc, reverse)
    X[13:26] = Q_d

    ## add noise to states
    X[0] = X[0] + random.gauss(0, sqrt(X[13])) # x position
    X[3] = X[3] + random.gauss(0, sqrt(X[16])) # x velocity 

    if frag:
        X[6] = rand_skew_norm(-6, X[6], sqrt(X[19]))   #X[6] + random.gauss(0, sqrt(X[19])) #
    
    else:
        if reverse:  
            ## TODO: test random skew norm for process noise addition     
            X[6] = rand_skew_norm(3, X[6], sqrt(abs(X[19])) )
            # X[6] = X[6] + random.gauss(0, sqrt(X[19])) # 

        else:            
            X[6] = rand_skew_norm(-3, X[6], sqrt(abs(X[19]))) 
            # X[6] = X[6] + random.gauss(0, sqrt(X[19])) # 


    X[9] = X[9] + random.gauss(0, sqrt(X[22])) # kappa
    X[10] = X[10] + random.gauss(0, sqrt(X[23])) # sig
    X[12] = X[12] + random.gauss(0, sqrt(X[25])) # tau
    
    X[30] = X[3]
    vel =  X[3]

    ## luminosity:
    ## TODO: there are a few different ways of calculating this. 
    ## There is a minimal difference but they are left here if needed.

    ## (1) -- equation for ablation and drag is:
    ## I = - tau (1 +2/(sig * v^2)) * v^2/2 * dm/dt : 

    Intensity1 = (- X[12] * (1 + 2 /(X[10] * pow(vel, 2))) * pow(vel, 2) / 2  * -abs((X[6] - init_x[6])/(tk- tkm1)))*1e7
    ## I       =  - tau   * (1 + 2 /(sigma * velocity^2))  * (velocity^2 / 2) *                      dm/dt          * conversion to watts
    
    ## (2) -- equation for just ablation is:
    ##  I = - tau * v^2 / 2 * dm/dt:
    
    Intensity2 = (- X[12] * pow(vel, 2) / 2 * -abs((X[6] - init_x[6])/(tk- tkm1)))*1e7
    
    ## (3) -- equation for ablation and drag without sigma involved:
    ## I = -tau (v^2 /2 dm/dt + m v dv/dt)

    Intensity3 = (- X[12] * (vel**2 / 2 * 
                 -abs((X[6] - init_x[6])/(tk- tkm1)) + 
                 X[6] * vel * -abs((vel - norm([init_x[3], init_x[4], init_x[5]]))/(tk- tkm1))))*1e7
    
    ## set which intensity result to use:
    Intensity = Intensity3

    X[25] = Intensity

    ## calculate visual magnitude that corresponds to the luminous intensity calculated

    ## conversion from absolute to visual magnitude
    ## depends on temperature. 1.95e10 for 4000K; 1.5e10 for 4500K. 
    ##See pg 365 of Ceplecha 1978
    ceplecha_number = 1.95e10
    X[11] = -2.5 * (np.log10(Intensity /ceplecha_number))

    ## TODO: calculate flight angle between t0 and t1
    #X[26] = #np.arccos(np.vdot([X[0], X[1], X[2]], [init_x[0], init_x[1], init_x[2]]) / (np.sqrt(X[0]**2 + X[1]**2 + X[2]**2)*np.sqrt(init_x[0]**2 + init_x[1]**2 + init_x[2]**2)))

    ## update particle index (don't change orig index X[28])
    X[27] = index


    ###### Weighting calculation #####

    ## get particle weight based on observations
    X[35], X[34] = Get_Weighting(X, obs_info, lum_info, N, t_end, m0_max, reverse)
        
    if len(uv_ECEF) > 0:
        new_pos_ECEF = X[0] * uv_ECEF.transpose() + np.asarray(x0_ECEF)
        grav, lat, longi, alt = tu.Gravity(np.vstack((new_pos_ECEF[0][0], new_pos_ECEF[0][1], new_pos_ECEF[0][2])))
        X[31] = lat
        X[32] = longi
        X[33] = alt  

    
    return X


def Get_Weighting(X, obs_info, lum_info, N, t_end, m0_max, reverse=False):
    """ for a given particle state, calculate the log likelihood of 
        the updated position prediction.
        Inputs:
        X:       ARRAY : [1x42] Particle array
        obs_info:ARRAY : [2 x number of observers] list of observations 
                         [distance from start along triangulated trajectory, uncertainty]
        lum_info:ARRAY : [1 x 2 * number of observations] list of brightness observations
                         [magnitude_1, uncertainty_1,  ... magnitude_n, uncertainty_n] for n number of observations
        N:       INT   : total number of particles
        t_end:   BOOL  : is this the final timestep? mass is allowed to become 0 if t_end
        m0_max:  DOUBLE: maximum mass from initiation step (prediction cant give higher masses)
    """

    ## initialise with equal weightings
    pos_weight = 1./N 
    lum_weight = 1./N

    if X[6] < 0 and t_end != True:
    ## if mass is <0 before the final timestep, give v. low weighting
        pos_weight = 0.
        lum_weight = 0.

    elif X[6] > 1.1 *m0_max and not reverse:
    ## dont allow mass to be greater than max mass if predicting 
    ## formward in time - give v. low weighting

        pos_weight = 0.
        lum_weight = 0.    
    
    else:
        ## if there are luminosities to compare with, calculate the luminous weighting
        if lum_info!= []:
            lum_info = np.reshape(lum_info, (-1, 2))

            Z = lum_info[:, 0]
            R = lum_info[:, 1]

            n_obs = len(Z)
            z_hat = np.asmatrix([X[11] for i in range(n_obs)])

            Z = np.asmatrix(np.reshape(Z, -1))
            #R = np.square(R)#/((tk-tkm1)/20))
            cov = np.asmatrix(np.diag(np.reshape(R,  -1)))
            
            for i in range(n_obs):
                ## if you are wanting to consider saturation, uncomment
                ## this section
                ## here there are two options if sensor was saturated 
                ## /!\ user defined value of saturation!
                
                # if z_hat[0, i]>-6.0:
                #     ## (1) if luminosity is less than saturation, set to zero
                #     ##     anything else is going to return the default value 
                #     ##     of 1/N (above)

                #     lum_weight *= -5000.

                #     ## (2) other option is to calculate a skew normal distribution. 
                #     ## this will unfavour really high values.
                #     ## lum_weight *= skew_norm_pdf(z_hat[0, i],Z[ 0, i],5,-3)

                # else:
                #     # seems to be a problem with multivariate Gaus calcultation...
                #     #lum_weight *= multi_var_gauss(z_hat.T, Z.T, cov, n_obs) 
                #     lum_weight *= Gaussian(z_hat[0,i], Z[0,i], R[i]) 

                lum_weight *= Gaussian(z_hat[0,i], Z[0,i], R[i]) 

        ## position observations

        observation = obs_info[:, 0]
        obs_err = obs_info[:, 1]
        n_obs = len(observation)
        z_hat =  np.matrix(X[0])    # predicted position

        for cam in range(n_obs):

            Z = observation[cam]
            R = obs_err[cam]**2

            pos_weight *= Gaussian(z_hat, Z, R)  

    return pos_weight, lum_weight
  
def Gaussian(z_hat, Z, R):
    """performs Gaussian PDF. 
        Inputs:
        z_hat - observation
        Z - mean
        R - variance"""

    diff = (z_hat - Z)

    ## calculates the probability of x for 1-dim Gaussian with mean mu and var. sigma
    return exp(- 0.5 * (np.dot(diff, diff.transpose())) / R) / sqrt(2.0 * pi * R)


def multi_var_gauss(pred, mean, cov, n_obs):
    """performs multivariate Gaussian PDF."""

    det_cov = np.linalg.det(cov)
    inv_cov = np.linalg.inv(cov)
    diff = pred - mean
    diff = np.asmatrix(abs(diff))
    
    ## multivariate equation:
    return pow((2*np.pi), -n_obs/2) * pow(det_cov, -.5) * np.exp(-.5*diff.T*inv_cov * diff)


def rand_skew_norm(fAlpha, fLocation = 0., var = 1., scale = 10.):

    """shg 2008-0919
        http://azzalini.stat.unipd.it/SN/faq.html

        Returns a random variable with skewed distribution
                fAlpha      = skew
                fLocation   = location
                fScale > 0  = scale
                """

    sigma = fAlpha / np.sqrt(scale + fAlpha**2)

    afRN = np.random.randn(2)
    u0 = afRN[0]
    v = afRN[1]
    u1 = sigma * u0 + np.sqrt(1 - sigma**2) * v

    if u0 >= 0:
        return u1*var + fLocation 
    return (-u1)*var + fLocation 

def skew_norm_pdf(x,e=0,w=1,a=0):
    ## a = shape (amount of skew)
    ## w = scale
    ## e = location
    ## adapated from:
    ## http://stackoverflow.com/questions/5884768/skew-normal-distribution-in-scipy
    
    t = (x-e) / w
    likelihood = 2.0 * w * scipy.stats.norm.pdf(t) * scipy.stats.norm.cdf(a*t)
    
    ## if likelihood -ve, set to ~0.
    if likelihood <= 0 or np.isnan(likelihood):
        print('caught a -ve likelihood in skew. setting to ~0')
        return np.exp(-500)
    else:
        ## return log of likelihoods
        return likelihood