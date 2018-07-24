#!/usr/bin/env python
"""
=============== 1D cartesian functions ===============

list of functions called by particle filter 
'main_MPI.py' when dim = 1.

1D analysis on a single, pre-triangulated 
trajectory file with X_geo, Y_geo and Z_geo 
information.

"""

# import modules

# general
from math import *
import copy, operator, random, time
import configparser, sys, os, argparse, glob

# science
import numpy as np
import scipy 
import scipy.integrate
from numpy.linalg import norm

# Astropy
from astropy.table import Table, Column, join, hstack
import astropy.units as u

# own
import bf_functions_3d as bf
import trajectory_utilities as tu

#------------------------------------------------------------------------------
# KEY (params in [] are not used in 1D filter)
# 0, [1, 2],   3,   [4,   5], 6,  7, 8,     9,    10,     11,  12,  
# x, [y, z], v_x, [v_y, v_z], m, cd, A, kappa, sigma, bright, tau, 
#
#  13,  [14,  15],    16,    [17,    18],  19,   20,   21,  22,    23,       24,    25,     
# Q_x, [Q_y, Q_z], Q_v_x, [Q_v_y, Q_v_z], Q_m, Q_cd, Q_cl, Q_K, Q_sig, Q_bright, Q_tau,  
#
#    26,                                   
# gamma/ new:temp lum weighting / temp:rho, 
#
#          27,        28 ,     29,       30,    31,     
#parent index, orig index, weight, vel_norm, p_lat, 
#
#     32,       33,    34,    35 
# p_long, p_height, pos_weight, lum_weight
#------------------------------------------------------------------------------



def Initialise(x0, v0, index, oindex, N, P, params, date_info, mass_opt=3, m0_max=2000., gamma= 0.7854, eci_bool=False, uv_ECEF=[], x0_ECEF=[]):
    """ create a random particle to represent a meteoroid. Random 
    distance along path, mass, velocity, ablation parameter and 
    shape-density parameter are created with Q_c noise. according to 
    global set ranges"""

    ## although this is a 1D filter, the number of columns remains the same
    ## so that outputs are consistent. y and z component columns are left 
    ## unfilled.
    X = np.zeros(42)
    [A_min, A_max, pm_mean, pm_std, random_meteor_type, tau_min, tau_max] = params

    ## default filter creates a spread of initial particle state values
    X[0] = random.gauss(x0, P[0])     # initial position in x using Gaussian pdf. P values are already sqrt(cov)

    X[3] = random.gauss(v0, P[3])     # initial velocity in x using Gaussian pdf. P values are already sqrt(cov)
    
    ## TODO: trying to correlate shape and drag
    # X[8] = random.random() * (A_max - A_min) + A_min
    # X[7] = 0.98182 * X[8]**2 - 1.7846 * X[8] + 1.6418
    X[8] = 1.5  #random.random() * (A_max - A_min) + A_min
    X[7] = 1.0  #0.98182 * X[8]**2 - 1.7846 * X[8] + 1.6418

    particle_choices = random.choice(random_meteor_type)
    X[26] = random.gauss(pm_mean[particle_choices], pm_std[particle_choices])
    X[9]  = X[8] * X[7]/pow(X[26], 2/3.)   #A * cd / density^(2/3.)
    
    ## TODO: abalation coefficient
    ## values for sigma are supposedly correlated to density
    ## ranges here are taken from table 17 of bible 
    # if X[26] > 5000:
    #     X[10] = random.gauss(0.07e-6, 0.01e-6)
    # elif X[26] > 2500:
    #     X[10] = random.gauss(0.014e-6, 0.005e-6)
    # elif X[26] > 1500:
    #     X[10] = random.gauss(0.042e-6, 0.005e-6)
    # else:
    #     X[10] = random.gauss(0.1e-6, 0.05e-6)
    X[10] = 2* 0.014e-6

    ## there are calculations for luminous efficiency too... 
    ## TODO: investigate magic numbers in those equations and 
    ## maybe apply here and in state equations (yeah nah)
    X[12] = 0.001 #random.random() * (tau_max - tau_min) + tau_min

    ## masses    
    if mass_opt == 1:   # use ballistic mass from metadata
        ho=7160         # scale height of atmosphere in m
        X[6] = pow(0.5 * ho * 1.29 * X[9] / (np.sin(gamma) * alpha), 3)
    elif mass_opt ==2:  # random log sampling from 1g to m0_max    
        X[6] = 10**random.uniform(np.log10(0.001), np.log10(m0_max))  
    else:               # random initial mass from 0 to m_0 (unifirm distribution)
        X[6] = random.random() * m0_max 
    
    ## extra info...
    X[27] = index
    X[28] = oindex
    X[29] = 1./N

    if len(uv_ECEF) > 0:
        new_pos_ECEF = X[0] * uv_ECEF.transpose() + np.asarray(x0_ECEF)
        grav, lat, longi, alt = tu.Gravity(np.vstack((new_pos_ECEF[0][0], new_pos_ECEF[0][1], new_pos_ECEF[0][2])))
        X[31] = lat
        X[32] = longi
        X[33] = alt   

    return X

def Prdct_Upd(X, mu, tkm1, tk, fireball_info, obs_info, lum_info, index, N, frag, t_end, Q_c, m0_max, reverse, eci_bool, uv_ECEF=[], x0_ECEF=[]):
    """ performs non linear integration of time step of fireball entry """
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
    ode_output = scipy.integrate.odeint(bf.NL_state_eqn_2d, init_x, calc_time, args = (param,)) 

    ## set new particle
    X[0:13] = ode_output[1]

    ## discretisation of covariance noise:
    Qc = copy.deepcopy(Q_c) #WTF python! If I don't do this, Q_mx_3d changes array!!
    Q_d = bf.Q_mx_2d(tkm1, tk,  init_x, mu, po, Qc)

    X[13:26] = Q_d

    ## add noise to states
    X[0] = X[0] + random.gauss(0, sqrt(X[13])) # x position
    #X[1] = X[1] + random.gauss(0, sqrt(X[14])) # y position
    #X[2] = X[2] + random.gauss(0, sqrt(X[15])) # z position
    X[3] = X[3] + random.gauss(0, sqrt(X[16])) # x velocity 
    #X[4] = X[4] + random.gauss(0, sqrt(X[17])) # y velocity 
    #X[5] = X[5] + random.gauss(0, sqrt(X[18])) # z velocity 
    # X[3] = X[3] - abs(random.gauss(0, sqrt(X[16])) )# x velocity 
    # X[6] = X[6] - abs(random.gauss(0, sqrt(X[19]))) #
    if frag:
        X[6] = rand_skew_norm(-6, X[6], sqrt(X[19]))   #X[6] + random.gauss(0, sqrt(X[19])) #
    
    else:
        if rev:
            X[6] = rand_skew_norm(3, X[6], sqrt(X[19])) 
        else:
            ## currently this is set to using a skew norm distribution. 
            ## Gaussian is commented out below if needed 
            X[6] = rand_skew_norm(-3, X[6], sqrt(X[19])) 
            # X[6] = X[6] + random.gauss(0, sqrt(X[19])) # 


    X[9] = X[9] + random.gauss(0, sqrt(X[22])) # kappa
    X[10] = X[10] + random.gauss(0, sqrt(X[23])) # sig
    X[12] = X[12] + random.gauss(0, sqrt(X[25])) # tau
 
    ## luminosity:
    ## TODO: there are a few different ways of calculating this. There is a minimal difference but they are left here if needed.
    vel = X[3]

    ## (1) -- equation for ablation and drag is:
    ## I = - tau (1 +2/(sig * v^2)) * v^2/2 * dm/dt : 

    ## I         =  - tau   * (1 + 2 /( sigma* velocity^2))  * (velocity^2 / 2) *                      dm/dt          * conversion to watts
    # Intensity1 = (- X[12] * (1 + 2 /(X[10] * pow(vel, 2))) * pow(vel, 2) / 2  * -abs((X[6] - init_x[6])/(tk- tkm1)))*1e7
    
    ## (2) -- equation for just ablation is:
    ##  I = - tau * v^2 / 2 * dm/dt:
    
    #Intensity2 = (- X[12] * pow(vel, 2) / 2 * -abs((X[6] - init_x[6])/(tk- tkm1)))*1e7
    
    ## (3) -- equation for ablation and drag without sigma involved:
    ## I = -tau (v^2 /2 dm/dt + m v dv/dt)

    Intensity3 = (- X[12] * (vel**2 / 2 * 
                 -abs((X[6] - init_x[6])/(tk- tkm1)) + 
                 X[6] * vel * -abs((vel - norm([init_x[3], init_x[4], init_x[5]]))/(tk- tkm1))))*1e7
    
    ## set which intensity result to use:
    Intensity = Intensity3

    X[24] = Intensity

    ## calculate visual magnitude that corresponds to the luminos intensity calculated

    ## magic number for conversion from absolute to visual magnitude
    ## depends on temperature. 1.95e10 for 4000K; 1.5e10 for 4500K. 
    ##See pg 365 of bible
    ceplecha_magic_number = 1.95e10
    X[11] = -2.5 * (np.log10(Intensity /ceplecha_magic_number))
 
    ## update particle index (don't change orig index X[28])
    X[27] = index

    ## Measurement Update:
    ## get particle weight based on observations
    X[29], X[34] = Get_Weighting(X, obs_info, lum_info, N, t_end, m0_max)
        
    if len(uv_ECEF) > 0:
        new_pos_ECEF = X[0] * uv_ECEF.transpose() + np.asarray(x0_ECEF)
        grav, lat, longi, alt = tu.Gravity(np.vstack((new_pos_ECEF[0][0], new_pos_ECEF[0][1], new_pos_ECEF[0][2])))
        X[31] = lat
        X[32] = longi
        X[33] = alt  

    return X


def Get_Weighting(X, obs_info, lum_info, N, t_end, m0_max):
    ## initialise with equal weightings
    pos_weight = 1./N 
    lum_weight = 1./N

    ## is mass is <0 before the final timestep, give zero weight
    if X[6] < 0 and t_end != True:
        pos_weight = -5000#-500.  
        lum_weight = -5000

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
                ## this section as an alternative to line 279
                # ## here there are two options if sensor was saturated 
                # ## /!\ user defined value of saturation!
                
                # if z_hat[0, i]>-6.0:
                #     ## (1) if luminosity is less than saturation, set to zero
                #     ##     anything else is going to return the default value 
                #     ##     of 1/N (above)

                #     lum_weight *= np.exp(-5000.)

                #     ## (2) other option is to calculate a skew normal distribution. 
                #     ## this will unfavour really high values.
                #     ## lum_weight *= skew_norm_pdf(z_hat[0, i],Z[ 0, i],5,-3)

                # else:
                #     # seems to be a problem with multivariate Gaus calcultation...
                #     #lum_weight *= multi_var_gauss(z_hat.T, Z.T, cov, n_obs) 
                #     lum_weight *= Gaussian(z_hat[0,i], Z[0,i], R[i]) 

                lum_weight += Gaussian(z_hat[0,i], Z[0,i], R[i]) 

        ## position observations

        observation = obs_info[:, 0]
        obs_err = obs_info[:, 1]
        n_obs = len(observation)
        z_hat =  np.matrix(X[0])    # predicted position

        for cam in range(n_obs):

            Z = observation[cam]
            R = obs_err[cam]**2


            pos_weight += Gaussian(z_hat, Z, R)  


    return pos_weight, lum_weight
  
def Gaussian(z_hat, Z, R):
    """performs Gaussian PDF. 
        Inputs:
        z_hat - observation
        Z - mean
        R - variance"""

    diff = (z_hat - Z)

    ## calculates the probability of x for 1-dim Gaussian with mean mu and var. sigma
    likelihood = exp(- 0.5 * (np.dot(diff, diff.transpose())) / R) / sqrt(2.0 * pi * R)
    
    ## if likelihood is too small (<1e-300 I think is the python limit) or nan, set to ~0.
    if likelihood <= 0 or np.isnan(likelihood):
        return -5000.
    else:
        return  np.log(likelihood)

def multi_var_gauss(pred, mean, cov, n_obs):
    """performs multivariate Gaussian PDF."""

    det_cov = np.linalg.det(cov)
    inv_cov = np.linalg.inv(cov)
    diff = pred - mean
    diff = np.asmatrix(abs(diff))
    
    ## multivariate equation:
    likelihood = pow((2*np.pi), -n_obs/2) * pow(det_cov, -.5) * np.exp(-.5*diff.T*inv_cov * diff)
    
    ## if likelihood is too small (<1e-300 I think is the python limit) or nan, set to ~0.
    if likelihood <= 0 or np.isnan(likelihood):
        #print('caught a -ve likelihood in multivar. setting to ~0')
        return -5000.
    else:
        return  np.log(likelihood)

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