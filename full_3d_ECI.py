#!/usr/bin/env python
"""
=============== 3D Ray functions ===============

list of functions called by particle filter 
'main_MPI.py' when dim = 3.

This performs a 3D analysis on calibrated astrometric 
observations (altitude and azimuth data required) 

"""

# import modules

# general
from math import *
import copy, operator, random, time

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

#------------------------------------------------------------------------------
# KEY:
# 0, 1, 2,   3,   4,   5, 6,  7, 8,     9,    10,     11,  12,  
# x, y, z, v_x, v_y, v_z, m, cd, A, kappa, sigma, bright, tau, 
#
#  13,  14,  15,    16,    17,    18,  19,   20,   21,  22,    23,       24,    25,     
# Q_x, Q_y, Q_z, Q_v_x, Q_v_y, Q_v_z, Q_m, Q_cd, Q_cl, Q_K, Q_sig, Q_bright, Q_tau,  
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


def Initialise(x0, v0, index, oindex, N, P, params, alpha, date_info, mass_opt=3, m0_max=2000., gamma= 0.7854, eci_bool=False):
    """ create a random particle to represent a meteoroid. Random 
    distance along path, mass, velocity, ablation parameter and 
    shape-density parameter are created with Q_c noise. according to 
    global set ranges"""

    X = np.zeros(42)

    ## range params defined on line 230 of main script.
    [A_min, A_max, pm_mean, pm_std, random_meteor_type, tau_min, tau_max] = params

    ## default filter creates a spread of initial particle state values
    X[0] = random.gauss(x0[0], P[0])     # initial position in x using Gaussian pdf. P values are already sqrt(cov)
    X[1] = random.gauss(x0[1], P[1])     # initial position in Y using Gaussian pdf. P values are already sqrt(cov)
    X[2] = random.gauss(x0[2], P[2])     # initial position in Z using Gaussian pdf. P values are already sqrt(cov)

    X[3] = random.gauss(v0[0], P[3]) # initial velocity in x using Gaussian pdf. P values are already sqrt(cov)
    X[4] = random.gauss(v0[1], P[4]) # initial velocity in y using Gaussian pdf. P values are already sqrt(cov)
    X[5] = random.gauss(v0[2], P[5]) # initial velocity in z using Gaussian pdf. P values are already sqrt(cov)

    T_jd = date_info[3].jd #+ fireball_info[7]/(3600*24)#+ TimeDelta(fireball_info[7], format='sec')

    if eci_bool:
        X[36:39] = X[0:3]
        X[39:42] = X[3:6]

        pos, vel = tu.ECI2ECEF(np.vstack((X[0], X[1], X[2])), np.vstack((X[3], X[4], X[5])), T_jd)
        X[0:3] = pos.transpose()
        X[3:6] = vel.transpose()

    else:
        pos, vel = tu.ECEF2ECI(np.vstack((X[0], X[1], X[2])), np.vstack((X[3], X[4], X[5])), T_jd)
        X[36:39] = pos.transpose()
        X[39:42] = vel.transpose()

    # TODO: trying to correlate shape and drag
    X[8] = random.random() * (A_max - A_min) + A_min
    X[7] = 1.3  #0.98182 * X[8]**2 - 1.7846 * X[8] + 1.6418

    particle_choices = random.choice(random_meteor_type)
    X[26] = random.gauss(pm_mean[particle_choices], pm_std[particle_choices])
    X[9]  = X[8] * X[7]/pow(X[26], 2/3.)   #A * cd / density^(2/3.)
    
    # TODO: abalation coefficient
    # values for sigma are supposedly correlated to density
    # ranges here are taken from table 17 of bible 
    if X[26] > 5000:
        X[10] = random.gauss(0.07e-6, 0.01e-6)
    elif X[26] > 2500:
        X[10] = random.gauss(0.014e-6, 0.005e-6)
    elif X[26] > 1500:
        X[10] = random.gauss(0.042e-6, 0.005e-6)
    else:
        X[10] = random.gauss(0.1e-6, 0.05e-6)

    #X[10] = random.random() * (sigma_max - sigma_min) + sigma_min #* X[7] 0.042*1e-6#
    #X[11] = random.gauss(x0[3], P[5])

    # there are calculations for luminous efficiency too... 
    # TODO: investigate magic numbers in those equations and 
    # maybe apply here and in state equations (yeah nah)
    X[12] = random.random() * (tau_max - tau_min) + tau_min

    # masses    
    if mass_opt == 1:   # use ballistic mass from metadata
        ho=7160         # scale height of atmosphere in m
        X[6] = pow(0.5 * ho * 1.29 * X[9] / (np.sin(gamma) * alpha), 3)
    elif mass_opt ==2:  # random log sampling from 1g to m0_max    
        X[6] = 10**random.uniform(np.log10(0.001), np.log10(m0_max))  
    else:               # random initial mass from 0 to m_0 (unifirm distribution)
        X[6] = random.random() * m0_max 


    # initial flight angle given to be 45 degrees. 
    #X[26] = gamma
                    #np.arccos(np.vdot([x0[0], x0[1], x0[2]], [x1[0], x1[1], x1[2]]) /
                    #            (np.sqrt(x0[0]**2 + x0[1]**2 + x0[2]**2)*np.sqrt(x1[0]**2 + x1[1]**2 + x1[2]**2)))


    X[27] = index       # parent index
    X[28] = oindex      # original index (same as parent index if this run starts at t0)
    X[29] = 1./N        # initial particle weighting
    X[30] = np.linalg.norm([X[39], X[40], X[41]])  # vel norm
    # get LLH for plotting
    grav, lat, longi, alt = tu.Gravity([X[0], X[1], X[2]])
    X[31] = lat
    X[32] = longi
    X[33] = alt
    #X[34] = 0. #X[9] * X[7]  # kappa
    #X[35] = 0. #X[10] / X[7] # sigma

    # print(X)

    return X

def Prdct_Upd(X, mu, tkm1, tk, fireball_info, obs_info, lum_info, index, N, frag, t_end, Q_c, m0_max, reverse=False, eci_bool=False):
    """ performs non linear integration of time step of fireball entry """
    if X[6] < 0:
        print(X)
        raise ValueError('cannot have a negative mass')

    ###### Prediction #####
    # extract state vector
    init_x = copy.deepcopy(X[0:13]) 
    # print(init_x)
    # print(X)

    ## constants: 
    grav, lat, lon, alt = tu.Gravity([X[0], X[1], X[2]])
    fireball_info[0:3] = [lat[0], lon[0], alt[0]]

    ## atmospheric parameters:
    [temp, po, atm_pres] = bf.Atm_nrlmsise_00(fireball_info)

    ## parameters to be passed to integration:
    param = [mu, po, grav]

    ## definite integral limits
    calc_time=[tkm1, tk]

    ##integration:
    T_jd = fireball_info[6].jd + fireball_info[7]/(3600*24)#+ TimeDelta(fireball_info[7], format='sec')
    # pos, vel = tu.ECEF2ECI(np.vstack((X[0], X[1], X[2])), np.vstack((X[3], X[4], X[5])), T_jd)
    # X[0:3] = pos.transpose()
    # X[3:6] = vel.transpose()
    init_x[0:3] = X[36:39]
    init_x[3:6] = X[39:42]

    with bf.stdout_redirected():
        ode_output = scipy.integrate.odeint(bf.NL_state_eqn_3d, init_x, calc_time, args = (param,)) 
    # print(ode_output)
    ## if you want to output gravity values:
    # if tk == t_end:
    # nograv = scipy.integrate.odeint(bf.NL_state_eqn_3d_nog, init_x, calc_time, args = (param,)) 
    # grav_dif = ode_output[1] - nograv[1]
    # dist_trav = ode_output[1] - init_x
    # grav = np.sqrt(grav_dif[0]**2 + grav_dif[1]**2 + grav_dif[2]**2)
    # dist = np.sqrt(dist_trav[0]**2 + dist_trav[1]**2 + dist_trav[2]**2)
    

    ## set new particle
    X[0:13] = ode_output[1]

    ## discretisation of covariance noise:
    Qc = copy.deepcopy(Q_c) #WTF python! If I don't do this, Q_mx_3d changes array!!
    Q_d = bf.Q_mx_3d(tkm1, tk,  init_x, mu, po, grav, Qc, reverse)

    
    X[13:26] = Q_d

    ## add noise to states
    X[0] = X[0] + random.gauss(0, sqrt(abs(X[13]))) # x position
    X[1] = X[1] + random.gauss(0, sqrt(abs(X[14]))) # y position
    X[2] = X[2] + random.gauss(0, sqrt(abs(X[15]))) # z position
    
    X[3] = X[3] + random.gauss(0, sqrt(abs(X[16]))) # x velocity
    X[4] = X[4] + random.gauss(0, sqrt(abs(X[17]))) # y velocity
    X[5] = X[5] + random.gauss(0, sqrt(abs(X[18]))) # z velocity

    


    if frag:
        X[3] = rand_skew_norm(-3, X[3], sqrt(X[16])) # x velocity
        X[4] = rand_skew_norm(-3, X[4], sqrt(X[17])) # y velocity
        X[5] = rand_skew_norm(-3, X[5], sqrt(X[18])) # z velocity

        X[6] = rand_skew_norm(-3, X[6], sqrt(abs(X[19])))   #X[6] + random.gauss(0, sqrt(X[19])) #
    
    else:
        # print(X[6], X[19])
        ## currently this is set to using a skew norm distribution. 
        ## Gaussian is commented out below if needed
        if reverse:  
            # X[3] = rand_skew_norm(2, X[3], sqrt(X[16])) # x velocity
            # X[4] = rand_skew_norm(2, X[4], sqrt(X[17])) # y velocity
            # X[5] = rand_skew_norm(2, X[5], sqrt(X[18])) # z velocity          
            
            X[6] = rand_skew_norm(3, X[6], sqrt(abs(X[19])) )

        else:
            # X[3] = rand_skew_norm(-2, X[3], sqrt(X[16])) # x velocity
            # X[4] = rand_skew_norm(-2, X[4], sqrt(X[17])) # y velocity
            # X[5] = rand_skew_norm(-2, X[5], sqrt(X[18])) # z velocity
            
            X[6] = rand_skew_norm(-3, X[6], sqrt(abs(X[19]))) 
        # print(X[6], X[19])
        # X[6] = X[6] + random.gauss(0, sqrt(X[19])) # 

    #X[7] = X[7] + random.gauss(0, sqrt(X[20])) # cd
    #X[8] = X[8] + random.gauss(0, sqrt(X[21])) # cl
    X[9] = X[9] + random.gauss(0, sqrt(abs(X[22]))) # kappa
    X[10] = X[10] + random.gauss(0, sqrt(abs(X[23]))) # sig
    X[12] = X[12] + random.gauss(0, sqrt(abs(X[25]))) # tau

    # get vel_norm
    vel = norm([X[3], X[4], X[5]])
    X[30] = vel

    ## luminosity:
    ## TODO: there are a few different ways of calculating this. There is a minimal difference but they are left here if needed.

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
        #X[11] = X[11] + random.gauss(0, sqrt(X[24])) # luminosity

    #print('magnitude:',-5/2 * (np.log10(Intensity /1.95e10) ), 'or, it could be ', -5/2 * (np.log10(Intensity /1.5e10) ),  '   or, using magic number 10.185:', -5/2 * np.log10(Intensity-10.185))

    # calculate flight angle between t0 and t1
    #X[26] = 0#np.arccos(np.vdot([X[0], X[1], X[2]], [init_x[0], init_x[1], init_x[2]]) / (np.sqrt(X[0]**2 + X[1]**2 + X[2]**2)*np.sqrt(init_x[0]**2 + init_x[1]**2 + init_x[2]**2)))

    # update particle index (don't change orig index X[28])
    X[27] = index

    ## Measurement Update:
    # get particle weight based on observations
    X[29], X[34] = Get_Weighting(X, obs_info, lum_info, N, t_end, m0_max,reverse, T_jd)

    #X[34] = X[9] * X[7]  # kappa
    #X[35] = X[10] / X[7] # sigma

    # pos, vel = tu.ECI2ECEF(np.vstack((X[0], X[1], X[2])), np.vstack((X[3], X[4], X[5])), T_jd)
    # X[0:3] = pos.transpose()
    # X[3:6] = vel.transpose()

    X[36:39] = X[0:3] 
    X[39:42] = X[3:6]

    T_jd = fireball_info[6].jd + fireball_info[7]/(3600*24)#+ TimeDelta(fireball_info[7], format='sec')
    pos, vel = tu.ECI2ECEF(np.vstack((X[0], X[1], X[2])), 
                           np.vstack((X[3], X[4], X[5])), T_jd)
    X[0:3] = pos.transpose()
    X[3:6] = vel.transpose()
    # print(init_x, X)

    ## recalculate LLH for updated particle
    grav, lat, longi, alt = tu.Gravity([X[0], X[1], X[2]])
    X[31] = lat
    X[32] = longi
    X[33] = alt
    # print(X)
    # ppp
    return X





# def Get_Weighting(X, obs_info, lum_info, N, t_end, m0_max, reverse, T_jd):
    
#     # initialise with equal weightings
#     pos_weight = 1./N 
#     lum_weight = 1./N

#     # is mass is <0 before the final timestep, give zero weight
#     if X[6] < 0 and t_end != True:
#         pos_weight = -5000#-500.  
#         lum_weight = -5000
#     elif X[6] > 1.1 *m0_max and not reverse:
#         pos_weight = -5000#-500.  
#         lum_weight = -5000    
    
#     else:
#         ## if there are luminosities to compare with, calculate the luminous weighting
#         if lum_info!= []:
#             lum_info = np.reshape(lum_info, (-1, 2))

#             Z = lum_info[:, 0]
#             R = lum_info[:, 1]

#             # using a 1D Gaussian...
#             # for i in range(n_obs):
#             #     print(Lum_Gaussian(z_hat, Z[i], R[i]**2) / 10.)
#             #     weight += Lum_Gaussian(z_hat, Z[i], R[i]**2) / 10.
#             #     print(i, 'weight', weight)

#             n_obs = len(Z)
#             z_hat = np.asmatrix([X[11] for i in range(n_obs)])

#             Z = np.asmatrix(np.reshape(Z, -1))
#             #R = np.square(R)#/((tk-tkm1)/20))
#             cov = np.asmatrix(np.diag(np.reshape(R,  -1)))
            
#             for i in range(n_obs):
#                 ## if you are wanting to consider saturation, uncomment
#                 ## this section as an alternative to line 279
#                 # ## here there are two options if sensor was saturated 
#                 # ## /!\ user defined value of saturation!
                
#                 # if z_hat[0, i]>-6.0:
#                 #     ## (1) if luminosity is less than saturation, set to zero
#                 #     ##     anything else is going to return the default value 
#                 #     ##     of 1/N (above)

#                 #     lum_weight *= np.exp(-5000.)

#                 #     ## (2) other option is to calculate a skew normal distribution. 
#                 #     ## this will unfavour really high values.
#                 #     ## lum_weight *= skew_norm_pdf(z_hat[0, i],Z[ 0, i],5,-3)

#                 # else:
#                 #     # seems to be a problem with multivariate Gaus calcultation...
#                 #     #lum_weight *= multi_var_gauss(z_hat.T, Z.T, cov, n_obs) 
#                 #     lum_weight *= Gaussian(z_hat[0,i], Z[0,i], R[i]) 

#                 lum_weight += Gaussian(z_hat[0,i], Z[0,i], R[i]) 

#         ## position observations
#         observation = obs_info[:, 0:2]
#         camera = obs_info[:, 2:5]   #LLH
#         R = obs_info[:, 5:]
#         n_obs = len(observation)

#         Cameras_LLH = obs_info[:, 2:5]   #LLH

#         for cam in range(n_obs):

#             obs_az = observation[cam, 0]
#             obs_el =  observation[cam, 1]
#             obs_az_error =  R[cam, 0]
#             obs_el_error =  R[cam, 1]
#             z_az, z_el = Part2AltAz_ECI(X[0], X[1], X[2], Cameras_LLH[cam, :], T_jd)
#             print(z_az, z_el, obs_az, obs_el, obs_az_error, obs_el_error)

#             z_hat = np.array([z_az, z_el])
#             Z = np.array([[obs_az], [obs_el]])
#             cov = np.array([obs_az_error**2, obs_el_error**2])
#             cov = np.asmatrix(np.diag(cov))
#             pos_weight += multi_var_gauss_angles(z_hat, Z, cov, 2)

#             print('weighting',pos_weight)

#     return pos_weight, lum_weight


# def multi_var_gauss_angles(pred, mean, cov, n_obs):
#     """performs multivariate Gaussian PDF. using angles between vectors."""

#     det_cov = np.linalg.det(cov)
#     inv_cov = np.linalg.inv(cov)
#     diff = pred - mean
#     diff = (diff + np.pi) % (np.pi*2) - np.pi       # gives 355' - 004' --> 9' (acute angle through 2pi) in degrees: (diff + 180) % 360 - 180
#     diff = np.asmatrix(abs(diff))
    
#     # multivariate equation:
#     likelihood = pow((2*np.pi), -n_obs/2) * pow(det_cov, -.5) * np.exp(-.5*diff.T*inv_cov * diff)
#     print('nonlog', likelihood)
#     # if likelihood is too small (<1e-300 I think is the python limit) or nan, set to ~0.
#     if likelihood <= 0 or np.isnan(likelihood):
#         #print('caught a -ve likeihood in multivar. setting to ~0')
#         return 0.
#     else:
#         return likelihood
  
# def Gaussian(z_hat, Z, R):
#     """performs Gaussian PDF. 
#         Inputs:
#         z_hat - observation
#         Z - mean
#         R - variance"""

#     diff = (z_hat - Z)

#     # calculates the probability of x for 1-dim Gaussian with mean mu and var. sigma
#     likelihood = exp(- 0.5 * (np.dot(diff, diff.transpose())) / R) / sqrt(2.0 * pi * R)
    
#     # if likelihood is too small (<1e-300 I think is the python limit) or nan, set to ~0.
#     if likelihood <= 0 or np.isnan(likelihood):
#         return 0.
#     else:
#         return likelihood

# def Gaussian_angles(z_hat, Z, R):
#     """performs Gaussian PDF. 
#         Inputs:
#         z_hat - observation
#         Z - mean
#         R - variance"""

#     diff = (z_hat - Z)
#     diff = (diff + np.pi) % (np.pi*2) - np.pi  # computes the angular distance through pi

#     # calculates the probability of x for 1-dim Gaussian with mean mu and var. sigma
#     likelihood = exp(- 0.5 * (np.dot(diff, diff.transpose())) / R) / sqrt(2.0 * pi * R)
    
#     # if likelihood is too small (<1e-300 I think is the python limit) or nan, set to ~0.
#     if likelihood <= 0 or np.isnan(likelihood):
#         return 0.
#     else:
#         # return log of likelihoods
#         return likelihood

#### LOGS::::
def Get_Weighting(X, obs_info, lum_info, N, t_end, m0_max, reverse, T_jd):
    
    # initialise with equal weightings
    pos_weight = 1./N 
    lum_weight = 1./N

    # is mass is <0 before the final timestep, give zero weight
    if X[6] < 0 and t_end != True:
        pos_weight = -5000#-500.  
        lum_weight = -5000
    elif X[6] > 1.1 *m0_max and not reverse:
        pos_weight = -5000#-500.  
        lum_weight = -5000    
    
    else:
        ## if there are luminosities to compare with, calculate the luminous weighting
        if lum_info!= []:
            lum_info = np.reshape(lum_info, (-1, 2))

            Z = lum_info[:, 0]
            R = lum_info[:, 1]

            # using a 1D Gaussian...
            # for i in range(n_obs):
            #     print(Lum_Gaussian(z_hat, Z[i], R[i]**2) / 10.)
            #     weight += Lum_Gaussian(z_hat, Z[i], R[i]**2) / 10.
            #     print(i, 'weight', weight)

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
        observation = obs_info[:, 0:2]
        camera = obs_info[:, 2:5]   #LLH
        R = obs_info[:, 5:]
        n_obs = len(observation)

        Cameras_LLH = obs_info[:, 2:5]   #LLH

        for cam in range(n_obs):

            obs_az = observation[cam, 0]
            obs_el =  observation[cam, 1]
            obs_az_error =  R[cam, 0]
            obs_el_error =  R[cam, 1]
            z_az, z_el = Part2AltAz_ECI(X[0], X[1], X[2], Cameras_LLH[cam, :], T_jd)
            # print(z_az, z_el, obs_az, obs_el, obs_az_error, obs_el_error)

            z_hat = np.array([z_az, z_el])
            Z = np.array([[obs_az], [obs_el]])
            cov = np.array([obs_az_error**2, obs_el_error**2])
            cov = np.asmatrix(np.diag(cov))
            pos_weight += multi_var_gauss_angles(z_hat, Z, cov, 2)

            # print('weighting',pos_weight)

    return pos_weight, lum_weight


def multi_var_gauss_angles(pred, mean, cov, n_obs):
    """performs multivariate Gaussian PDF. using angles between vectors."""

    det_cov = np.linalg.det(cov)
    inv_cov = np.linalg.inv(cov)
    diff = pred - mean
    diff = (diff + np.pi) % (np.pi*2) - np.pi       # gives 355' - 004' --> 9' (acute angle through 2pi) in degrees: (diff + 180) % 360 - 180
    diff = np.asmatrix(abs(diff))
    
    # multivariate equation:
    likelihood = pow((2*np.pi), -n_obs/2) * pow(det_cov, -.5) * np.exp(-.5*diff.T*inv_cov * diff)
    # print('nonlog', likelihood )
    # if likelihood is too small (<1e-300 I think is the python limit) or nan, set to ~0.
    if likelihood <= 0 or np.isnan(likelihood):
        #print('caught a -ve likeihood in multivar. setting to ~0')
        return -50000
    else:
        # print('log', np.log(likelihood))
        return np.log(likelihood)
  
def Gaussian(z_hat, Z, R):
    """performs Gaussian PDF. 
        Inputs:
        z_hat - observation
        Z - mean
        R - variance"""

    diff = (z_hat - Z)

    # calculates the probability of x for 1-dim Gaussian with mean mu and var. sigma
    likelihood = exp(- 0.5 * (np.dot(diff, diff.transpose())) / R) / sqrt(2.0 * pi * R)
    
    # if likelihood is too small (<1e-300 I think is the python limit) or nan, set to ~0.
    if likelihood <= 0 or np.isnan(likelihood):
        return -50000
    else:
        return np.log(likelihood)

def Gaussian_angles(z_hat, Z, R):
    """performs Gaussian PDF. 
        Inputs:
        z_hat - observation
        Z - mean
        R - variance"""

    diff = (z_hat - Z)
    diff = (diff + np.pi) % (np.pi*2) - np.pi  # computes the angular distance through pi

    # calculates the probability of x for 1-dim Gaussian with mean mu and var. sigma
    likelihood = exp(- 0.5 * (np.dot(diff, diff.transpose())) / R) / sqrt(2.0 * pi * R)
    
    # if likelihood is too small (<1e-300 I think is the python limit) or nan, set to ~0.
    if likelihood <= 0 or np.isnan(likelihood):
        return -5000
    else:
        # return log of likelihoods
        return np.log(likelihood)

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
    
    # if likelihood -ve, set to ~0.
    if likelihood <= 0 or np.isnan(likelihood):
        print('caught a -ve likelihood in skew. setting to ~0')
        return np.exp(-500)
    else:
        # return log of likelihoods
        return likelihood


def Part2AltAz_ECI(x, y, z, obs_LLH, T_jd):
    """ calculate the alt/el of a particle at a geocentric 
        ECEF position (x, y, z) in relation to a camera with 
        a lat,long,height of obs_LLH
    """
    
    ## Convert the camera's angular coords to radians
    #obs_LLH[0] = np.deg2rad(obs_LLH[0] )
    #obs_LLH[1] = np.deg2rad(obs_LLH[1] )

    # Camera coordinates
    cam_ECEF = tu.LLH2ECEF(obs_LLH)
    cam_ECI = tu.ECEF2ECI_pos(cam_ECEF, T_jd)
    part_ECI = np.vstack((x, y, z))
    
    lat = float(obs_LLH[0]); lon = float(obs_LLH[1])

    z_hat_ECI = part_ECI - cam_ECI
    z_hat_ECEF = tu.ECI2ECEF_pos(z_hat_ECI, T_jd)


    # Compute the transformation matrix
    trans = tu.ECEF2ENU(lon, lat)

    # Current position in local ENU coords
    ENU = np.dot(trans, z_hat_ECEF)

    # Calculate the azimuth & elevation from the cameras to the data points
    alt = np.arctan2(ENU[2], np.sqrt(ENU[0]**2 + ENU[1]**2))
    azi = np.arctan2(ENU[0], ENU[1]) % (2 * np.pi)  # Always between 0 and 2pi
    

    return azi, alt

##########  End  ###################

## deprecated functions below (mainly for weightings in logs)

# this weighting is wexperimenatal ECI functions
# def Get_Weighting(X, obs_info, lum_info, N, t_end, m0_max, reverse, T_jd):
    
#     # initialise with equal weightings
#     pos_weight = 1./N 
#     lum_weight = 1./N

#     # is mass is <0 before the final timestep, give zero weight
#     if X[6] < 0 and t_end != True:
#         pos_weight = np.exp(-5000)#-500.  
#         lum_weight = np.exp(-5000)
#     elif X[6] > 1.1 *m0_max and not reverse:
#         pos_weight = np.exp(-5000)#-500.  
#         lum_weight = np.exp(-5000)     
    
#     else:
#         ## if there are luminosities to compare with, calculate the luminous weighting
#         if lum_info!= []:
#             lum_info = np.reshape(lum_info, (-1, 2))

#             Z = lum_info[:, 0]
#             R = lum_info[:, 1]

#             # using a 1D Gaussian...
#             # for i in range(n_obs):
#             #     print(Lum_Gaussian(z_hat, Z[i], R[i]**2) / 10.)
#             #     weight += Lum_Gaussian(z_hat, Z[i], R[i]**2) / 10.
#             #     print(i, 'weight', weight)

#             n_obs = len(Z)
#             z_hat = np.asmatrix([X[11] for i in range(n_obs)])

#             Z = np.asmatrix(np.reshape(Z, -1))
#             #R = np.square(R)#/((tk-tkm1)/20))
#             cov = np.asmatrix(np.diag(np.reshape(R,  -1)))
            
#             for i in range(n_obs):
#                 ## if you are wanting to consider saturation, uncomment
#                 ## this section as an alternative to line 279
#                 # ## here there are two options if sensor was saturated 
#                 # ## /!\ user defined value of saturation!
                
#                 # if z_hat[0, i]>-6.0:
#                 #     ## (1) if luminosity is less than saturation, set to zero
#                 #     ##     anything else is going to return the default value 
#                 #     ##     of 1/N (above)

#                 #     lum_weight *= np.exp(-5000.)

#                 #     ## (2) other option is to calculate a skew normal distribution. 
#                 #     ## this will unfavour really high values.
#                 #     ## lum_weight *= skew_norm_pdf(z_hat[0, i],Z[ 0, i],5,-3)

#                 # else:
#                 #     # seems to be a problem with multivariate Gaus calcultation...
#                 #     #lum_weight *= multi_var_gauss(z_hat.T, Z.T, cov, n_obs) 
#                 #     lum_weight *= Gaussian(z_hat[0,i], Z[0,i], R[i]) 

#                 lum_weight *= Gaussian(z_hat[0,i], Z[0,i], R[i]) 

#         ## position observations
#         observation = obs_info[:, 0:2]
#         camera = obs_info[:, 2:5]   #LLH
#         R = obs_info[:, 5:]
#         n_obs = len(observation)

#         # print('td', T_jd)
#         # # Extracting the camera location from table and determine best fit plane
#         # NumberCams = len(camera)
#         Cameras_LLH = obs_info[:, 2:5]   #LLH
#         # Cameras_ECEF = np.zeros((3, NumberCams))
#         # Cameras_ECI = np.zeros((3, NumberCams))
#         # n_ECEF = np.zeros((3, NumberCams))
#         for cam in range(n_obs):
            
#             # Extract the camera's position            
#             # Define the camera locations

#             # cam_ECEF[:, cam:cam + 1] = tu.LLH2ECEF(Cameras_LLH[0, :])
#             # cam_ECI[:, cam:cam + 1] = tu.ECEF2ECI_pos(cam_ECEF[:, cam:cam + 1], T_jd)


#             ## Extract the camera data
#             ## CamData=gft(CamFiles[cam],names=True,delimiter=',',skip_header=1,dtype=None)
#             #astrometry_table = astrometry_tables[cam]
#             #Time_isot = astrometry_table['datetime']
#             ## Fetch the az/el data from the table
#             obs_az = observation[cam, 0]
#             obs_el =  observation[cam, 1]
#             obs_az_error =  R[cam, 0]
#             obs_el_error =  R[cam, 1]
#             z_az, z_el = Part2AltAz_ECI(X[0], X[1], X[2], Cameras_LLH[cam, :], T_jd)

#             # # Convert the line of sight angles to ENU coords
#             # obs_UV_ENU = np.vstack((np.cos(obs_el) * np.sin(obs_az),
#             #                     np.cos(obs_el) * np.cos(obs_az),
#             #                     np.sin(obs_el)))

#             # z_UV_ENU = np.vstack((np.cos(z_el) * np.sin(z_az),
#             #                     np.cos(z_el) * np.cos(z_az),
#             #                     np.sin(z_el)))
#             z_hat = np.array([z_az, z_el])
#             Z = np.array([[obs_az], [obs_el]])
#             cov = np.array([obs_az_error**2, obs_el_error**2])
#             cov = np.asmatrix(np.diag(cov))
#             pos_weight *= multi_var_gauss_angles(z_hat, Z, cov, 2)
            

#         #     # Compute the other transformation matrix
#         #     GS_ECI = tu.ECEF2ECI_pos(cam_ECEF, T_jd)
#         #     Cameras_ECI.extend([ GS_ECI ])

#         #     GS_ra = float(np.arctan2(GS_ECI[1], GS_ECI[0]))
#         #     GS_LLH_plus = Cameras_LLH[:, cam:cam+1] + np.vstack((0,0,100))
#         #     GS_ECI_plus = tu.ECEF2ECI_pos(tu.LLH2ECEF(GS_LLH_plus), T_jd)
            
#         #     E = np.array([-np.sin(GS_ra), np.cos(GS_ra), 0])
#         #     U = (GS_ECI_plus - GS_ECI).flatten() / norm(GS_ECI_plus - GS_ECI)
#         #     N = np.cross(U,E)
#         #     C_eci2enu = np.vstack((E, N, U))
#         #     C_ECI2ENU.extend([ C_eci2enu ])

#         #     # Convert the line of sight vector, UV, to ECI coordinates
#         #     UV = C_eci2enu.T.dot(UV_ENU) # ECI

#         #     # # The first six elements of the state vector are the first two
#         #     # # positions from the first camera given the second cameras normal.
#         #     # if cam == 0 and j == 0:

#         #     #     # Get second camera's position
#         #     #     GS2_ECEF = Cameras_ECEF[:, 1:2]
#         #     #     GS2_ECI = ECEF2ECI_pos(GS2_ECEF, T_jd)

#         #     #     # Find the unit vector in optimised plane
#         #     #     UV_opt = UV - np.dot(UV.T, n1) * n1

#         #     #     # Add the xyz positions to the state vector
#         #     #     d = n2.T.dot(GS2_ECI - GS_ECI) / n2.T.dot(UV_opt)
#         #     #     X0.extend( (UV_opt * d + GS_ECI)[:,0].tolist() )

#         #     #     # Check it is pointing down
#         #     #     if np.hstack((X0)).dot(V0) > 0:
#         #     #         V0 = -V0
                    
#         #     #     ra_eci = np.arctan2(V0[1], V0[0])  # arctan2 default is -pi < ra < pi
#         #     #     dec_eci = np.arcsin(V0[2] / norm(V0)) # arcsin default is -pi/2 < dec < pi/2
#         #     #     X0.extend( [ra_eci, dec_eci] )
            
#         #     # Calculate the first approx lengths, L, from the first point on
#         #     # the radient to a point closest the line of sight vector for all
#         #     # cameras and all time points.
#         #     # else:
#         #     #     # Add the lengths to the state vector
#         #     #     # Vector perpendicular to the radiant and the line of sight
#         #     #     v_perp = np.cross(V0, UV, axis=0)
                
#         #     #     # Normal to plane with UV and v_perp
#         #     #     n_UV = np.cross(v_perp, UV, axis=0) / norm(np.cross(v_perp, UV, axis=0))
#         #     #     L = np.dot(n_UV.T, (GS_ECI - np.reshape(X0[:3],(3, 1)))) / (np.dot(n_UV.T, V0))
#         #     #     X0.extend([L])

#         # # Reshape the arrays into column vectors
#         # y_obs = np.array(y_obs)
#         # Sigma2 = np.array(Sigma2)
#         # X0 = np.array(X0)
#         # Cameras_ECI = np.hstack(Cameras_ECI)
#         # C_ECI2ENU = np.hstack(C_ECI2ENU)
#         # args = [y_obs, Sigma2, Cameras_ECI, C_ECI2ENU, NumberTimePts]



#         # R = obs_info[:, 5:]
#         # n_obs = len(observation)

#         # ## covert particle ECEF to alt az from camera
#         # z_hat = np.zeros((1, n_obs*2))

#         # for i in range(n_obs):
#         #     z_hat[0, 2*i], z_hat[0, 2*i+1] = bf.Part2AltAz(X[0], X[1], X[2], camera[i, :])

#         # Z = np.reshape(observation, -1)
#         # R = np.reshape(np.square(R), -1)
#         # #R = np.reshape(R, -1)

#         # z_hat = np.asmatrix(z_hat)
#         # Z = np.asmatrix(Z)
#         # cov = np.asmatrix(np.diag(R))

#         # pos_weight *= multi_var_gauss_angles(z_hat.T, Z.T, cov, n_obs)

#         # observation = obs_info[:, 0:2]
#         # camera = obs_info[:, 2:5]
#         # R = obs_info[:, 5:]

#         # n_obs = len(observation)
#         # z_hat = np.zeros((1, n_obs*2))

#         # for i in range(n_obs):
#         #     ## covert particle ECEF to alt az from camera
#         #     az, el = bf.Part2AltAz(X[0], X[1], X[2], camera[i, :])

#         #     pos_weight *= Gaussian_angles(az, observation[i, 0], R[i, 0])
#         #     pos_weight *= Gaussian_angles(el, observation[i, 1], R[i, 1])

#     return pos_weight, lum_weight



# def Part2AltAz_ECI(x, y, z, obs_LLH, T_jd):
#     """ calculate the alt/el of a particle at a geocentric 
#         ECEF position (x, y, z) in relation to a camera with 
#         a lat,long,height of obs_LLH
#     """
    
#     ## Convert the camera's angular coords to radians
#     #obs_LLH[0] = np.deg2rad(obs_LLH[0] )
#     #obs_LLH[1] = np.deg2rad(obs_LLH[1] )

#     # Camera coordinates
#     cam_ECEF = tu.LLH2ECEF(obs_LLH)
#     cam_ECI = tu.ECEF2ECI_pos(cam_ECEF, T_jd)
#     part_ECEF = np.vstack((x, y, z))
#     part_ECI = tu.ECEF2ECI_pos(part_ECEF, T_jd)
#     lat = float(obs_LLH[0]); lon = float(obs_LLH[1])

#     z_hat_ECI = part_ECI - cam_ECI
#     z_hat_ECEF = tu.ECI2ECEF_pos(z_hat_ECI, T_jd)


#     # Compute the transformation matrix
#     trans = tu.ECEF2ENU(lon, lat)

#     # Current position in local ENU coords
#     ENU = np.dot(trans, z_hat_ECEF)

#     # Calculate the azimuth & elevation from the cameras to the data points
#     alt = np.arctan2(ENU[2], np.sqrt(ENU[0]**2 + ENU[1]**2))
#     azi = np.arctan2(ENU[0], ENU[1]) % (2 * np.pi)  # Always between 0 and 2pi
    
#     # obs_coord = SkyCoord(ra=azi * u.rad, dec=alt * u.rad)  

    
#     #         cam_ECI = tu.ECEF2ECI_pos(cam_ECEF, T_jd)

#     #         GS_ra = float(np.arctan2(cam_ECI[1], cam_ECI[0]))
#     #         GS_LLH_plus = Cameras_LLH[:, cam:cam+1] + np.vstack((0,0,100))
#     #         GS_ECI_plus = tu.ECEF2ECI_pos(tu.LLH2ECEF(GS_LLH_plus), T_jd)
#     #         E = np.array([-np.sin(GS_ra), np.cos(GS_ra), 0])
#     #         U = (GS_ECI_plus - GS_ECI).flatten() / norm(GS_ECI_plus - GS_ECI)
#     #         N = np.cross(U,E)
#     #         C_eci2enu = np.vstack((E, N, U))

#     #         # Convert the line of sight vector, UV, to ECI coordinates
#     #         UV = C_eci2enu.T.dot(UV_ENU) # ECI
#     # z_az, z_el = Part2AltAz_ECI(X[0], X[1], X[2], camera[i, :])

#     # z_UV_ENU = np.vstack((np.cos(z_el) * np.sin(z_az),
#     #                     np.cos(z_el) * np.cos(z_az),
#     #                     np.sin(z_el)))

#     #         pos_weight *= gauss_angles(z_hat.T, Z.T, cov, n_obs)
            

#     #         # Compute the other transformation matrix
#     #         GS_ECI = tu.ECEF2ECI_pos(cam_ECEF, T_jd)
#     #         Cameras_ECI.extend([ GS_ECI ])

#     #         GS_ra = float(np.arctan2(GS_ECI[1], GS_ECI[0]))
#     #         GS_LLH_plus = Cameras_LLH[:, cam:cam+1] + np.vstack((0,0,100))
#     #         GS_ECI_plus = tu.ECEF2ECI_pos(tu.LLH2ECEF(GS_LLH_plus), T_jd)
            
#     #         E = np.array([-np.sin(GS_ra), np.cos(GS_ra), 0])
#     #         U = (GS_ECI_plus - GS_ECI).flatten() / norm(GS_ECI_plus - GS_ECI)
#     #         N = np.cross(U,E)
#     #         C_eci2enu = np.vstack((E, N, U))
#     #         C_ECI2ENU.extend([ C_eci2enu ])

#     #         # Convert the line of sight vector, UV, to ECI coordinates
#     #         UV = C_eci2enu.T.dot(UV_ENU) # ECI
#     return alt, azi


# def multi_var_gauss(pred, mean, cov, n_obs):
#     """performs multivariate Gaussian PDF. using angles between vectors."""

#     det_cov = np.linalg.det(cov)
#     inv_cov = np.linalg.inv(cov)
#     #print('inv', inv_cov)
#     diff = pred - mean
#     diff = (diff + np.pi) % (np.pi*2) - np.pi       # gives 355' - 004' --> 9' (acute angle through 2pi) in degrees: (diff + 180) % 360 - 180
#     diff = np.asmatrix(abs(diff))
    
#     # multivariate equation:
#     likelihood = pow((2*np.pi), -n_obs/2) * pow(det_cov, -.5) * np.exp(-.5*diff.T*inv_cov * diff)
    
#     # if likelihood -ve, set to ~0.
#     if likelihood <= 0:
#         print('caught a -ve likelihood. setting to ~0')
#         return -500.
#     else:
#         # return log of likelihoods
#         return np.log(likelihood)
  
# def Gaussian(z_hat, Z, R):
#     """performs Gaussian PDF. 
#         Inputs:
#         z_hat - observation
#         Z - mean
#         R - variance"""

#     diff = (z_hat - Z)

#     # calculates the probability of x for 1-dim Gaussian with mean mu and var. sigma
#     likelihood = exp(- 0.5 * (np.dot(diff, diff.transpose())) / R) / sqrt(2.0 * pi * R)
    
#     # if likelihood -ve, set to ~0.
#     if likelihood <= 0:
#         print('caught a -ve likelihood. setting to ~0')
#         return -500.
#     else:
#         # return log of likelihoods
#         return np.log(likelihood)

# def rand_skew_norm(fLocation, variance, fAlpha):
#     ## literal adaption from:
#     ## http://stackoverflow.com/questions/4643285/how-to-generate-random-numbers-that-follow-skew-normal-distribution-in-matlab
#     ## original at:
#     ## http://www.ozgrid.com/forum/showthread.php?t=108175

#     sigma = fAlpha / np.sqrt(1.0 + fAlpha**2) 
#     fScale = pow(1-(2*sigma**2/3.14159),0.5)
#     afRN = np.random.randn(2)
#     u0 = afRN[0]
#     v = afRN[1]
#     u1 = sigma*u0 + np.sqrt(1.0 -sigma**2) * v 
#     if u0 >= 0:
#         return u1*fScale + fLocation 
#     return (-u1)*fScale + fLocation 

# def skew_norm_pdf(x,e=0,w=1,a=0):
#     ## a = shape (amount of skew)
#     ## w = scale
#     ## e = location
#     ## adapated from:
#     ## http://stackoverflow.com/questions/5884768/skew-normal-distribution-in-scipy
    
#     t = (x-e) / w
#     likelihood = 2.0 * w * scipy.stats.norm.pdf(t) * scipy.stats.norm.cdf(a*t)
    
#     # if likelihood -ve, set to ~0.
#     if likelihood <= 0:
#         print('caught a -ve likelihood. setting to ~0')
#         return -500
#     else:
#         # return log of likelihoods
#         return np.log(likelihood)

# def tau_cal(v, m):

#     k1 = -1.494
#     k2 = -3.488

#     if v<25372.0:
#         y = k1 - 10.307*log(v) + 9.781*log(v)**2 - 3.0414*log(v)**3 + 0.3213*log(v)**4
#         tau_vel =  np.exp(y)
#     else:
#         tau_vel =  np.exp(log(v) + k2)

#     return tau_vel



     #    observation = obs_info[:, 0:2]
     #    camera = obs_info[:, 2:5]
     #    R = obs_info[:, 5:]

     #    n_obs = len(observation)

     # ## covert particle ECEF to alt az from camera
     #    z_hat = np.zeros((1, n_obs*2))

     #    for i in range(n_obs):
            
     #        z_hat[0, 2*i], z_hat[0, 2*i+1] = bf.Part2AltAz(X[0], X[1], X[2], camera[i, :])


     #    Z = np.reshape(observation, -1)
     #    #R = np.reshape(R, -1)
     #    R = np.reshape(np.square(R), -1)

     #    #print(z_hat, '\n', Z, '\n',R, '\n', cov)
        
     #    #for i in range(len(z_hat[0])):
     #    #    print(z_hat[0, i], '\n', Z[i], '\n',R[i])
     #    #    weight *= Gaussian(z_hat[0, i], Z[i], R[i])

     #    z_hat = np.asmatrix(z_hat)
     #    Z = np.asmatrix(Z)
     #    cov = np.asmatrix(np.diag(R))

     #    #weight_mv = 1./N #X[29]
     #    pos_weight += multi_var_gauss(z_hat.T, Z.T, cov, n_obs)
     #    #print('final weight,', weight)




     #  observation = obs_info[:, 0:2]
     #    camera = obs_info[:, 2:5]
     #    R = obs_info[:, 5:]

     #    n_obs = len(observation)

     # ## covert particle ECEF to alt az from camera
     #    z_hat = np.zeros((1, n_obs*2))

     #    for i in range(n_obs):
     #        az, el = bf.Part2AltAz(X[0], X[1], X[2], camera[i, :])

     #        pos_weight += Gaussian(az, observation[i, 0], R[i, 0])
     #        pos_weight += Gaussian(el, observation[i, 1], R[i, 1])