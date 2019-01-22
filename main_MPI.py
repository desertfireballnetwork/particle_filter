#!/usr/bin/env python
"""
=============== Meteoroid Trajectory Analysis ===============
============ using a bootstrap particle filter ==============

Created on Mon Jul 16 2017
@author: Eleanor Kate Sansom

Uses particle filter methodology outlined in Arulampalam et al.
(2002) -- A tutorial on particle filters for online nonlinear/
non-Gaussian Bayesian tracking (doi: 10.1109/78.978374).


## Requirements:
- From Triangulation folder: trajectory_utilities.py, 
                             dfn_utils.py
- From Brightflight folder: bf_functions_3d.py, 
                            ballistic_params.py
- From Brightflight/particle_filters folder: nrlmsise_00.py, 
                                             nrlmsise_00_data.py, 
                                             nrlmsise_00_header.py
## Running the program:
run in commandline as 
$ mpirun -n <#nodes> python main_MPI.py <userinputs>

There are three run options depending on combination of data 
available. 
(1) 1D particle filter:            1D analysis on a single, 
                                   pre-triangulated trajectory 
                                   file with X_geo, Y_geo and 
                                   Z_geo information.

(2) 3D particle filter, cartesian: 3D analysis on single or 
                                   multiple pre-triangulated 
                                   files with X_geo, Y_geo and 
                                   Z_geo information.

(3) 3D particle filter, rays:      3D analysis on calibrated 
                                   astrometric observations 
                                   (altitude and azimuth data 
                                   required) 

Inputs: required:
        -i --dimension: select from (1) (2) or (3) described 
             above.
        -d --inputdirectory: input directory of folder 
             containing files with extension .ecsv
        -p --numparts: number of particles to run, testing run 
             100; quick run try 1000; good run try 10,000; 
             magnus 100,000+
        
        optional:
        -m --mass_option: Specify how you would like entry masses 
             to be initiated. (1) calculated using ballistic 
             coefficient in metadata; (2) using random logarithmic 
             distribution; (3) random uniform distribution. 
             Default is 3
        -l --luminosity_weighting:  weighting for luminosity
        -M0 --m0_max: if using -m 2 or 3, specify a maximum 
              initial mass. Default is 2000",default=2000.)
        -c --comment: add a version name to appear in saved 
             file titles. If unspecified, _testing_ is used.
        -k --save_kml: save observation rays as kmls
        -o --old: previous .fits file
        
        to be tested:
        -f --fragment: If this is a fragment run, it will 
             increase covariance values at the specified 
             times given by -t option
        -t --fragment_start: If this is a fragment run, give  
             starting time (seconds in UTC, not event relative)
        -r --reverse time to start particle filter at the end 
             of the brightflight trajectory
        

Output: HDU table fits file. 
        HDU table[0] is the primary and holds information on 
        all other indices and does not contain data
        access this info using table.info(). 
        See http://docs.astropy.org/en/stable/io/fits/ 
        for more info. 

        Table at each time step is saved as a new index in HDU 
        table. 
        access last timestep table using table[-1].data

        Each table is in the format:
        #-------------------------------------------------------
        column index KEY:
         0 : 'X_geo'       - X posistion in ECEF (m)
         1 : 'Y_geo'       - Y posistion in ECEF (m)
         2 : 'Z_geo'       - Z posistion in ECEF (m) 
         3 : 'X_geo_DT'    - X velocity (dx/dt) in ECEF (m/s) 
         4 : 'Y_geo_DT'    - Y velocity (dy/dt) in ECEF (m/s) 
         5 : 'Z_geo_DT'    - Z velocity (dz/dt) in ECEF (m/s) 
         36: 'X_eci'       - X posistion in ECI (m)
         37: 'Y_eci'       - Y posistion in ECI (m)
         38: 'Z_eci'       - Z posistion in ECI (m)
         39: 'X_eci_DT'    - X velocity (dx/dt) in ECI (m/s)
         40: 'Y_eci_DT'    - Y velocity (dy/dt) in ECI (m/s)
         41: 'Z_eci_DT'    - Z velocity (dz/dt) in ECI (m/s)
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
         14: 'Q_y'         - variance of process noise for Y position
         15: 'Q_z'         - variance of process noise for Z position
         16: 'Q_v_x'       - variance of process noise for X velocity
         17: 'Q_v_y'       - variance of process noise for Y velocity
         18: 'Q_v_z'       - variance of process noise for Z velocity
         19: 'Q_m'         - variance of process noise for mass
         20: 'Q_cd'        - variance of process noise for drag coefficient (unused)
         21: 'Q_cl'        - variance of process noise for lift coefficient (unsued)
         22: 'Q_k'         - variance of process noise for kappa
         23: 'Q_s'         - variance of process noise for sigma
         24: 'Q_tau'       - variance of process noise for luminous efficiency
         25: 'brightness'  - luminous intensiy
         26: 'rho'         - initial density (kg/m3)
         27: 'parent_index'- index of parent particle (t-1)
         28: 'orig_index'  - index of original particle assigned in dim.Initialise() (t0)
         29: 'weight'      - combined position and luminous weighting (assigned in main)
         30: 'D_DT'        - magnitude of velocity vector: norm(vel_x, vel_y, vel_z)
         31: 'latitude'    - latitude (degrees)
         32: 'longitude'   - longitude (degrees)
         33: 'height'      - height (m)
         34: 'lum_weight'  - luminous weighting
         35: 'pos_weight'  - position weighting

        additional columns include time (relative to event t0)
        and datetime
        

    Still TODO:
    - Intensity calculation - there are 3 ways of doing it...
    - 1D pf errors are being set within the weighting function 
      rather than being passed from inputs

"""

# import modules used by all dims

# general
from math import *
import copy, random
import sys, os, argparse, glob
import contextlib

# science
import numpy as np
import scipy 
import scipy.integrate

# Astropy
from astropy.table import Table, Column, join, hstack
import astropy.units as u
from astropy.io import fits
from astropy.time import Time, TimeDelta

# own
import bf_functions_3d as bf
from nrlmsise_00_header import *
from nrlmsise_00 import *
import dfn_utils 
import bf_functions_3d as bf
import trajectory_utilities as tu

#import multiprocessing
from mpi4py import MPI



def ParticleFilterParams(dim):
    """ takes in dim value so that 1D uncertainties can be adjusted

        returns particle filter function parameters. 
        Q_c:      process noise variances as a row vector
        Q_c_frag: process noise variances as a row vector if there is a fragmentation event 
                  (higher for mass and vels)
        P:        initial state varances (position and velocity only)
        range_params: other dynamic equation parameter ranges for initiation of 
                      particles

    """

    ## Particle filter parameters

    # Q_c will be the time continuous covariance matrix. 
    #This should be the errors in the model.
    # in the form [x_cov, y_cov, z_cov, 
    #              vel_x_cov, vel_y_co, vel_z_cov, 
    #              mass_cov,                      
    #              sigma_cov, shape_cov, brightness_cov, tau_cov]
    
    if dim ==1:
        Q_c = [150., 0., 0., 
               300., 0., 0., 
               0.01, 0, 0,
               1e-4, 1e-10, 0., 0.0001]

    else:
        Q_c = [50., 50., 50., 
               150., 150., 150., 
               0.01, 0, 0,
               1e-4, 1e-10, 0., 0.0001]


    print('Qc values used:', Q_c)

    Q_c = np.asarray([i**2 for i in Q_c])

    
    # Q_c_frag is used at reinitialisation if the fragmentation option is used
    
    Q_c_frag = [0., 0., 0., 
                0.02, 0.02, 0.02, 
                0.5,  0, 0,
                2e-3, 5e-9, 0., 0.]

    Q_c_frag = [i**2 for i in Q_c_frag]

    ## P: starting uncertainty to initialise gaussian spread of particals. 
    ## P2: starting uncertainty at reinitialisation if the fragmentation option is used
    ## in the form [x_cov, y_cov, z_cov, % of vel_x_cov, % of vel_y_co, % of  vel_z_cov]
    if dim ==1:
        P = [150., 0., 0., 750., 0., 0.]
    else:
        P = [150., 150., 150., 750., 750., 750.]

    ## Initialise state ranges


    ## shape parameter close to a rounded brick (1.8) (A for a sphere =1.21)
    A_min = 1.21
    A_max = 3.0    

    ## luminosity coefficient
    tau_min = 0.0001
    tau_max = 0.1

    ## lists of typical meteorite densities for different types. [chond, achond, stony-iron, iron, cometary]
    pm_mean = [3000, 3100, 4500, 7500, 850]
    pm_std = [420, 133, 133,  167, 117 ]

    ## to choose density values according to a distribution of meteorite percentages:
    particle_choices = []

    # this is created using commented out lines below; uncomment if percentages need changing.
    random_meteor_type = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 3, 3, 3, 3, 3, 4, 4]

    #random_meteor_type = []
    #for i in range(80):                 # 80 % Chondrites
    #   random_meteor_type.append(0)
    #for i in range(11):                 # 11 % Achondrites
    #   random_meteor_type.append(1)
    #for i in range(2):
    #   random_meteor_type.append(2)   # 2 % Stony-Iron
    #for i in range(5):
    #   random_meteor_type.append(3)   # 5 % iron
    #for i in range(2):
    #   random_meteor_type.append(4)   # 2 % cometary

    ## ablation coefficeint 
    #sigma_min = 0.001*1e-6
    #sigma_max = 0.5*1e-6


    #range_params = [m0_max, A_mean, A_std, pm_mean, pm_std, random_meteor_type, cd_mean, cd_std, sigma_min, sigma_max, K_min, K_max, tau_min, tau_max]
    range_params = [A_min, A_max, pm_mean, pm_std, random_meteor_type, tau_min, tau_max]

    return Q_c, Q_c_frag, P,  range_params

if __name__ == '__main__':

    # Define MPI message tags
    READY, START, DONE, EXIT = 0, 1, 2, 3
    # Initializations and preliminaries
    comm = MPI.COMM_WORLD   # get MPI communicator object
    size = comm.size        # total number of processes
    rank = comm.Get_rank()        # rank of this process
    status = MPI.Status()   # get MPI status object

    if rank ==0:
        parser = argparse.ArgumentParser(description='Run particle filter on raw camera files.')
        #inputgroup = parser.add_mutually_exclusive_group(required=True)
        parser.add_argument("-d","--inputdirectory",type=str,
                help="input directory of folder containing triangulated files with extension .ecsv", required=True)
        parser.add_argument("-p","--numparts",type=int,
                help="number of particles to run. Must be an integer.", required=True)
        parser.add_argument("-i","--dimension",type=int,
                help="would you like to run \n(1) 1D analysis on a single, pre-triangulated trajectory file, \n(2) 3D analysis on multiple pre-triangulated files, \n(3) 3D analysis on calibrated raw observations in ECI, \n(4) 3D analysis on pointwise data",required=True)
        parser.add_argument("-c","--comment",type=str,
                help="add a version name to appear in saved file titles. If unspecified, _testing_ is used.",default='testing')
        parser.add_argument("-f","--fragment",action="store_true",
                help="If this is a fragment run",default=False)
        parser.add_argument("-t","--fragment_start",type=float, nargs='+',
                help="If this is a fragment run, give starting time (seconds in UTC, not event relative)",default=[])  
        parser.add_argument("-k","--save_kml",action="store_true",
                help="save observation kmls?",default=False)
        parser.add_argument("-m","--mass_option",type=int,
                help="initial masses to be calculated using \n(1) ballistic coefficient;\n(2) using random logarithmic distribution;\n(3) random uniform distribution. Default is 3",default=3)
        parser.add_argument("-M0","--m0_max",type=float,
                help="if using -m 2 or 3, specify a maximum initial mass. Default is 2000",default=2000.)
        parser.add_argument("-o","--old",type=str,
                help="previous .fits file",default='')
        parser.add_argument("-l","--luminosity_weighting",type=float,
                help="weighting for luminosity",default=0.)
        parser.add_argument("-r","--time_reverse",action="store_true",
                help="would you like to run in reverse?",default=False)
        #parser.add_argument("-a","--alpha",type=int,

        args = parser.parse_args()   

        dim = int(args.dimension)
        #alpha_cam = int(args.alpha)
        mass_opt = int(args.mass_option)
        m0_max = float(args.m0_max)
        prev_file = args.old
        reverse = args.time_reverse

        if dim==1:
            import geo_1d as df
        elif dim==2 or dim==4:
            import geo_3d as df
        elif dim==3:
            import full_3d_ECI as df

        # number of particles
        num_parts = args.numparts

        # how is initial mass going to be initiated?
        mass_opt = int(args.mass_option)
        
        # what is max initial mass if using mass opt 2 or 3
        m0_max = float(args.m0_max)

        # are we reloading an old run that terminated early?
        prev_file = args.old

        # not sure fragment runs work yet.
        if args.fragment and not args.fragment_start:
            print("if you're running fragments, you need a fragmentation time also. see -help")
            sys.exit() 

        # fragment run?
        fragmentation = args.fragment

        # times of defined fragmentation times
        t_frag = args.fragment_start

        # save observation rays as kml?
        kmls = args.save_kml
        if dim==4:
            kmls = False
        

        # weighting of luminosity observations
        lum_weighting_coef = args.luminosity_weighting

        # output name defined by inputs
        lum_str = '%.2f' % lum_weighting_coef
        lum_str = lum_str.replace('.', 'p')
        version = 'testing_' + str(dim) +'d_' + lum_str if args.comment == '' else '_' + args.comment + '_' + str(dim) + 'd_' + lum_str
        
        ## check if the input directory given exists and pull all ecsv files, and extract data
        if (os.path.isdir(args.inputdirectory)):
            working_dir = args.inputdirectory
            
            # list all altaz files in this directory
            all_ecsv_files = glob.glob(os.path.join(working_dir,"*.ecsv"))
            ecsv = [f for f in all_ecsv_files if '_CUT' not in f]
            filenames = []

            # check if ecsv files have times and use only those that do
            for f in ecsv:
                if "notime" not in f and "LeastSquare" not in f:
                    filenames.append(str(f))
                else:
                    print(f, 'does not contain timing data and will not be used')

            n_obs = len(filenames)

            data, t0, T0, eci_bool = bf.Fireball_Data(filenames,  reverse)

            yday = T0.yday.split(':')
            y = float(yday[0])
            d = float(yday[1])
            s = float(yday[2]) * 60 * 60 + float(yday[3]) * 60 + float(yday[4])
            t_stack = np.vstack((y, d, s))


            ## get data depending on particle filter flavour
            # 1D filter:
            P_test=0
            if dim == 1:
                x0 = data['dist_from_start'][0]

                # currently first data point has no velocity, so use first velocity value as close approximation.
                v0 = 0
                i = 0
                while v0<1:
                    if data['D_DT'][i] >0:
                        v0 = data['D_DT'][i]
                    else:
                        i+=1

                     
                out_name = data['datetime'][0].split('T')[0].replace('-','') + '_1D'

            else:
                
                if dim==2 or dim==4:   # 3D cartesian 
                    ## t0 is start of filter, T0 is start of fireball

                    if dim==2:
                        out_name = data['datetime'][0].split('T')[0].replace('-','') + '_3Dtriangulated'

                        data.sort('time')    

                        if 'X_eci' in data.colnames:
                            print('Running in ECI')
                            [x0, v0, date_info] = [[data['X_eci'][0], data['Y_eci'][0], data['Z_eci'][0]],
                                                   [(data['X_eci'][3] - data['X_eci'][0])/(data['time'][3] - data['time'][0]),
                                                    (data['Y_eci'][3] - data['Y_eci'][0])/(data['time'][3] - data['time'][0]),
                                                    (data['Z_eci'][3] - data['Z_eci'][0])/(data['time'][3] - data['time'][0])],
                                                   t_stack]
                        else:
                            print('Running in ECEF')
                            [x0, v0, date_info] = [[data['X_geo'][0], data['Y_geo'][0], data['Z_geo'][0]],
                                                   [(data['X_geo'][3] - data['X_geo'][0])/(data['time'][3] - data['time'][0]),
                                        


                                                    (data['Y_geo'][3] - data['Y_geo'][0])/(data['time'][3] - data['time'][0]),
                                                    (data['Z_geo'][3] - data['Z_geo'][0])/(data['time'][3] - data['time'][0])],
                                                   t_stack]
                        
                        

                elif dim==3 :  # 3D rays
                    if n_obs>1:
                        ## t0 is start of filter, T0 is start of fireball
                        # data, t0, T0, eci_bool = bf.Geo_Fireball_Data(filenames, pse, reverse)
                        out_name = data['datetime'][0].split('T')[0].replace('-','') + '_3Draw'
                        
                        data.sort('time')    
                        if reverse:
                            data.reverse()
                        [x0, v0, x0_err, v0_err, date_info] = bf.RoughTriangulation(data, t0, reverse, eci_bool)


                    else: 
                        print('you need at least two camera files for your choice of -i option.')
                        exit(1)
                else:
                    print('invalid dimension key -i')
                    exit(1)

                
            ## define list of unique timesteps and their locations in data table
            data.sort('time')                    

            if reverse: 
                date_info = np.append(t_stack, Time(data['datetime'][0], format='isot', scale='utc'))
                data.reverse()
            else: 
                date_info = np.append(t_stack, T0) 

            data_times = data['time']

            data_t = np.array([[float(data_times[0])], [str(data['datetime'][0])], [int(0)]])
            for i in range(1, len(data_times)):
                if data_times[i] != float(data_t[0, -1]):
                        data_t = np.hstack((data_t, [[float(data_times[i])], [str(data['datetime'][i])], [int(i)]]))

        else:
            print('not a valid input directory')
            exit(1)

       
        ## distinguish between fragmentation filter runs and simple run, create output directory.
        if fragmentation:
            print('this will be a fragmentation run')
            # create output directory
            out_name = out_name + "_frag_"

            if not os.path.exists(os.path.join(working_dir ,"outputs", out_name)):
                os.makedirs(os.path.join(working_dir ,"outputs", out_name))

            ## save times given for fragmentation events 
            #  and adds an extra index at the end which is beyond last iteration 
            #  so line 588 if statement will work after frag event has happened.
            for i in range(len(t_frag)+1):
                if i<len(t_frag):
                    t_frag[i] = (np.abs(np.array(float(data_t[0, :]))-t_frag[i])).argmin()
                else:
                    t_frag = np.hstack([t_frag, len(data_t[0, :])+1])
            print(t_frag)
        else:
            ## create output directory
            if not os.path.exists(os.path.join(working_dir,"outputs",out_name)):
                os.makedirs(os.path.join(working_dir,"outputs",out_name))

            ## no fragmentation times are given
            t_frag = [len(data_t[0])+1]

        ## save raw data for visualisation after
        if eci_bool:
            eci_name = 'eci'
        else:
            eci_name = 'ecef'

        name= os.path.join(working_dir ,"outputs", out_name, out_name +version +'_' + eci_name + '_mass'+ str(mass_opt) + '_inputdata.csv')
        data.write(name, format='ascii.csv', delimiter=',')
        
        ## save kmls of observation vectors from all cameras
        if kmls:
            bf.RawData2KML(filenames, pse)
            bf.RawData2ECEF(filenames, pse)
        
        ## defining number of particles to run
        n = int(num_parts/(comm.size)) +1              # number of particles for each worker to use
        N = n * comm.size                            # total number of particles (n*24)

        ## info to share with ranks:
        T = len(data_t[0])      # number of timesteps
        p_all = np.empty([N, 42])
        
        ## empy space for effectiveness array
        n_eff_all = np.zeros(T)  # empy space for effectiveness array
        f = 0                     # index of fragmentation event (incremented in frag section at end of function) 
        l_weight = False        # if low weright, use fragmentation scattering

        [Q_c, Q_c_frag, P,  range_params] = ParticleFilterParams(dim)
        ####
        # if P_test>0:
        #     P[3] = P_test
        ## grouping everything to send
        alpha = data['alpha'][0]
        init_info = [version, T, n, N, data_t, data, x0, v0, out_name, f, t_frag, dim, l_weight, alpha, mass_opt, m0_max, reverse, date_info, eci_bool, eci_name]
        # print(init_info)
        ## send it all ranks
        for i in range(1, size):
            comm.send(init_info, dest = i)

        ## TIMER
        #t_1 = timeit.default_timer()-t_start
        #print('time to initialise code', t_1)

    else:
        [version, T, n, N, data_t, data, x0, v0, out_name, f, t_frag, dim, l_weight, alpha, mass_opt, m0_max, reverse, date_info, eci_bool, eci_name] = comm.recv(source=0)
        if dim == 1:
            import geo_1d as df
        elif dim == 2 or dim == 4:
            import geo_3d as df
        elif dim == 3:
            import full_3d_ECI as df
            
        [Q_c, Q_c_frag, P,  range_params] = ParticleFilterParams()


        p_all = None


    #########################################################
    ## all ranks get an empty working array

    p_working = np.empty([n, 42])#, dtype='O')  
  
    comm.Barrier()
    #---------------------------------------------------------
    # SIS Particle filter
    #---------------------------------------------------------
    ################# Step 1 - Initialisaton #################
    ## master scatters initial empty array to all ranks to be filled for t0

    comm.Scatterv(p_all, p_working, root=0)

    for i in range(n):
        p_working[i, :] = df.Initialise(x0, v0, rank*n+i, rank*n+i, N, P, range_params, alpha, date_info, mass_opt, m0_max, data['gamma'][0], eci_bool)

    comm.Gatherv( p_working, p_all, root=0)
    
    comm.Barrier()

    ##############  Master saves initial timestep  ###########
    if rank ==0:
        ## TIMER
        #t_2 = timeit.default_timer()-t_start - t_1
        #print('time to initialise particles', t_2)

        if prev_file != '':       
            # if -o option used, this overwrites table just initialised with previous file given
            # p_all = np.empty([N, 42])#, dtype='O')  
          
            name  = prev_file
            results_list = fits.open(prev_file)

            name_end = prev_file.replace(".fits", "_end.fits")
            results_prev_open = fits.open(name_end)

            results_prev = results_prev_open[1].data      #results_prev = results_list[-1].data

            temp_time = [float(x) for x in data_t[0]]
            t0 =    (np.abs(temp_time-results_prev['time'][0])).argmin() +1

            results_prev = Table(results_prev)
            results_prev.remove_columns(['datetime', 'time'])

            
            p_all = np.vstack([ results_prev['X_geo'].data,      # - 0 
                                results_prev['Y_geo'].data,      # - 1 
                                results_prev['Z_geo'].data,      # - 2 
                                results_prev['X_geo_DT'].data,   # - 3 
                                results_prev['Y_geo_DT'].data,   # - 4 
                                results_prev['Z_geo_DT'].data,   # - 5 
                                results_prev['mass'].data,       # - 6 
                                results_prev['cd'].data,         # - 7 
                                results_prev['A'].data,          # - 8 
                                results_prev['kappa'].data,      # - 9 
                                results_prev['sigma'].data,      # - 10
                                results_prev['mag_v'].data,      # - 11
                                results_prev['tau'].data,        # - 12
                                results_prev['Q_x'].data,        # - 13
                                results_prev['Q_y'].data,        # - 14
                                results_prev['Q_z'].data,        # - 15
                                results_prev['Q_v_x'].data,      # - 16
                                results_prev['Q_v_y'].data,      # - 17
                                results_prev['Q_v_z'].data,      # - 18
                                results_prev['Q_m'].data,        # - 19
                                results_prev['Q_cd'].data,       # - 20
                                results_prev['Q_cl'].data,       # - 21
                                results_prev['Q_k'].data,        # - 22
                                results_prev['Q_s'].data,        # - 23
                                results_prev['Q_tau'].data,      # - 24
                                results_prev['brightness'].data, # - 25
                                results_prev['rho'].data,        # - 26
                                results_prev['parent_index'].data,# - 27
                                results_prev['orig_index'].data, # - 28
                                results_prev['weight'].data,     # - 29
                                results_prev['D_DT'].data,       # - 30
                                np.deg2rad(results_prev['latitude']).data,   # - 31
                                np.deg2rad(results_prev['longitude']).data,  # - 32
                                results_prev['height'].data,     # - 33
                                results_prev['lum_weight'].data, # - 34
                                results_prev['pos_weight'].data, # - 35
                                results_prev['X_eci'].data,      # - 36
                                results_prev['Y_eci'].data,      # - 37
                                results_prev['Z_eci'].data,      # - 38
                                results_prev['X_eci_DT'].data,   # - 39
                                results_prev['Y_eci_DT'].data,   # - 40
                                results_prev['Z_eci_DT'].data])  # - 41
            p_all = copy.deepcopy(np.asarray(p_all).T)

            # if number of particles no longer matches the number of cores to run, add blank lines
            if n != len(p_all[0])/comm.size:
                p_all = np.vstack([p_all, np.zeros([abs(n * comm.size - len(p_all)), len(p_all[0])])])

            
            
        else:
            initialise = np.hstack((p_all, 
                                    np.ones((N,1))* float(data_t[0,0]), 
                                    np.array([data_t[1,0] for i in range(N)]).reshape(-1, 1)))

            ## initialise output table
            results = fits.PrimaryHDU()
            name= os.path.join(working_dir ,"outputs", out_name, out_name + version + '_' + eci_name + '_mass'+ str(mass_opt) + "_outputs.fits")
            
            results.writeto(name, clobber=True)

            ## create first HDU table and save 

            results = fits.BinTableHDU.from_columns([fits.Column(name='time', format='D', array=initialise[:, 42]),
                                           fits.Column(name='datetime', format='25A', array=initialise[:, 43]),
                                           fits.Column(name='X_geo', format='D', array=initialise[:, 0]),
                                           fits.Column(name='Y_geo', format='D', array=initialise[:, 1]),
                                           fits.Column(name='Z_geo', format='D', array=initialise[:, 2]),
                                           fits.Column(name='X_geo_DT', format='D', array=initialise[:, 3]),
                                           fits.Column(name='Y_geo_DT', format='D', array=initialise[:, 4]),
                                           fits.Column(name='Z_geo_DT', format='D', array=initialise[:, 5]),
                                           fits.Column(name='X_eci', format='D', array=initialise[:, 36]),
                                           fits.Column(name='Y_eci', format='D', array=initialise[:, 37]),
                                           fits.Column(name='Z_eci', format='D', array=initialise[:, 38]),
                                           fits.Column(name='X_eci_DT', format='D', array=initialise[:, 39]),
                                           fits.Column(name='Y_eci_DT', format='D', array=initialise[:, 40]),
                                           fits.Column(name='Z_eci_DT', format='D', array=initialise[:, 41]),
                                           fits.Column(name='mass', format='D', array=initialise[:, 6]),
                                           fits.Column(name='cd', format='D', array=initialise[:, 7]),
                                           fits.Column(name='A', format='D', array=initialise[:, 8]),
                                           fits.Column(name='kappa', format='D', array=initialise[:, 9]),
                                           fits.Column(name='sigma', format='D', array=initialise[:, 10]),
                                           fits.Column(name='mag_v', format='D', array=initialise[:, 11]),
                                           fits.Column(name='tau', format='D', array=initialise[:, 12]),
                                           fits.Column(name='Q_x', format='D', array=initialise[:, 13]),
                                           fits.Column(name='Q_y', format='D', array=initialise[:, 14]),
                                           fits.Column(name='Q_z', format='D', array=initialise[:, 15]),
                                           fits.Column(name='Q_v_x', format='D', array=initialise[:, 16]),
                                           fits.Column(name='Q_v_y', format='D', array=initialise[:, 17]),
                                           fits.Column(name='Q_v_z', format='D', array=initialise[:, 18]),
                                           fits.Column(name='Q_m', format='D', array=initialise[:, 19]),
                                           fits.Column(name='Q_cd', format='D', array=initialise[:, 20]),
                                           fits.Column(name='Q_cl', format='D', array=initialise[:, 21]),
                                           fits.Column(name='Q_k', format='D', array=initialise[:, 22]),
                                           fits.Column(name='Q_s', format='D', array=initialise[:, 23]),
                                           fits.Column(name='Q_tau', format='D', array=initialise[:, 24]),
                                           fits.Column(name='brightness', format='D', array=initialise[:, 25]),
                                           fits.Column(name='rho', format='D', array=initialise[:, 26]),
                                           fits.Column(name='parent_index', format='D', array=initialise[:, 27]),
                                           fits.Column(name='orig_index', format='D', array=initialise[:, 28]),
                                           fits.Column(name='weight', format='D', array=initialise[:, 29]),
                                           fits.Column(name='D_DT', format='D', array=initialise[:, 30]),
                                           fits.Column(name='latitude', format='D', array=[np.rad2deg(float(x)) for x in initialise[:, 31]]),
                                           fits.Column(name='longitude', format='D', array=[np.rad2deg(float(x)) for x in initialise[:, 32]]),
                                           fits.Column(name='height', format='D', array=initialise[:, 33]),
                                           fits.Column(name='lum_weight', format='D', array=initialise[:, 34]),
                                           fits.Column(name='pos_weight', format='D', array=initialise[:, 35])])

            results_list = fits.open(name, mode= 'append')
            results_list.append(results)
            results_list[-1].name='time_0'
            results_list.writeto(name, clobber=True)
            print('data saved to ', name)

            # as no previous file was given, the first prediction step will be index = 1 in the time steps. 
            t0 = 1

        ## TIMER
        #t_3 = timeit.default_timer()-t_start - t_2
        #print('time to initialise table', t_3)

        for i in range(1, size):
            comm.send(t0, dest = i)

    else:
        t0 = comm.recv(source=0)
        
    comm.Barrier()

    #################  ALL  ####################################
    ## performing iterative filter
    for t in range(t0, T):

        ## everyone gets time 
        tk = float(data_t[0, t]) 
        tkm1 = float(data_t[0, t-1])
        t_end = False

        ## does time = time of user defined fragmetation event
        if t == t_frag[f]:
            frag = True
            f +=1
        ## if low weighting, use fragmentation covariance to scatter particles more. 
        elif l_weight:
            frag = True

        else:
            frag = False
        ## if this is the final timestep, prediction step allows mass to go to 0.
        if t == T-1:
            t_end = True

    ############  Master gets observation data  ################
        if rank ==0:
            print('iteration is: ', t, 'of', T-1, 'at time:', tk)

            # find the indices of the data in the data table that correspond to the current time
            obs_index_st = int(data_t[2, t])
            obs_index_ed = int(data_t[2, int(t+1)]) if len(data_t[0])> t+1 else int(len(data))
            
            # determine if there are luminosity values available
            lum_info = []
            for i in range(obs_index_st, obs_index_ed):
                #print(data['magnitude'][i])
                if data['magnitude'][i] <50: # if 'luminosity' in data.colnames:
                    lum_info.append([data['magnitude'][i]])
                    lum_info.append([data['mag_error'][i]])

            if dim == 1:  # 1D filter
                obs_info = np.zeros((obs_index_ed - obs_index_st, 2))
                for i in range(0, obs_index_ed-obs_index_st):
                    obs_info[i,:]  = [data['dist_from_start'][i+obs_index_st], data['cross_track_error'][i+obs_index_st]]
                fireball_info= [data['Lat_rad'][obs_index_st], data['Lon_rad'][obs_index_st], data['height'][obs_index_st],  date_info[0], date_info[1], date_info[2]+tk, data['g_sin_gamma'][obs_index_st]] 
                
            elif dim == 2 or dim == 4:  # 3D cardesian
                obs_info = np.zeros((obs_index_ed - obs_index_st, 6))
                if eci_bool:
                    for i in range(0, obs_index_ed-obs_index_st):
                        obs_info[i,:] = [data['X_eci'][i+obs_index_st], data['Y_eci'][i+obs_index_st], data['Z_eci'][i+obs_index_st], data['R_X_eci'][i+obs_index_st], data['R_Y_eci'][i+obs_index_st], data['R_Z_eci'][i+obs_index_st]]
                else:
                  for i in range(0, obs_index_ed-obs_index_st):
                        obs_info[i,:] = [data['X_geo'][i+obs_index_st], data['Y_geo'][i+obs_index_st], data['Z_geo'][i+obs_index_st], data['R_X_geo'][i+obs_index_st], data['R_Y_geo'][i+obs_index_st], data['R_Z_geo'][i+obs_index_st]]
                  
                fireball_info= [0, 0, 0, date_info[0], date_info[1], date_info[2]+tk, date_info[3], tk] 

            elif dim == 3:   # 3D rays
                obs_info = np.zeros((obs_index_ed - obs_index_st, 7))
                for i in range(0, obs_index_ed-obs_index_st):
                    ##use table errors
                    obs_info[i,:] = [data['azimuth'][i+obs_index_st], data['altitude'][i+obs_index_st], data['obs_lat'][i+obs_index_st], data['obs_lon'][i+obs_index_st], data['obs_hei'][i+obs_index_st], data['R_azi'][i+obs_index_st], data['R_alt'][i+obs_index_st]]
                    ## use double table errors
                    # obs_info[i,:] = [data['azimuth'][i+obs_index_st], data['altitude'][i+obs_index_st], data['obs_lat'][i+obs_index_st], data['obs_lon'][i+obs_index_st], data['obs_hei'][i+obs_index_st], data['R_azi'][i+obs_index_st]*2, data['R_alt'][i+obs_index_st]*2]
                    ## use 0.1 degrees
                    # obs_info[i,:] = [data['azimuth'][i+obs_index_st], data['altitude'][i+obs_index_st], data['obs_lat'][i+obs_index_st], data['obs_lon'][i+obs_index_st], data['obs_hei'][i+obs_index_st], data['R_UV'][i+obs_index_st], data['R_UV'][i+obs_index_st]]
                
                fireball_info= [0, 0, 0, date_info[0], date_info[1], date_info[2]+tk, date_info[3], tk] 


            for i in range(1, size):
                comm.send([obs_info, lum_info, frag, fireball_info], dest = i)
            
            ## TIMER
            #t_pfstart = timeit.default_timer()

        else:
            [obs_info, lum_info, frag, fireball_info] = comm.recv(source=0)

    ############# Step 2 - Predict and update ##################
        ## master sends particles to ranks to perform 'forward step' which includes
        ## non-linear integration of state, model covariance and then calculates particle 
        ## likelihood. These are sent back to master.
        comm.Barrier()

        comm.Scatterv(p_all, p_working, root=0)

        ## each rank loops though their array of objects performing 'forward step' which includes
        # non-linear integration of state, model covariance and then calculates particle likelihood
        if frag: 
            for i in range(n):
                p_working[i, :] = df.particle_propagation(p_working[i], 2/3., tkm1, tk, fireball_info, obs_info, lum_info, rank*n+i, N, frag, t_end, Q_c_frag, m0_max, reverse, eci_bool)
        else:
            for i in range(n):

                p_working[i, :] = df.particle_propagation(p_working[i], 2/3., tkm1, tk, fireball_info, obs_info, lum_info, rank*n+i, N, frag, t_end, Q_c, m0_max, reverse, eci_bool)

        comm.Gatherv( p_working, p_all, root=0)

    ##########  Master calculates weights and resamples ########
        if rank ==0:
            print('master collected all ')
            ## TIMER
            #t_4 = timeit.default_timer()-t_pfstart
            #print('time to integrate', t_4)

            ## if you want to turn resampling on/off... do it here
            if t_end:
                resamp = False
            else:
                resamp = True

            #####################
            # resampling for log weights calculated in particle_propagation:
            if resamp:
                w = np.empty([8, N])

                ## 'w' - is an array for the weight calculations. 
                ## Row indices are: [pos_weight, 
                ##                   lum_weight
                ##                   normalised pos_weight,
                ##                   normalised lum_weight,
                ##                   combined normalised weight,
                ##                   exp of combined normalised weight,
                ##                   cumulative weight,
                ##                   col in p_all array (depreciated)]

            
                for i in range(N):

                    ## first set any NAN weightings to approx = 0
                    if np.isnan(p_all[i, 35]):
                        p_all[i, :] = p_all[i, :] * 0.
                        p_all[i, 35] = 0.
                    elif np.isnan(p_all[i, 34]):
                        p_all[i, :] = p_all[i, :] * 0.
                        p_all[i, 34] = 0.

                    ## fill in 'w' with position and luminous weightings and particle index

                    w[:, i] = np.array([p_all[i, 35], p_all[i, 34],0., 0., 0., 0., 0., i]).T
            
                
           
                ## calculate sum of weights
                weights_sum_p = np.nansum(w[0, :])
                weights_sum_l = np.nansum(w[1, :])

                ## TODO.. currently permanently set this to false. 
                l_weight = False


                w[2, :] = w[0, :] / weights_sum_p  # fill in normalised sum 
                w[3, :] = w[1, :] / weights_sum_l  # fill in normalised sum      
                #weights_sqr_sum = sum(w[0, :]**2)
                # print(mx_p, mx_l, weights_sum_p, weights_sum_l)
                # print(w)


                # l_weight = False

                w[4, :] = w[2, :]

                # print(mx, weights_sum)
                
                weights_sum = np.nansum(w[4, :])
                w[4, :] = w[4, :] / weights_sum

                p_all[:, 35] = w[2, :]
                p_all[:, 34] = w[3, :]
                p_all[:, 29] = w[4, :]            

                # w[5, :] = [exp(i) for i in w[4,:]]  # take exp of normalised sum

                w[6, :] = np.cumsum(w[4, :])   # fill in cumulative sum

                ## calculate particle effectiveness for degeneracy
                n_eff = 1/np.sum(w[4, :]**2)
                #n_eff_all[t] = n_eff
                print('sum of weights: ', weights_sum)
                print('effectiveness: ', n_eff/ N * 100, '%')
                # print(w)

                ## resampling
                # print(w)
                draw = np.random.uniform(0, 1 , N)
                # draw = np.cumsum([np.ones(N) * 1/N ])- 1./(2*N)
                index = np.searchsorted(w[6, :], draw, side='left')
                # print(draw, index,N)

                p2 = np.asarray([p_all[int(w[7, index[j]])]  for j in range(N)]) # saved in a new array so that nothing is overwritten. 
                # print(p2)
                # print(w)
                #p2[:, 29] = np.asarray([w[4, w[7, index[j]]]  for j in range(N)]) 3 should do the same as line "p_all[:, 29] = w[4, :]"
                
                weights_sum = np.nansum(p2[:, 29])

                p2[:, 29] =  p2[:, 29] / weights_sum
    #

                p_all = np.asarray(copy.deepcopy(p2))



            avg_vel = 0.
            avg_mass = 0.
            avg_kappa = 0.
            avg_sigma = 0.
            avg_mag = 0.

            for i in range(N):
                ## if printing averages to terminal,, uncomment next 6 lines:
                avg_vel += np.linalg.norm([p_all[i, 3], p_all[i, 4], p_all[i, 5]]) *  p_all[i, 29]
                avg_mass += p_all[i, 6] *  p_all[i, 29]
                avg_kappa += p_all[i, 9] *  p_all[i, 29]
                avg_sigma += p_all[i, 10] *  p_all[i, 29]
                avg_mag += p_all[i, 11] *  p_all[i, 29]
            
            print('mean velocity: ', avg_vel )       
            print('mean mass: ', avg_mass)
            print('mean kappa: ', avg_kappa)
            print('mean sigma: ', avg_sigma * 1e6)
            print('mean M_v: ', avg_mag)
            print('observed M_vs: ', lum_info)
                
                

            ## TIMER
            #t_5 = timeit.default_timer()-t_pfstart-t_4
            #print('time to resample', t_5)
            
            # save resulting table in HDU fits file
            p_out = np.hstack((p_all, 
                               np.ones((N,1))*tk, 
                               np.array([data_t[1,t] for i in range(N)]).reshape(-1, 1)))


            results = fits.BinTableHDU.from_columns([fits.Column(name='time', format='D', array=p_out[:, 42]),
                                           fits.Column(name='datetime', format='25A', array=p_out[:, 43]),
                                           fits.Column(name='X_geo', format='D', array=p_out[:, 0]),
                                           fits.Column(name='Y_geo', format='D', array=p_out[:, 1]),
                                           fits.Column(name='Z_geo', format='D', array=p_out[:, 2]),
                                           fits.Column(name='X_geo_DT', format='D', array=p_out[:, 3]),
                                           fits.Column(name='Y_geo_DT', format='D', array=p_out[:, 4]),
                                           fits.Column(name='Z_geo_DT', format='D', array=p_out[:, 5]),
                                           fits.Column(name='X_eci', format='D', array=p_out[:, 36]),
                                           fits.Column(name='Y_eci', format='D', array=p_out[:, 37]),
                                           fits.Column(name='Z_eci', format='D', array=p_out[:, 38]),
                                           fits.Column(name='X_eci_DT', format='D', array=p_out[:, 39]),
                                           fits.Column(name='Y_eci_DT', format='D', array=p_out[:, 40]),
                                           fits.Column(name='Z_eci_DT', format='D', array=p_out[:, 41]),
                                           fits.Column(name='mass', format='D', array=p_out[:, 6]),
                                           fits.Column(name='cd', format='D', array=p_out[:, 7]),
                                           fits.Column(name='A', format='D', array=p_out[:, 8]),
                                           fits.Column(name='kappa', format='D', array=p_out[:, 9]),
                                           fits.Column(name='sigma', format='D', array=p_out[:, 10]),
                                           fits.Column(name='mag_v', format='D', array=p_out[:, 11]),
                                           fits.Column(name='tau', format='D', array=p_out[:, 12]),
                                           fits.Column(name='Q_x', format='D', array=p_out[:, 13]),
                                           fits.Column(name='Q_y', format='D', array=p_out[:, 14]),
                                           fits.Column(name='Q_z', format='D', array=p_out[:, 15]),
                                           fits.Column(name='Q_v_x', format='D', array=p_out[:, 16]),
                                           fits.Column(name='Q_v_y', format='D', array=p_out[:, 17]),
                                           fits.Column(name='Q_v_z', format='D', array=p_out[:, 18]),
                                           fits.Column(name='Q_m', format='D', array=p_out[:, 19]),
                                           fits.Column(name='Q_cd', format='D', array=p_out[:, 20]),
                                           fits.Column(name='Q_cl', format='D', array=p_out[:, 21]),
                                           fits.Column(name='Q_k', format='D', array=p_out[:, 22]),
                                           fits.Column(name='Q_s', format='D', array=p_out[:, 23]),
                                           fits.Column(name='Q_tau', format='D', array=p_out[:, 24]),
                                           fits.Column(name='brightness', format='D', array=p_out[:, 25]),
                                           fits.Column(name='rho', format='D', array=p_out[:, 26]),
                                           fits.Column(name='parent_index', format='D', array=p_out[:, 27]),
                                           fits.Column(name='orig_index', format='D', array=p_out[:, 28]),
                                           fits.Column(name='weight', format='D', array=p_out[:, 29]),
                                           fits.Column(name='D_DT', format='D', array=p_out[:, 30]),
                                           fits.Column(name='latitude', format='D', array=[np.rad2deg(float(x)) for x in p_out[:, 31]]),
                                           fits.Column(name='longitude', format='D', array=[np.rad2deg(float(x)) for x in p_out[:, 32]]),
                                           fits.Column(name='height', format='D', array=p_out[:, 33]),
                                           fits.Column(name='lum_weight', format='D', array=p_out[:, 34]),
                                           fits.Column(name='pos_weight', format='D', array=p_out[:, 35])])

            results_list = fits.open(name, mode='append')
            results_list.append(results)
            results_list[-1].name='time_'+str(t)
            results_list.flush()#name, clobber=True)
            results_list.close()
            print('data saved to ', name)

            # save this table in its own righ as an 'end' file in case code is interrupted.
            # this means only this will need to be read in rather than trying to extract 
            # end table only from large HDU file
            name_end = name.replace(".fits", "_end.fits")
            results.writeto(name_end, overwrite=True)

            if resamp:
                # for i in range(N):
                p_all[:, 29] = 1./N
                p_all[:, 35] = 1./N
                p_all[:, 34] = 1./N


            # end this iteration
            print("now I've done collective things, start again. end timestep #", t, "at time ", tk, "secs")
        
            ## TIMER
            #t_6 = timeit.default_timer()-t_pfstart-t_5
            #print('time to resample', t_6)

        comm.Barrier()  ## all ranks are held while master performs resampling.

    print("we're all happy workers :-). Now saving all data to one table")
    comm.Barrier()

    ##########  Master saves table with all particles ########
    ## master extracts all HDU tables and appends them to one large output table 
    ## for plotting together.
    if rank==0:
        tabs = fits.open(name)

        nrows = int(N *T)
        all_results = fits.BinTableHDU.from_columns(tabs[1].columns, nrows=nrows)

        for colname in tabs[1].columns.names:
            for i in range(2,T+1):
                j = i-1
                all_results.data[colname][N*j:N*j+N] = tabs[i].data[colname]
        
        name= os.path.join(working_dir ,"outputs", out_name, out_name + version + '_' + eci_name + '_mass'+ str(mass_opt) + '_outputs_all.fits')

        all_results.writeto(name, clobber=True)

        print('data saved to ', name)


        ## saves a table of means for each timestep
        mean_results = fits.BinTableHDU.from_columns(tabs[1].columns, nrows=T)

        for colname in tabs[1].columns.names:
            for i in range(2, T+1):
                if colname != 'time' and colname != 'datetime':
                    col_data = sum(tabs[i].data[colname] * tabs[i].data['weight'])
                    mean_results.data[colname][i-1] = col_data

                else:
                    col_data = tabs[i].data[colname]
                    mean_results.data[colname][i-1] = col_data[0]

        
        name= os.path.join(working_dir ,"outputs", out_name, out_name + version + '_' + eci_name + '_mass'+ str(mass_opt) + '_outputs_mean.fits')

        mean_results.writeto(name, clobber=True)

        print('mean data saved to ', name)


#------------------------------------------------------------------------------
# hack to fix lsoda problem
#------------------------------------------------------------------------------
#
def fileno(file_or_fd):
    fd = getattr(file_or_fd, 'fileno', lambda: file_or_fd)()
    if not isinstance(fd, int):
        raise ValueError("Expected a file (`.fileno()`) or a file descriptor")
    return fd

@contextlib.contextmanager
def stdout_redirected(to=os.devnull, stdout=None):
    """
    https://stackoverflow.com/a/22434262/190597 (J.F. Sebastian)
    """
    if stdout is None:
       stdout = sys.stdout

    stdout_fd = fileno(stdout)
    # copy stdout_fd before it is overwritten
    #NOTE: `copied` is inheritable on Windows when duplicating a standard stream
    with os.fdopen(os.dup(stdout_fd), 'wb') as copied: 
        stdout.flush()  # flush library buffers that dup2 knows nothing about
        try:
            os.dup2(fileno(to), stdout_fd)  # $ exec >&to
        except ValueError:  # filename
            with open(to, 'wb') as to_file:
                os.dup2(to_file.fileno(), stdout_fd)  # $ exec > to
        try:
            yield stdout # allow code to be run with the redirected stdout
        finally:
            # restore stdout to its previous value
            #NOTE: dup2 makes stdout_fd inheritable unconditionally
            stdout.flush()
            os.dup2(copied.fileno(), stdout_fd)  # $ exec >&copied



##########  End of particle filter code ########

