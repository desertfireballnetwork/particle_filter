# import modules

# general
from __future__ import division
from math import *
import copy, operator, random, datetime
import os, sys
import contextlib

# science
import numpy as np
import scipy 
from scipy import linalg, integrate, stats, interpolate
from numpy.linalg import norm
from scipy.optimize import minimize, leastsq

# Astropy
from astropy.table import Table, Column, join, hstack, vstack
from astropy.time import Time, TimeDelta
from astropy import units as u
from astropy.coordinates import SkyCoord, EarthLocation, AltAz


#------------------------------------------------------------------------------
# hack to fix lsoda problem in nonlinear integration
#------------------------------------------------------------------------------

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

#------------------------------------------------------------------------------
# uses nrlmsise_00 to determine atmospheric density and temp
#------------------------------------------------------------------------------
def Atm_nrlmsise_00(fireball_info):
    """function input ALTITUDE in metres and finds the bounding altitudes from
       the look-up table. A linear (y=m*x+b) equation is created from these
       two data points and the given altitude h is used to return the according
       atmospheric density
    """
    import nrlmsise_00 as nrl


    from nrlmsise_00_header import nrlmsise_flags
    from nrlmsise_00_header import nrlmsise_input
    from nrlmsise_00_header import nrlmsise_output

    flags = nrlmsise_flags()
    Input = nrlmsise_input()
    output = nrlmsise_output()



 # *   Switches: to turn on and off particular variations use these switches.
 # *   0 is off, 1 is on, and 2 is main effects off but cross terms on.
 # *
 # *   Standard values are 0 for switch 0 and 1 for switches 1 to 23. The
 # *   array "switches" needs to be set accordingly by the calling program.
 # *   The arrays sw and swc are set internally.
 # *
 # *   switches[i]:
 # *    i - explanation
 # *   -----------------
 # *    0 - output in centimeters instead of meters
 # *    1 - F10.7 effect on mean
 # *    2 - time independent
 # *    3 - symmetrical annual
 # *    4 - symmetrical semiannual
 # *    5 - asymmetrical annual
 # *    6 - asymmetrical semiannual
 # *    7 - diurnal
 # *    8 - semidiurnal
 # *    9 - daily ap [when this is set to -1 (!) the pointer
 # *                  ap_a in struct nrlmsise_input must
 # *                  point to a struct ap_array]
 # *   10 - all UT/long effects
 # *   11 - longitudinal
 # *   12 - UT and mixed UT/long
 # *   13 - mixed AP/UT/LONG
 # *   14 - terdiurnal
 # *   15 - departures from diffusive equilibrium
 # *   16 - all TINF var
 # *   17 - all TLB var
 # *   18 - all TN1 var
 # *   19 - all S var
 # *   20 - all TN2 var
 # *   21 - all NLB var
 # *   22 - all TN3 var
 # *   23 - turbo scale height var
 # */

    flags.switches[0] = 1   #cm vs m
    flags.switches[1] = 1   #F10.7 effect on mean ignored if =0.
    for i in range(2, 23):
        flags.switches[i]=1
    #flags.switches[9] = -1 #Daily AP. If you set this field to -1, the block uses the entire matrix of magnetic index information (APH) instead of APH(:,1)
    Input.g_lat=np.rad2deg(fireball_info[0]);
    Input.g_long=np.rad2deg(fireball_info[1]);
    Input.alt=fireball_info[2]/1000;
    Input.year=fireball_info[3]; #/* without effect */
    Input.doy=fireball_info[4]
    Input.sec=fireball_info[5];

 ##    Input[0].lst=     # local apparent solar time (hours), see note below */
 ##    Input[0].f107A=   # 81 day average of F10.7 flux (centered on doy) */
 ##    Input[0].f107=    # daily F10.7 flux for previous day */
 ##    Input[0].ap=4     # magnetic index(daily) */

    nrl.gtd7(Input, flags, output)
 #--------------------------------------------------------------------------
 #*   OUTPUT VARIABLES:
 #*      d[0] - HE NUMBER DENSITY(CM-3)
 #*      d[1] - O NUMBER DENSITY(CM-3)
 #*      d[2] - N2 NUMBER DENSITY(CM-3)
 #*      d[3] - O2 NUMBER DENSITY(CM-3)
 #*      d[4] - AR NUMBER DENSITY(CM-3)
 #*      d[5] - TOTAL MASS DENSITY(GM/CM3) [includes d[8] in td7d]
 #*      d[6] - H NUMBER DENSITY(CM-3)
 #*      d[7] - N NUMBER DENSITY(CM-3)
 #*      d[8] - Anomalous oxygen NUMBER DENSITY(CM-3)
 #*      t[0] - EXOSPHERIC TEMPERATURE
 #*      t[1] - TEMPERATURE AT ALT
 #*
 #*
 #*      O, H, and N are set to zero below 72.5 km
 #*
 #*      t[0], Exospheric temperature, is set to global average for
 #*      altitudes below 120 km. The 120 km gradient is left at global
 #*      average value for altitudes below 72 km.
 #*
 #*      d[5], TOTAL MASS DENSITY, is NOT the same for subroutines GTD7
 #*      and GTD7D
 #*
 #*        SUBROUTINE GTD7 -- d[5] is the sum of the mass densities of the
 #*        species labeled by indices 0-4 and 6-7 in output variable d.
 #*        This includes He, O, N2, O2, Ar, H, and N but does NOT include
 #*        anomalous oxygen (species index 8).
 #*
 #*        SUBROUTINE GTD7D -- d[5] is the "effective total mass density
 #*        for drag" and is the sum of the mass densities of all species
 #*        in this model, INCLUDING anomalous oxygen.
 #------------------------------------------------------------------------

    T= output.t[1]

    HE=output.d[0]*1e6
    O =output.d[1]*1e6
    N2=output.d[2]*1e6
    O2=output.d[3]*1e6
    AR=output.d[4]*1e6
    H =output.d[6]*1e6
    N =output.d[7]*1e6

    sum_mass= HE + O + N2 + O2 +AR + H + N

    HE_mass= 4.0026
    O_mass= 15.9994
    N2_mass= 28.013
    O2_mass= 31.998
    AR_mass= 39.948
    H_mass= 1.0079
    N_mass= 14.0067

    mol_mass_air= HE_mass * HE/sum_mass + O_mass*O/sum_mass + N2_mass*N2/sum_mass + O2_mass*O2/sum_mass + AR_mass*AR/sum_mass + H_mass*H/sum_mass + N_mass*N/sum_mass

    po=output.d[5]

    R=8.3144621/mol_mass_air*1000     #287.058 #J/(kg*K) in SI units, and 53.35 (ft*lbf)/(lbm* deg_R)

    atm_pres = po*T*R

    return [T, po, atm_pres]

    
#------------------------------------------------------------------------------
# inputs
#------------------------------------------------------------------------------ 
    
    
def RoughTriangulation(data, t0, rev=False, eci_bool=False):
    """ performs a SLLS on 6 points at start/end to determine a good 
        starting position for a 3D ray particle filter. 
        
        data: fireball dable data of multiple viewpoints. Output of Geo_Fireball_Data
        t0:   the relative time since start of fireball 
              from where the particle filter is to be initiated. 
              For example, if only end points are being used, then start of fireball
              will not be the start of the filter and t0 could be 0.02 or 0.06.
        rev:  bool, True if performing filter in reverse
    """
    from PointWiseTriangulation import PointTriangulation
    
    data.sort('time')    
    [T_rel, T_rel_counts] = np.unique(data['time'], return_counts=True)
    T_rel_plus = T_rel[T_rel_counts > 1]

    ## Assign the output name ['yyyymmdd']
    T0 = Time(data['datetime'][0], format='isot', scale='utc')
    yday = T0.yday.split(':'); y = float(yday[0]); d = float(yday[1])
    s = float(yday[2]) * 60 * 60 + float(yday[3]) * 60 + float(yday[4])
    atm_info = np.vstack((y, d, s))

    # Check the number of paired points
    n = 8
    if len(T_rel_plus) < 3:
        print('Cannot roughly triangulate! Consider using SLLS code.'); exit()
    elif len(T_rel_plus) < 8:
        print('The initial estimates are quite rough: Particle filter may not work :(')
        n = len(T_rel_plus)

    # Crop to only n data points
    if rev:  # if running in reverse, take last points.
        T_rel_crop = T_rel_plus[-n:]
        CroppedData = data[data['time'] >= np.min(T_rel_crop)]
    else:  # if running forward, take first points.
        T_rel_crop = T_rel_plus[:n]
        CroppedData = data[data['time'] <= np.max(T_rel_crop)]

    CroppedData['altitude'] = np.rad2deg(CroppedData['altitude'])
    CroppedData['azimuth'] = np.rad2deg(CroppedData['azimuth'])

    # Find the ECEF positions of n paired data points
    Pos_pw = PointTriangulation(CroppedData)[0] #[3,n]

    # Pos_pw from PointTriangulation is positions in ECEF. 
    # If data has been imported in ECI (eci_bool=True)
    # then Pos_pw needs to be converted to ECI:
    if eci_bool:
        t_jd = T0.jd + T_rel_crop / (24*60*60)
        Pos_pw = ECEF2ECI_pos(Pos_pw, t_jd)

    def residuals(X, Pos_pw, dt):
        X = np.vstack((X)); [x0, v0, a0] = [X[:3], X[3:6], X[6]]
        Pos = x0 + v0 * dt + 0.5 * v0/norm(v0) * a0 * dt**2
        return (Pos_pw - Pos).flatten()

    # Pack variables and least square them
    X_est = np.ones(7); X_est[:3] = Pos_pw[:,0]
    if rev: X_est[:3] = Pos_pw[:,-1]

    [X, X_cov_reduced, infodict] = leastsq(residuals, X_est, 
                args=(Pos_pw, T_rel_crop - t0), full_output=True)[:3]

    # Parameter standard deviation
    res_vector = infodict['fvec']
    reduced_chi_sq = np.sum(res_vector**2) / (len(res_vector) - len(X))
    X_cov = X_cov_reduced * reduced_chi_sq
    X_std = np.sqrt(np.diag(X_cov))

    # Unpack mean and std variables
    x0 = np.vstack((X[:3])); x0_err = np.vstack((X_std[:3]))
    v0 = np.vstack((X[3:6])); v0_err = np.vstack((X_std[3:6]))



    return x0, v0, x0_err, v0_err, atm_info
    

def Fireball_Data(filenames, reverse=False):
    """
       This is where a list of files gets read into astropy data tables
       For every input file if observation times are reliable, the 
       observer lat,long,height are recorded, along with any pre triangulated data
       
    """
    from dfn_utils import is_type_pipeline, has_reliable_timing

    data_packet = []
    data_tables = []
    alpha = []
    beta = []
    t0 = []
    x0 = np.arange(3)

    # for flight angle calculations, need a reference location for ENU transformations
    approx_Data = Table.read(filenames[0], format='ascii.ecsv', guess=False, delimiter=',', data_start = 1, data_end = 2)

    # Compute the transformation matrix
    Trans = ECEF2ENU(np.deg2rad(approx_Data['longitude'][0]), np.deg2rad(approx_Data['latitude'][0])) 

    for camfile in filenames:
        # use this if statement to check the incoming ecsv files are (reliable) datafiles. Here we just check that timing uncertainties are sufficient for useable data
        t = Table.read(camfile, format='ascii.ecsv', guess=False, delimiter=',')
        if has_reliable_timing(t):
            print(camfile)
            
            Data = t

            # Extract the individual camera locations from metadata
            obs_lat = np.deg2rad(Data.meta['obs_latitude'])
            obs_lon = np.deg2rad(Data.meta['obs_longitude'])
            obs_hei = Data.meta['obs_elevation']
            
            ## Convert altitude and azimuth observations into radians
            alt = np.deg2rad(Data['altitude'])
            azi = np.deg2rad(Data['azimuth'])

            # Convert from spherical to geocentric cartesian coordinates (ENU)
            UV_ENU = np.vstack((np.cos(alt) * np.sin(azi),
                                np.cos(alt) * np.cos(azi),
                                np.sin(alt)))
                                
            # Convert from ENU to ECEF coordinates
            UV_ECEF = np.dot(ENU2ECEF(obs_lon, obs_lat), UV_ENU)

            # Create the table columns:
            
            ## UTC datetime format as string : "yyyy-mm-dd 'T' hh:mm:ss.decimal" 
            date_time_col = Column(name='datetime', data=Data['datetime'])
            
            ## Camera locations
            obs_lat_col = Column(name='obs_lat', data=np.ones(len(Data['datetime'])) * obs_lat)
            obs_lon_col = Column(name='obs_lon', data=np.ones(len(Data['datetime'])) * obs_lon)
            obs_hei_col = Column(name='obs_hei', data=np.ones(len(Data['datetime'])) * obs_hei)
           
            ## Unit vector of fireball observations from camera
            UV_i_col = Column(name='UV_i', data=UV_ECEF[0])
            UV_j_col = Column(name='UV_j', data=UV_ECEF[1])
            UV_k_col = Column(name='UV_k', data=UV_ECEF[2])
            
            ## Make uniform observation weighting of 0.1... for now. ##TODO##
            ang_error = np.deg2rad(0.1)
            ang_error_col = Column(name='R_UV', data=np.ones(len(alt)) * ang_error)
            
            ## altitude and azimuth data with errors
            alt_col = Column(name='altitude', data=alt)
            azi_col = Column(name='azimuth', data=azi)
            alt_err_col = Column(name='R_alt', data=np.deg2rad(Data['err_plus_altitude']))
            azi_err_col = Column(name='R_azi', data=np.deg2rad(Data['err_plus_azimuth']))
            
            ## check if SLLS has been run in ECI:
            if 'X_eci' in Data.colnames:
                ## positions and velocities calculated by triangulation script for plotting comparison later:
                ## meteoroid positions from triangulated data
                x_col = Column(name='X_eci', data=Data['X_eci'])
                y_col = Column(name='Y_eci', data=Data['Y_eci'])
                z_col = Column(name='Z_eci', data=Data['Z_eci'])

                ## errors in position calculated from triangulation (Note: only taken the positive errors)
                ##TODO: currently cross track error used for ECI cartesian errors. 
                
                x_err_col = Column(name='R_X_eci', data=np.ones(len(Data['cross_track_error'])) * 100.)  #Data['X_err_plus'])
                y_err_col = Column(name='R_Y_eci', data=np.ones(len(Data['cross_track_error'])) * 100.)  #Data['cross_track_error']* 10.)  #Data['Y_err_plus'])
                z_err_col = Column(name='R_Z_eci', data=np.ones(len(Data['cross_track_error'])) * 100.)  #Data['cross_track_error']* 10.)  #Data['Z_err_plus'])

                ## velocities calculated by triangulation script positions with time. 
                vels = [np.nan]
                f_time = Time(Data['datetime'].data.tolist(), format='isot', scale='utc') # not sure if this works: t = Time(data['datetime'])

                for i in range(1,len(Data['X_eci'])):
                    vels.append(norm([(Data['X_eci'][i] - Data['X_eci'][i-1]),
                                      (Data['Y_eci'][i] - Data['Y_eci'][i-1]),
                                      (Data['Z_eci'][i] - Data['Z_eci'][i-1])])
                                      /((f_time[i].jd - f_time[i-1].jd)*3600*24))

                ddt = Column(name='D_DT', data=vels)  

                eci_bool = True

            else:

                ## positions and velocities calculated by triangulation script for plotting comparison later:
                ## meteoroid positions from triangulated data
                x_col = Column(name='X_geo', data=Data['X_geo'])
                y_col = Column(name='Y_geo', data=Data['Y_geo'])
                z_col = Column(name='Z_geo', data=Data['Z_geo'])

                ## errors in position calculated from triangulation (Note: only taken the positive errors)
                ##TODO: currently cross track error used for ECEF cartesian errors. 
             
                x_err_col = Column(name='R_X_geo', data=np.ones(len(Data['cross_track_error'])) * 100.)  #Data['X_err_plus'])
                y_err_col = Column(name='R_Y_geo', data=np.ones(len(Data['cross_track_error'])) * 100.)  #Data['Y_err_plus'])
                z_err_col = Column(name='R_Z_geo', data=np.ones(len(Data['cross_track_error'])) * 100.)  #Data['Z_err_plus'])

                ## velocities calculated by triangulation script positions with time. 
                vels = [np.nan]
                f_time = Time(Data['datetime'].data.tolist(), format='isot', scale='utc') # not sure if this works: t = Time(data['datetime'])

                for i in range(1,len(Data['X_geo'])):
                    vels.append(norm([(Data['X_geo'][i] - Data['X_geo'][i-1]),
                                      (Data['Y_geo'][i] - Data['Y_geo'][i-1]),
                                      (Data['Z_geo'][i] - Data['Z_geo'][i-1])])
                                      /((f_time[i].jd - f_time[i-1].jd)*3600*24))

                ddt = Column(name='D_DT', data=vels)  

                eci_bool = False

            ## velocities calculated by triangulation script positions with time. 
            ddtgeo = Column(name='D_DT_geo', data=Data['D_DT_geo']) 

            # add a column for cross track error
            pos_err = Column(name='cross_track_error', data=Data['cross_track_error'] * 10.)

            # for alpha calculations, also for plotting in spherical later. 
            lat_col = Column(name='Lat_rad', data=np.deg2rad(Data['latitude']))
            lon_col = Column(name='Lon_rad', data=np.deg2rad(Data['longitude']))
            lat_col_deg = Column(name='latitude', data=Data['latitude'])
            lon_col_deg = Column(name='longitude', data=Data['longitude'])
            height_col = Column(name='height', data=Data['height'])

            ## flight angle at start:
            ### Construct the ECEF position vector
            X_ECEF = np.vstack((x_col, y_col, z_col))

            ### Current position in local ENU coords
            ENU = np.dot(Trans, X_ECEF)
            temp = [np.arctan((ENU[2,i]-ENU[2,i+1])  / np.sqrt((ENU[0,i]-ENU[0,i+1])**2 + (ENU[1,i]-ENU[1,i+1])**2) ) for i in range(len(ENU[0])-1)]
            temp.append(temp[-1])

            gamma = Column(name='gamma', data=temp)

            ## calculate gravity component --> (grav * sin(gamma))
            [G, M] = grav_params()
            g_sin_gamma = Column(name='g_sin_gamma', data=[G*M*sin(temp[i])/(ENU[2, i]**2) for i in range(len(ENU[0]))])

            ## check whether there is a luminosity column to import and import data whether complete or nan values.
            if is_type_pipeline(Data, 'absolute_photometric'):
                print('Luminosity values detected')

                t = Time(Data['datetime'])
                rel_t = (t - t[0]).sec
                # create missing values mask
                w = np.isnan(Data['M_V'])

                # create interpolator using only valid input
                interp_magnitudes = interpolate.interp1d(rel_t[~w], Data['M_V'][~w],  bounds_error=False)
                #interp_luminosity = interpolate.interp1d(rel_t[~w], Data['I'][~w],  bounds_error=False) # Depreciated. If you want I's, calculate and add ;)

                # overwrite values  
                Data['M_V'] = interp_magnitudes(rel_t)
                #Data['I'] = interp_luminosity(rel_t) # Depreciated. If you want I's, calculate and add ;)


                Data['M_V'] = Data['M_V']
                #Data['I'] = Data['I']# Depreciated. If you want I's, calculate and add ;)

                mag_col = Column(name='magnitude', data=Data['M_V'])
                #lum_col = Column(name='luminosity', data=Data['I'])
                mag_err = Column(name='mag_error', data=np.ones(len(Data['datetime'])) *0.5) # TODO currently luminosity error is fixed at 0.5 Magnitudes

            else:
                mag_col = Column(name='magnitude', data=np.ones(len(Data['datetime'])) * np.nan)
                #lum_col = Column(name='luminosity', data=np.ones(len(Data['datetime'])) * np.nan)
                mag_err = Column(name='mag_error', data=np.ones(len(Data['datetime'])) * np.nan)

            if 'ballistic_entry_mass_all' in Data.meta:
                alpha = Data.meta['ballistic_alpha_all']
                beta = Data.meta['ballistic_beta_all']
            # Create data packets for each camera
            data_packet.append(Table([date_time_col, obs_lat_col,obs_lon_col, obs_hei_col,
                                      UV_i_col, UV_j_col, UV_k_col, ang_error_col,
                                      alt_col, azi_col, alt_err_col, azi_err_col, 
                                      ddt, ddtgeo, x_col, y_col, z_col, x_err_col, y_err_col, z_err_col, 
                                      mag_col, mag_err, #lum_col, 
                                      lat_col, lon_col, lat_col_deg, lon_col_deg, height_col, gamma, g_sin_gamma, pos_err]))

            if norm([x_col[0], y_col[0],z_col[0]]) > norm(x0):
                x0 = [x_col[0], y_col[0], z_col[0]]

    # Stack the data packets 
    data = vstack(data_packet)                    

    # Find the absolute beginning of bright flight
    All_time = Time(data['datetime'].data.tolist(), format='isot', scale='utc') # not sure if this works: t = Time(data['datetime'])

    if not t0:
        t0 = All_time[np.argmin(All_time.jd)]

    # Change the time column to relative time (sec)
    rel_time = np.round((All_time.jd - t0.jd) * (24. * 60. * 60.), 2)

    time_col = Column(name='time', data=rel_time)
    data.add_column(time_col, index=1) 

    # Sort the data in time
    data.sort('time')    
    if reverse:
        data.reverse()

    # really only needed for 1D particle filter data but thought it was worth calculating and saving for all anyway
    if eci_bool:    
        lengths = [pow(float(data['X_eci'][i] - x0[0])**2 + float(data['Y_eci'][i] - x0[1])**2 + float(data['Z_eci'][i] - x0[2])**2, 1/2.) for i in range(len(data))]
        uv = np.array([[(data['X_eci'][-1] - x0[0]) / lengths[-1]], [(data['Y_eci'][-1] - x0[1]) / lengths[-1]], [(data['Z_eci'][-1] - x0[2]) / lengths[-1]]])
    else:
        lengths = [pow(float(data['X_geo'][i] - x0[0])**2 + float(data['Y_geo'][i] - x0[1])**2 + float(data['Z_geo'][i] - x0[2])**2, 1/2.) for i in range(len(data))]
        uv = np.array([[(data['X_geo'][-1] - x0[0]) / lengths[-1]], [(data['Y_geo'][-1] - x0[1]) / lengths[-1]], [(data['Z_geo'][-1] - x0[2]) / lengths[-1]]])
    

    dfs_col = Column(name='dist_from_start', data= lengths)
    data.add_column(dfs_col, index=1) 
    
    # calculate ballistic parameters if they don't already exist
    if not alpha:
        Gparams= Q4_min(data)

        alpha = Gparams[0]
        beta = Gparams[1]

    data.add_column(Column(name='alpha', data=np.ones(len(data['datetime'])) * alpha))     
    data.add_column(Column(name='beta', data=np.ones(len(data['datetime'])) * beta)) 

    return data, data['time'][0], t0, eci_bool


#------------------------------------------------------------------------------
# Non-linear estimation of x[k]
#------------------------------------------------------------------------------
def NL_state_eqn_3d(X, t, param):

    [mu, po, grav] = param
    
    Xdot=[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]
    if X[6]>0.001:
        vel_x = X[3]
        vel_y = X[4]
        vel_z = X[5]
        mass  = X[6]
        K = X[9]
        sig = X[10]

        velocity = sqrt(vel_x**2 + vel_y**2 +vel_z**2)

        kv = 0.5 * K * po
        km = kv * sig

        Xdot[0] = vel_x
        Xdot[1] = vel_y
        Xdot[2] = vel_z
        Xdot[3] = -kv*pow(abs(mass),(mu-1))*vel_x*velocity + grav[0]
        Xdot[4] = -kv*pow(abs(mass),(mu-1))*vel_y*velocity + grav[1]
        Xdot[5] = -kv*pow(abs(mass),(mu-1))*vel_z*velocity + grav[2]
        Xdot[6] = -km*pow(velocity,3)*pow(abs(mass),mu)

    return Xdot

def NL_state_eqn_2d(X, t, param): 
    [mu,  po, g_sin_gamma] = param
    [l, v, m, k, s, u] = [X[0], X[3], X[6], X[9], X[10], X[12]] 
    Xdot=[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.] 

    ## constants
    kv = 0.5 * k * po
    km = kv * s 

    Xdot[0] = v
    Xdot[3] = -kv*pow(v,2)*pow(abs(m),(mu-1))+ g_sin_gamma
    Xdot[6] = -km*pow(v,3)*pow(abs(m),mu)
    #Xdot[12] = 0#pow(abs(m),(mu-1)) * mu * po * k * u * v /2 * (s * v**2 + 2)   -   u * v * Xdot[1] * Xdot[2]  
    
    return Xdot #
 
#------------------------------------------------------------------------------
# Linearisation of continuous time approximated matrices
#------------------------------------------------------------------------------
def Q_mx_3d(t1, t2, init_x, mu, po, Qc, rev=False):
    
    # Qc[3] = pow(np.sqrt(Qc[3])*init_x[3], 2)
    # Qc[4] = pow(np.sqrt(Qc[4])*init_x[4], 2)
    # Qc[5] = pow(np.sqrt(Qc[5])*init_x[5], 2)

    Qc[6] = pow(np.sqrt(Qc[6])*init_x[6], 2)

    length = len(Qc)
    Qc = np.diag(Qc)
    Qd0 = np.zeros(length**2)


    [l_x, l_y, l_z, v_x, v_y, v_z, m , cd, cl, k, s, I, tau] = init_x

    v = pow((v_x**2 + v_y**2+v_z**2), 0.5)

    vxvx = -po/2 * k * pow(abs(m), mu-1) * (2 * v_x**2 + v_y**2 + v_z**2) / v 
    vxvy = -po/2 * k * v_x * v_y * pow(abs(m), mu-1) / v
    vxvz = -po/2 * k * v_x * v_z * pow(abs(m), mu-1) / v
    vxm = -po/2 *(mu-1) * k * v_x * pow(abs(m), (mu-2)) * v
    vxk = -po/2 * v_x * pow(abs(m), mu-1) * v 

    vyvy = -po/2 * k * pow(abs(m), mu-1) * (2 * v_y**2 + v_x**2 + v_z**2) / v
    vyvx = -po/2 * k * v_y * v_x * pow(abs(m), mu-1) / v
    vyvz = -po/2 * k * v_y * v_z * pow(abs(m), mu-1) / v
    vym = -po/2 *(mu-1) * k * v_y * pow(abs(m), (mu-2)) * v
    vyk = -po/2 * v_y * pow(abs(m), mu-1) * v 

    vzvz = -po/2 * k * pow(abs(m), mu-1) * (2 * v_z**2 + v_x**2 + v_y**2) / v
    vzvx = -po/2 * k * v_z * v_x * pow(abs(m), mu-1) / v
    vzvy = -po/2 * k * v_z * v_y * pow(abs(m), mu-1) / v
    vzm = -po/2 *(mu-1) * k * v_z * pow(abs(m), (mu-2)) * v
    vzk = -po/2 * v_z * pow(abs(m), mu-1) * v 



    mvx = -3/2 * po * k * s * v_x * v * pow(abs(m), mu)
    mvy = -3/2 * po * k * s * v_y * v * pow(abs(m), mu)
    mvz = -3/2 * po * k * s * v_z * v * pow(abs(m), mu)
    mm  = -po/2 * mu * k * s * pow(v, 3) * pow(abs(m), mu-1)
    ms  = -po/2 * k * pow(v, 3) * pow(abs(m), mu)
    mk  = -po/2 * s * pow(v, 3) * pow(abs(m), mu)



    ## TODO brightness values...

    f = np.matrix([ [0., 0.,0.,  1.,  0.,  0., 0.,0., 0., 0.,  0., 0., 0.], 
                    [0., 0.,0.,  0.,  1.,  0., 0.,0., 0., 0.,  0., 0., 0.], 
                    [0., 0.,0.,  0.,  0.,  1, 0., 0., 0., 0.,  0., 0., 0.], 
                    [0., 0.,0.,vxvx,vxvy,vxvz,vxm,0., 0., vxk, 0., 0., 0.], 
                    [0., 0.,0.,vyvx,vyvy,vyvz,vym,0., 0., vyk, 0., 0., 0.], 
                    [0., 0.,0.,vzvx,vzvy,vzvz,vzm,0., 0., vzk, 0., 0., 0.], 
                    [0., 0.,0., mvx, mvy, mvz, mm,0., 0., mk,  ms, 0., 0.], 
                    [0., 0.,0.,  0.,  0.,  0., 0.,0., 0., 0.,  0., 0., 0.], 
                    [0., 0.,0.,  0.,  0.,  0., 0.,0., 0., 0.,  0., 0., 0.], 
                    [0., 0.,0.,  0.,  0.,  0., 0.,0., 0., 0.,  0., 0., 0.], 
                    [0., 0.,0.,  0.,  0.,  0., 0.,0., 0., 0.,  0., 0., 0.], 
                    [0., 0.,0.,  0.,  0.,  0., 0.,0., 0., 0.,  0., 0., 0.], 
                    [0., 0.,0.,  0.,  0.,  0., 0.,0., 0., 0.,  0., 0., 0.]])

    #TODO... check the reverse integration times
    param = [length, f, Qc] 
    calc_time = [0, abs(t1-t2)]
    with stdout_redirected():
        Qd = integrate.odeint(Qd_3d_integ, Qd0, calc_time, args = (param,), mxstep=10)
    Qd = np.reshape(Qd[1, :], (length, length))
    # Qd = np.diag(Qd)
    
    if rev:
        return np.diag(Qc-Qd)
    else:
        return np.diag(Qc+Qd)

def Qd_3d_integ(X, t, param):

    [length, f, Qc] = param
    Qd = np.asmatrix(np.reshape(X, (length, length)))

    f_tau =np.multiply(t, f)
    fT_tau =np.multiply(t, f.getT())

    e_f_tau = scipy.linalg.expm3(f_tau, q=20) # q is order of taylor series
    e_f_tau = np.asmatrix(e_f_tau)

    e_fT_tau = scipy.linalg.expm3(fT_tau, q=20) # q is order of taylor series
    e_fT_tau = np.asmatrix(e_fT_tau)

    Qd_dot = e_f_tau * Qc * e_fT_tau

    Xdot = np.reshape(np.asarray(Qd_dot), -1)
    return Xdot

def Q_mx_1d(t1, t2, init_x, mu, po, Qc, reverse=False): 
    # Qc[3] = pow(np.sqrt(Qc[3])*init_x[3], 2)
    # Qc[4] = pow(np.sqrt(Qc[4])*init_x[4], 2)
    # Qc[5] = pow(np.sqrt(Qc[5])*init_x[5], 2)

    Qc[6] = pow(np.sqrt(Qc[6])*init_x[6], 2)

    length = len(Qc)
    Qc = np.diag(Qc)
    Qd0 =  np.reshape(Qc, -1)  #np.zeros(length**2)# 

    [l, v, m, k, s, u] = [init_x[0], init_x[3], init_x[6], init_x[9], init_x[10], init_x[12]] 

    mm = -po/2 * k * s * pow(v, 3) * pow(abs(m), mu) * mu / m
    mv = -po/2 * k * s * pow(v, 3) * pow(abs(m), mu) * 3  / v
    ms = -po/2 * k * s * pow(v, 3) * pow(abs(m), mu) / s 
    mk = -po/2 * k * s * pow(v, 3) * pow(abs(m), mu) / k
    vm = po/2  * k   *   pow(v, 2) * pow(abs(m), (mu-1)) * (mu-1)/ m
    vv = -po/2 * k   *   pow(v, 2) * pow(abs(m), (mu-1)) * 2 / v
    vk = -po/2 * k   *   pow(v, 2) * pow(abs(m), (mu-1)) / k 

    f = np.matrix([ [0., 0., 0.,  1.,  0.,  0., 0., 0., 0., 0., 0., 0., 0.], 
                    [0., 0., 0.,  0.,  0.,  0., 0., 0., 0., 0., 0., 0., 0.], 
                    [0., 0., 0.,  0.,  0.,  0., 0., 0., 0., 0., 0., 0., 0.], 
                    [0., 0., 0.,  vv,  0.,  0., vm, 0., 0., vk, 0., 0., 0.], 
                    [0., 0., 0.,  0.,  0.,  0., 0., 0., 0., 0., 0., 0., 0.], 
                    [0., 0., 0.,  0.,  0.,  0., 0., 0., 0., 0., 0., 0., 0.], 
                    [0., 0., 0.,  mv,  0.,  0., mm, 0., 0., mk, ms, 0., 0.], 
                    [0., 0., 0.,  0.,  0.,  0., 0., 0., 0., 0., 0., 0., 0.], 
                    [0., 0., 0.,  0.,  0.,  0., 0., 0., 0., 0., 0., 0., 0.], 
                    [0., 0., 0.,  0.,  0.,  0., 0., 0., 0., 0., 0., 0., 0.], 
                    [0., 0., 0.,  0.,  0.,  0., 0., 0., 0., 0., 0., 0., 0.], 
                    [0., 0., 0.,  0.,  0.,  0., 0., 0., 0., 0., 0., 0., 0.], 
                    [0., 0., 0.,  0.,  0.,  0., 0., 0., 0., 0., 0., 0., 0.]]) 

    param = [length, f, Qc] 
    with stdout_redirected():
        Qd = integrate.odeint(Qd_1d_integ, Qd0, [t1, t2], args = (param,), mxstep=10)
    Qd = np.reshape(Qd[1, :], (length, length))
    Qd = np.diag(Qd) 

    return Qd 

def Qd_1d_integ(X, t, param): 
    [length, f, Qc] = param 
    Qd = np.asmatrix(np.reshape(X, (length, length))) 
    
    f_tau =np.multiply(t, f)
    fT_tau =np.multiply(t, f.getT()) 

    e_f_tau = scipy.linalg.expm(f_tau, q=20) # q is order of taylor series
    e_f_tau = np.asmatrix(e_f_tau) 

    e_fT_tau = scipy.linalg.expm(fT_tau, q=20) # q is order of taylor series
    e_fT_tau = np.asmatrix(e_fT_tau) 

    Qd_dot = e_f_tau * Qc * e_fT_tau 

    Xdot = np.reshape(np.asarray(Qd_dot), -1) 

    return Xdot 

#------------------------------------------------------------------------------
# Ballistic functions from M. Gritsevich 2007
#------------------------------------------------------------------------------


def Q4_min(data):
    alt=data['height']
    vel=data['D_DT']

    #Vels
    v0 = vel[0]
    Vvalues=vel/v0      #creates a matrix of V/Ve to give a dimensionless parameter for velocity

    # Height
    ho=7160
    Yvalues=alt/ho  #        %creates a matrix of h/h0 to give a dimensionless parameter for altitude
    
    params = np.vstack((Vvalues, Yvalues))

    #minfun = lambda x: np.sum( pow(2 * x[0] * exp(-y[1, :]) - (scipy.special.expi(x[1]) - scipy.special.expi(x[1]* y[0, :]**2) ) * exp(-x[1]) , 2))

    x0 = [100,2]
    bnds = ((0.001, 1000), (0.001, 100))
    res = minimize(min_fun, x0, args=(Vvalues, Yvalues),bounds=bnds)

    return res.x    

    #result = minimize(TotalAngSep, x_est, args=(obs_ECEF_all, UV_ECEF_all))#, options={'disp':True})

    #return sum(((x(1).*exp(-yData(2:end))).*2-(mfun('ei',x(2))-mfun('ei',(x(2).*(vData.^2)))).*exp(-x(2))).^2);
                                      #sum...alpha*e^-y*2         |______-del______________________________________|     *e^-beta
                                      
def min_fun(x, vvals, yvals):

    res = 0.
    for i in range(len(vvals)):
        res += pow(2 * x[0] * exp(-yvals[i]) - (scipy.special.expi(x[1]) - scipy.special.expi(x[1]* vvals[i]**2) ) * exp(-x[1]) , 2)
    return res

#------------------------------------------------------------------------------
# Coordinate transforms
#------------------------------------------------------------------------------

def ECEF2ENU(lon, lat):
    """
    # convert to local ENU coords
    # http://www.navipedia.net/index.php/Transformations_between_ECEF_and_ENU_coordinates
    # Title     Transformations between ECEF and ENU coordinates
    # Author(s)   J. Sanz Subirana, J.M. Juan Zornoza and M. Hernandez-Pajares, Technical University of Catalonia, Spain.
    # Year of Publication     2011 

    # use long to make greenwich mean turn to meridian: A clockwise rotation over the z-axis by and angle to align the east-axis with the x-axis
    # use lat to rotate z to zenith
    """
    
    ECEF2ENU = np.array([[-np.sin(lon)         , np.cos(lon)            , 0], 
        [-np.cos(lon)*np.sin(lat) , -np.sin(lon) * np.sin(lat), np.cos(lat)],
        [np.cos(lon) * np.cos(lat), np.sin(lon) * np.cos(lat) , np.sin(lat)]])

    return ECEF2ENU

def ECEF2ECI_pos(Pos_ECEF, t):

    T = Time(t, format='jd', scale='utc')
    
    Pos_ECEF_SC = ITRS(x=Pos_ECEF[0] * u.m, y=Pos_ECEF[1] * u.m, 
        z=Pos_ECEF[2] * u.m, obstime=T)
    Pos_ECI_SC = Pos_ECEF_SC.transform_to(GCRS(obstime=T))
    Pos_ECI = np.vstack(Pos_ECI_SC.cartesian.xyz.value)

    return Pos_ECI

def grav_params():
    """gravitational constant and mass of earth.
    """
    G=6.6726e-11    # N-m2/kg2
    M=5.98e24      # kg Mass of the earth 
    return G, M

def ENU2ECEF(lon, lat):
    """transform of ENU2ECEF
    """
    lon = float(lon); lat = float(lat)
    C_ENU2ECEF = np.array([[-np.sin(lon), -np.sin(lat) * np.cos(lon), np.cos(lat) * np.cos(lon)],
                           [ np.cos(lon), -np.sin(lat) * np.sin(lon), np.cos(lat) * np.sin(lon)],
                           [     0      ,         np.cos(lat)       ,        np.sin(lat)       ]])
    return C_ENU2ECEF