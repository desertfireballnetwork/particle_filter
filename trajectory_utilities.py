#!/usr/bin/env python

import subprocess

# import science modules
import numpy as np
import astropy.units as u
from astropy.time import Time
from numpy.linalg import norm
from scipy.linalg import block_diag
from astropy.coordinates import SkyCoord, EarthLocation, \
    get_body_barycentric, solar_system_ephemeris, \
    ITRS, ICRS, HCRS, GCRS, ITRS, PrecessedGeocentric

# Define constants
MU_EARTH = 3.986005000e14 #4418e14 # Earth's standard gravitational parameter (m3/s2)
MU_SUN = 1.32712440018e20  # Sun's standard gravitational parameter (m3/s2)
OMEGA_EARTH = 7.2921158553e-5  # Earth's rotation rate (rad/s)


class PoorTriangulationResiduals(Exception):
    '''
    Exception class used when StraightLineLeastSquares produces a poor SLLS residual
    '''
    pass


#------------------------------------------------------------------------------
# Determine Earths parameters
#------------------------------------------------------------------------------
def grav_params():
    """gravitational constant and mass of earth.
    """
    G=6.6726e-11    # N-m2/kg2
    M=5.98e24      # kg Mass of the earth 
    return G, M

def Gravity(ECEF):
    """uses [x, y, z]_ECEF position of a particle to get LLH
       in order to transform gravity at that location in ENU coordinates 
       to ECEF. Also outputs local altitude.
       particle[x, y, z]_ECEF--->>particle[LLH]
       gravity at particle[LLH] ---->> gravity at particle in ECEF"""

    gravity = - MU_EARTH * np.asarray(ECEF) / (np.linalg.norm(ECEF)**3)
    [lat, lon, height] = ECEF2LLH(ECEF)

    # grav = MU_EARTH / (np.linalg.norm(ECEF)**2)
    # [lat, lon, height] = ECEF2LLH(ECEF)
    # trans = ENU2ECEF(lon, lat)

    # gravity = np.dot(trans , np.vstack((0, 0, -grav)))

    return gravity, lat, lon, height

def EarthRadius(lat):
    """radius of the Earth as fn of latitude"""
 #    http://en.wikipedia.org/wiki/Earth_radius#Radii_with_location_dependence
    #latr=radians(lat)

    # WGS84 Defining Parameters
    a = 6378137.0  # Semi-major axis
    f = 1 / 298.257223563  # Flattening
    b = a * (1 - f)  # Semi-minor axis
    
    nominator = (a**2 * np.cos(lat))**2 + (b**2 * np.sin(lat))**2
    denominator = (a * np.cos(lat))**2 + (b * np.sin(lat))**2
    
    return np.sqrt( nominator / denominator )


#########################################################
# Trajectory maths / transforms
#########################################################
def get_zenith_and_bearing(table, segment):
    '''
    Choose the part of the trajectory to calculate: 'all', 'beg', or 'end'.
    '''
    
    if segment == 'all':
        X_beg = np.vstack((table['X_geo'][ 0], table['Y_geo'][ 0], table['Z_geo'][ 0]))
        X_end = np.vstack((table['X_geo'][-1], table['Y_geo'][-1], table['Z_geo'][-1]))
        X_mid = (X_beg + X_end) / 2.0
    
    elif segment == 'beg':
        X_beg = np.vstack((table['X_geo'][0], table['Y_geo'][0], table['Z_geo'][0]))
        X_end = np.vstack((table['X_geo'][1], table['Y_geo'][1], table['Z_geo'][1]))
        X_mid = X_beg
    
    elif segment == 'end':
        X_beg = np.vstack((table['X_geo'][-2], table['Y_geo'][-2], table['Z_geo'][-2]))
        X_end = np.vstack((table['X_geo'][-1], table['Y_geo'][-1], table['Z_geo'][-1]))
        X_mid = X_end
        
    else:
        raise NameError("Must choose between segment of either 'all', 'beg', or 'end'.")
    
    [[X],[Y],[Z]] = X_mid
    try:
        X_mid_EL = EarthLocation(x=X.value*u.m, y=Y.value*u.m, z=Z.value*u.m)
    except AttributeError:
        X_mid_EL = EarthLocation(x=X*u.m, y=Y*u.m, z=Z*u.m)
        
    X_mid_LLH = X_mid_EL.to_geodetic()
    
    lon = np.deg2rad(X_mid_LLH[0].value)
    lat = np.deg2rad(X_mid_LLH[1].value)
    
    # Rotation matrix from ENU to ECEF (geo) coordinates
    trans = ECEF2ENU(lon, lat)
    X_ENU = np.dot(trans, X_end - X_beg)
    [[E],[N],[U]] = X_ENU
    
    # Convert the cartesian to spherical coordinates
    try:
        zenith = np.rad2deg(np.arctan2(np.sqrt(E**2 + N**2), abs(U))).value * u.deg
    except AttributeError:
        zenith = np.rad2deg(np.arctan2(np.sqrt(E**2 + N**2), abs(U))) * u.deg
    
    try:
        bearing = np.rad2deg(np.arctan2(E, N)) % (360 * u.deg)
    except:
        bearing = (np.arctan2(E, N)  * u.rad) % (360 * u.deg)
    
    return zenith, bearing

def ShortestMidPoint(obs_ECEF_all, UV_ECEF_all):
    ''' Finds the point mid way between the closest section of two rays '''
    # obs_ECEF, UV_ECEF are in list form. i.e [np.vstack((x,y,z)), np.vstack(()), ]

    UV1 = UV_ECEF_all[0]; GS1 = obs_ECEF_all[0]
    UV2 = UV_ECEF_all[1]; GS2 = obs_ECEF_all[1]

    V = np.cross(UV1, UV2, axis=0) # Vector at right angles to both UV1 & UV2 
    n1 = np.cross(V, UV1, axis=0) # Normal of plane with UV1 embedded
    n2 = np.cross(V, UV2, axis=0) # Normal of plane with UV2 embedded

    d1 = n2.T.dot(GS2 - GS1) / n2.T.dot(UV1) # Distance along UV1 to closest approach
    d2 = n1.T.dot(GS1 - GS2) / n1.T.dot(UV2) # Distance along UV2 to closest approach

    x_opt = ((GS1 + UV1*d1) + (GS2 + UV2*d2)) / 2 # Average of two closest points

    return x_opt

def TotalAngSep(x, obs_ECEF, UV_ECEF, scaling=1.0):
    # obs_ECEF, UV_ECEF are in list form. i.e [np.vstack((x,y,z)), np.vstack(()), ]

    x = np.vstack((x * scaling))
    
    theta = np.zeros(len(obs_ECEF))
    for i in range(len(obs_ECEF)):
        
        obs = obs_ECEF[i]
        UV = UV_ECEF[i]
        
        theta[i] = np.arccos( (x - obs).T.dot(UV) / (norm(x - obs) * norm(UV)) )

    return norm(theta)

def angular_difference(reference_angle, target_angle):
    ''' There may be a better way, but this works '''
    diff = reference_angle - target_angle
    return np.arctan2(np.sin(diff), np.cos(diff))

def angular_difference_2d(point1, point2, input_type='uv'): #[n,2or3],[n,2or3]
    
    if input_type == 'ang':
        ''' Haversine Formula '''
        if point1.ndim == 1: point1 = point1.reshape((1,2))
        if point2.ndim == 1: point2 = point2.reshape((1,2))

        ra1 = point1[:,0]; dec1 = point1[:,1]
        ra2 = point2[:,0]; dec2 = point2[:,1]
        a = np.sin((dec1-dec2)/2)**2 + np.cos(dec1) * np.cos(dec2) * np.sin((ra1-ra2)/2)**2
        ang_diff = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a)) #[n]

    elif input_type == 'uv':
        if point1.ndim == 1: point1 = point1.reshape((1,3))
        if point2.ndim == 1: point2 = point2.reshape((1,3))

        UV1 = point1; UV2 = point2 #[n,3]
        ang_diff = np.arctan2(norm(np.cross(UV1, UV2, axis=1), axis=1), 
                np.sum(UV1*UV2, axis=1)) #[n]

    else:
        print('Not a valid input_type, soz.'); exit()

    return ang_diff #[n]

def track_errors(Pos, Vel, Obs, UV_obs, eci=True): #[3,n],[3,n],[3,n],[3,m]

    # Relative pos/vel needed to determine the rotation matrix
    Pos_rel = Pos - Obs; ground_rot = 0 * Obs
    if eci: ground_rot[2] = OMEGA_EARTH
    Vel_rel = Vel - np.cross(ground_rot, Obs, axis=0)
    
    # Estimated LOS based on model fit 
    UV_est = Pos_rel / norm(Pos_rel, axis=0)

    # Find the velocity components w.r.t. UV_est
    v_para = np.sum(Vel_rel * UV_est, axis=0) * UV_est
    v_perp = Vel_rel - v_para

    # Construct the ECI-to-body rotation matrix
    x_body = UV_est #[3,n]
    y_body = v_perp / norm(v_perp, axis=0) #[3,n]
    z_body = np.cross(x_body, y_body, axis=0) #[3,n]
    C_ECI2BODY = np.vstack((x_body.flatten('f'),
        y_body.flatten('f'), z_body.flatten('f'))) #[3,3n] or [3,3]

    # Convert UV_obs into the body frame and determine the track errors
    UV_obs_block = block_diag(*np.hsplit(UV_obs, len(Obs[0]))) #[3n,n] or [3,m]
    UV_body = C_ECI2BODY.dot(UV_obs_block) #[3,n] or [3,m]
    ATE = np.arctan2(UV_body[1], UV_body[0]) #[n] or [m]
    CTE = np.arctan2(UV_body[2], UV_body[0]) #[n] or [m]

    return ATE, CTE

def track_error_jac(Pos, Vel, Obs, z, eci=True): #[3,1],[3,1],[3,1],[3,1]
    ''' Jacobian of the track errors using central differencing '''

    # Setup the state step - h_opt according to step_size_optimiser
    dang = np.diag([1e-5, 1e-5])

    # Compute the jacobian
    te_pos = np.vstack((track_errors(Pos, Vel, Obs, ang2uv(z + dang).T, eci))).T #[2,2]
    te_neg = np.vstack((track_errors(Pos, Vel, Obs, ang2uv(z - dang).T, eci))).T #[2,2]

    step = np.tile(np.vstack(np.diag(dang)),(1,2)) #[2,2]
    jac = (te_pos - te_neg) / (2 * step)
    
    return jac #[2,2]

# ''' Spherical to Cartesian functions '''
def ang2uv(mean_ang): #[n,2]
    if mean_ang.ndim == 1: mean_ang = mean_ang.reshape((1,2))

    ra = mean_ang[:,0]; dec = mean_ang[:,1]
    mean_uv = np.vstack((np.cos(dec) * np.cos(ra),
        np.cos(dec) * np.sin(ra), np.sin(dec))).T

    return mean_uv #[n,3]

def uv2ang(mean_uv): #[n,3]
    if mean_uv.ndim == 1: mean_uv = mean_uv.reshape((1,3))

    dist = norm(mean_uv, axis=1)
    ra = np.arctan2(mean_uv[:,1], mean_uv[:,0])
    dec = np.arcsin(mean_uv[:,2] / dist)
    mean_ang = np.vstack((ra, dec)).T

    return mean_ang #[n,2]

# def ang2uv_jac(mean_ang): #[n,2]
#     if mean_ang.ndim == 1: mean_ang = mean_ang.reshape((1,2))

#     # Setup the state step - move by one arcsec
#     dang = np.deg2rad(1./3600) * np.eye(2)
#     n = len(mean_ang); DANG = np.tile(dang,(n,1))
#     MEAN_ang = np.hstack([mean_ang]*2).reshape((2*n,2)) #[2n,2]

#     # Compute the jacobian
#     uv_pos = ang2uv(MEAN_ang + DANG) #[2n,3]
#     uv_neg = ang2uv(MEAN_ang - DANG) #[2n,3]

#     step = np.tile(np.vstack(np.diag(dang)),(n,3)) #[2n,3]
#     jac = (uv_pos - uv_neg) / (2 * step)
#     jac = jac.reshape((n,2,3))
    
#     return jac #[n,2,3]

# def convert_angle_to_uv(ra, dec, ra_err, dec_err):

#     # Calculate the uv means
#     mean_ang = np.vstack((ra,dec)).T #[n,2]
#     mean_uv = ang2uv_jac(mean_ang) #[n,3]

#     # Construct the angular covariance matrix
#     cov_ang = np.zeros((len(mean_ang),2,2)) #[n,2,2]
#     cov_ang[:,0,0] = ra_err**2; cov_ang[:,1,1] = dec_err**2

#     # Determine the uv covariance matrix
#     thi = ang2uv_jac(mean_ang) #[n,2,3]
#     cov_uv = np.matmul(np.transpose(thi,(0,2,1)), \
#                 np.matmul(cov_ang, thi)) #[n,3,3]

#     ### Might need to flatten here if n=1

#     return mean_uv, cov_uv #[n,3], [n,3,3]

# def ECEF2LLH(ECEF):
#     '''
#     Converts coords Earth Centered Earth Fixed (ECEF=[rx;ry;rz])
#     into LLH=[longitude;latitude;height] coords.
#     '''
#     # WGS84 Defining Parameters      
#     a_earth = 6378137.0  # Semi-major axis
#     f_earth = 1 / 298.257223563  # Flattening
#     b_earth = a_earth * (1 - f_earth)  # Semi-minor axis
#     e_earth = np.sqrt(1 - (b_earth**2) / (a_earth**2))  # Eccentricity
#     e_prime_earth = np.sqrt((a_earth**2) / (b_earth**2) - 1)  # Second eccentricity
#     [G, M] = grav_params()

#     # Separate the variables
#     X = ECEF[0]
#     Y = ECEF[1]
#     Z = ECEF[2]

#     # Auxiliary values
#     p = np.sqrt(X**2 + Y**2)
#     theta = np.arctan2(Z * a_earth, p * b_earth)

#     # Calculate the LLH coords
#     Lat = np.arctan2(Z + (e_prime_earth**2) * b_earth * (np.sin(theta))**3,
#                      p - (e_earth**2) * a_earth * (np.cos(theta))**3)
#     Long = np.arctan2(Y, X)
#     H = p / np.cos(Lat) - a_earth / np.sqrt(1 - (e_earth**2) * (np.sin(Lat))**2)
    
#     LLH = np.vstack((Lat, Long, H))
    
#     return Lat, Long, H  #LLH
    
# def LLH2ECEF(LLH):
#     '''
#     Converts geodetic LLH=[longitude;latitude;height] coords to 
#     Earth Centered Earth Fixed (ECEF=[rx;ry;rz]) coords.
#     '''
#     # WGS84 Defining Parameters      
#     a_earth = 6378137.0  # Semi-major axis
#     f_earth = 1 / 298.257223563  # Flattening
#     b_earth = a_earth * (1 - f_earth)  # Semi-minor axis
#     e_earth = np.sqrt(1 - (b_earth**2) / (a_earth**2))  # Eccentricity
#     e_prime_earth = np.sqrt((a_earth**2) / (b_earth**2) - 1)  # Second eccentricity
#     [G, M] = grav_params()

#     # Separate the variables
#     Lat = LLH[0]
#     Long = LLH[1]
#     H = LLH[2]

#     # Calculate the ECEF coords
#     N = a_earth / np.sqrt(1 - e_earth**2 * (np.sin(Lat))**2)
#     rx = (N + H) * np.cos(Lat) * np.cos(Long)
#     ry = (N + H) * np.cos(Lat) * np.sin(Long)
#     rz = (N * (1 - e_earth**2) + H) * np.sin(Lat)

#     # Construct the ECEF vector
#     ECEF = np.vstack((rx, ry, rz))
    
#     return ECEF

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

def ENU2ECEF(lon, lat):
    """transform of ENU2ECEF
    """
    lon = float(lon); lat = float(lat)
    C_ENU2ECEF = np.array([[-np.sin(lon), -np.sin(lat) * np.cos(lon), np.cos(lat) * np.cos(lon)],
                           [ np.cos(lon), -np.sin(lat) * np.sin(lon), np.cos(lat) * np.sin(lon)],
                           [     0      ,         np.cos(lat)       ,        np.sin(lat)       ]])
    return C_ENU2ECEF

def LLH2ECEF(Pos_LLH):
    
    Pos_EL = EarthLocation(lat=Pos_LLH[0] * u.rad, lon=Pos_LLH[1] * u.rad, height=Pos_LLH[2] * u.m)
    Pos_ECEF_temp = np.vstack((Pos_EL.x.value, Pos_EL.y.value, Pos_EL.z.value))
    
    Pos_ECEF = np.full(np.shape(Pos_ECEF_temp), np.nan)
    Pos_ECEF[:,~np.isnan(Pos_ECEF_temp[2])] = Pos_ECEF_temp[:,~np.isnan(Pos_ECEF_temp[2])]
    
    return Pos_ECEF

def ECEF2LLH(Pos_ECEF):
    
    Pos_EL = EarthLocation(x=Pos_ECEF[0] * u.m, y=Pos_ECEF[1] * u.m, z=Pos_ECEF[2] * u.m)
    Pos_LLH_temp = np.vstack((Pos_EL.lat.rad, Pos_EL.lon.rad, Pos_EL.height.value))
    
    Pos_LLH = np.full(np.shape(Pos_LLH_temp), np.nan)
    Pos_LLH[:,~np.isnan(Pos_LLH_temp[2])] = Pos_LLH_temp[:,~np.isnan(Pos_LLH_temp[2])]
    
    return Pos_LLH

def ECI2ECEF_pos(Pos_ECI, t):

    T = Time(t, format='jd', scale='utc')

    dist_vect = norm(Pos_ECI, axis=0)
    ra_vect = np.arctan2(Pos_ECI[1], Pos_ECI[0])
    dec_vect = np.arcsin(Pos_ECI[2] / dist_vect)

    Pos_ECI_SC = GCRS(ra=ra_vect * u.rad, dec=dec_vect * u.rad, 
        distance=dist_vect * u.m, obstime=T)
    Pos_ECEF_SC = Pos_ECI_SC.transform_to(ITRS(obstime=T))
    Pos_ECEF = np.vstack(Pos_ECEF_SC.cartesian.xyz.value)

    return Pos_ECEF
        
def ECI2ECEF(Pos_ECI, Vel_ECI, t):

    Pos_ECEF = ECI2ECEF_pos(Pos_ECI, t)
    
    PV_ECI = Pos_ECI + Vel_ECI
    PV_ECEF = ECI2ECEF_pos(PV_ECI, t)
    Vel_ECEF = PV_ECEF - Pos_ECEF - np.cross( np.vstack((0,0,OMEGA_EARTH)), Pos_ECEF, axis=0)
    
    return Pos_ECEF, Vel_ECEF


def ECEF2ECI_pos(Pos_ECEF, t):

    T = Time(t, format='jd', scale='utc')
    
    Pos_ECEF_SC = ITRS(x=Pos_ECEF[0] * u.m, y=Pos_ECEF[1] * u.m, 
        z=Pos_ECEF[2] * u.m, obstime=T)
    Pos_ECI_SC = Pos_ECEF_SC.transform_to(GCRS(obstime=T))
    Pos_ECI = np.vstack(Pos_ECI_SC.cartesian.xyz.value)

    return Pos_ECI

def ECEF2ECI(Pos_ECEF, Vel_ECEF, t):
    
    Pos_ECI = ECEF2ECI_pos(Pos_ECEF, t)

    PV_ECEF = Pos_ECEF + Vel_ECEF + np.cross( np.vstack((0,0,OMEGA_EARTH)), Pos_ECEF, axis=0)
    PV_ECI = ECEF2ECI_pos(PV_ECEF, t)
    Vel_ECI = PV_ECI - Pos_ECI 
    
    return Pos_ECI, Vel_ECI

def ECI2TEME_pos(Pos_ECI, t):

    T = Time(t, format='jd', scale='utc')
    precessed_frame = PrecessedGeocentric(equinox=T, obstime=T)

    dist_vect = norm(Pos_ECI, axis=0)
    ra_vect = np.arctan2(Pos_ECI[1], Pos_ECI[0])
    dec_vect = np.arcsin(Pos_ECI[2] / dist_vect)

    Pos_ECI_SC = GCRS(ra=ra_vect * u.rad, dec=dec_vect * u.rad, 
        distance=dist_vect * u.m, obstime=T)
    Pos_TEME_SC = Pos_ECI_SC.transform_to(precessed_frame)
    Pos_TEME = np.vstack(Pos_TEME_SC.cartesian.xyz.value)

    return Pos_TEME

''' There is a milli-arcsec difference between ICRS and J2000, but it 
    is at least three orders of magnitude smaller than the error on 
    our best orbital prediction. Hence, it has been deemed negligable.
    See the following references for material:

    https://www.aanda.org/articles/aa/full/2004/02/aa3851/aa3851.html
    http://cdsads.u-strasbg.fr/cgi-bin/nph-bib_query?1998A&A...331L..33F
    https://www.iers.org/IERS/EN/Science/ICRS/ICRS.html
'''
# eps = np.deg2rad(np.array([-5.1, 17.2, -78.0]) / (3600 * 1000))
# R_x = np.array([[ 1,       0       ,       0       ],
#                 [ 0, np.cos(eps[0]),-np.sin(eps[0])],
#                 [ 0, np.sin(eps[0]), np.cos(eps[0])]])
# R_y = np.array([[ np.cos(eps[1]), 0, np.sin(eps[1])],
#                 [       0       , 1,       0       ],
#                 [-np.sin(eps[1]), 0, np.cos(eps[1])]])
# R_z = np.array([[ np.cos(eps[2]),-np.sin(eps[2]), 0],
#                 [ np.sin(eps[2]), np.cos(eps[2]), 0],
#                 [       0       ,       0       , 1]])
# ICRS2J2000 = R_x.dot(R_y).dot(R_z)

# Obliquity at J2000 (Chapront et al. | 2002)
# http://hpiers.obspm.fr/eop-pc/models/constants.html
e = np.deg2rad(23 + 26/60 + 21.4119/3600)
C_EQ2ECLIP = np.array([[1,     0     ,    0     ],
                       [0,  np.cos(e), np.sin(e)],
                       [0, -np.sin(e), np.cos(e)]])

def HCRS2HCI(HCRS):
    HCI = C_EQ2ECLIP.dot(HCRS)
    return HCI

def HCI2HCRS(HCI):
    HCRS = C_EQ2ECLIP.T.dot(HCI)
    return HCRS

def ECI2HCI_pos(Pos_ECI, t):

    # Position & velocity relative to the sun
    T = Time(t, format='jd', scale='utc')

    dist_vect = norm(Pos_ECI, axis=0)
    ra_vect = np.arctan2(Pos_ECI[1], Pos_ECI[0])
    dec_vect = np.arcsin(Pos_ECI[2] / dist_vect)

    Pos_ECI_SC = GCRS(ra=ra_vect * u.rad, dec=dec_vect * u.rad, 
        distance=dist_vect * u.m, obstime=T)
    Pos_HCRS = np.vstack(Pos_ECI_SC.transform_to(HCRS(obstime=T)).cartesian.xyz.value)
    Pos_HCI = HCRS2HCI(Pos_HCRS)

    return Pos_HCI

def ECI2HCI(Pos_ECI, Vel_ECI, t):

    Pos_HCI = ECI2HCI_pos(Pos_ECI, t)

    PV_ECI = Pos_ECI + Vel_ECI
    PV_HCI = ECI2HCI_pos(PV_ECI, t)
    Vel_HCI = PV_HCI - Pos_HCI + EarthVelocity(t)

    return Pos_HCI, Vel_HCI


def HCI2ECI_pos(Pos_HCI, t):
    
    # Position & velocity relative to the sun
    T = Time(t, format='jd', scale='utc')

    Pos_HCRS = HCI2HCRS(Pos_HCI)
    # TODO. HARD: had to remove it
    #Vel_HCRS_rel = HCI2HCRS(Vel_HCI - EarthVelocity(t))

    Pos_HCRS_SC = HCRS(x=Pos_HCRS[0]*u.m, y=Pos_HCRS[1]*u.m, z=Pos_HCRS[2]*u.m,
        representation='cartesian', obstime=T)
    Pos_ECI = np.vstack(Pos_HCRS_SC.transform_to(GCRS(obstime=T)).cartesian.xyz.value)

    return Pos_ECI

def HCI2ECI(Pos_HCI, Vel_HCI, t):

    Pos_ECI = HCI2ECI_pos(Pos_HCI, t)

    PV_HCI = Pos_HCI + Vel_HCI - EarthVelocity(t)
    PV_ECI = HCI2ECI_pos(PV_HCI, t)
    Vel_ECI = PV_ECI - Pos_ECI

    return Pos_ECI, Vel_ECI

def PosVel2OrbitalElements(Pos, Vel, OrbitBody, Type):
    if Pos.ndim == 1: Pos = np.vstack(Pos)
    if Vel.ndim == 1: Vel = np.vstack(Vel)

    if OrbitBody == 'Sun':
        mu = MU_SUN
    elif OrbitBody == 'Earth':
        mu = MU_EARTH
    else:
        print('Not valid OrbitBody: PosVel2OrbitalElements')
        exit()

    # Pre-calculations
    dim = np.shape(Pos)[1]
    w_hat = np.cross(Pos, Vel, axis=0) / norm(np.cross(Pos, Vel, axis=0), axis=0)
    i_hat = np.vstack((np.ones((1, dim)), np.zeros((2, dim))))
    k_hat = np.vstack((np.zeros((2, dim)), np.ones((1, dim))))
    n_hat = np.cross(k_hat, w_hat, axis=0) / norm(np.cross(k_hat, w_hat, axis=0), axis=0)
    e_vec = ((norm(Vel, axis=0)**2 - mu / norm(Pos, axis=0)) * Pos - np.diag(np.dot(Pos.T, Vel)) * Vel) / mu

    # Classical Orbital Elements
    a = mu * norm(Pos, axis=0) / (2 * mu - norm(Pos, axis=0) * norm(Vel, axis=0)**2)
    e = norm(e_vec, axis=0)
    i = np.arccos(np.diag(np.dot(w_hat.T, k_hat)))
    omega = np.arctan2(np.diag(np.dot(np.cross(n_hat, e_vec, axis=0).T, w_hat)),
                       np.diag(np.dot(n_hat.T, e_vec))) % (2 * np.pi)
    Omega = np.arctan2(np.diag(np.dot(np.cross(i_hat, n_hat, axis=0).T, k_hat)),
                       np.diag(np.dot(i_hat.T, n_hat))) % (2 * np.pi)
    theta = np.arctan2(np.diag(np.dot(np.cross(e_vec, Pos, axis=0).T, w_hat)),
                       np.diag(np.dot(e_vec.T, Pos))) % (2 * np.pi)

    if Type == 'Classical':

        COE = np.vstack((a, e, i, omega, Omega, theta))

        return COE

    elif Type == 'Equinoctial':

        # Convert to equinoctial orbital elements
        p = a * (1 - e**2)
        f = e * np.cos(omega + Omega)
        g = e * np.sin(omega + Omega)
        h = np.tan(i / 2) * np.cos(Omega)
        k = np.tan(i / 2) * np.sin(Omega)
        L = (Omega + omega + theta) % (2 * np.pi)

        EOE = np.vstack((p, f, g, h, k, L))

        return EOE

    else:
        print('Not valid Type: PosVel2OrbitalElements')
        exit()

def OrbitalElements2PosVel(OE, OrbitBody, Type):

    if OrbitBody == 'Sun':
        mu = MU_SUN
    elif OrbitBody == 'Earth':
        mu = MU_EARTH
    else:
        print('Not valid OrbitBody: OrbitalElements2PosVel')
        exit()

    if Type == 'Classical':

        # Assigning the elements
        a = OE[0]; e = OE[1]; i = OE[2]; omega = OE[3]; Omega = OE[4]; theta = OE[5]

        # Convert to equinoctial orbital elements
        p = a * (1 - e**2)
        f = e * np.cos(omega + Omega)
        g = e * np.sin(omega + Omega)
        h = np.tan(i / 2) * np.cos(Omega)
        k = np.tan(i / 2) * np.sin(Omega)
        L = Omega + omega + theta

    elif Type == 'Equinoctial':

        # Assigning the elements
        p = OE[0]; f = OE[1]; g = OE[2]; h = OE[3]; k = OE[4]; L = OE[5]

    else:
        print('Not valid Type: OrbitalElements2PosVel')
        exit()

    w = 1 + f * np.cos(L) + g * np.sin(L)
    s2 = 1 + h**2 + k**2
    alpha2 = h**2 - k**2
    r = p / w

    # Positions
    Pos = r / s2 * np.vstack((np.cos(L) + alpha2 * np.cos(L) + 2 * h * k * np.sin(L),
                              np.sin(L) - alpha2 * np.sin(L) + 2 * h * k * np.cos(L),
                              2 * (h * np.sin(L) - k * np.cos(L))))
    # Velocities
    Vel = -1 / s2 * np.sqrt(mu / p) * np.vstack((np.sin(L) + alpha2 * np.sin(L) - 2 * h * k * np.cos(L) + g - 2 * f * h * k + alpha2 * g,
                                                 -np.cos(L) + alpha2 * np.cos(L) + 2 * h * k * np.sin(L) - f + 2 * g * h * k + alpha2 * f,
                                                 -2 * (h * np.cos(L) + k * np.sin(L) + f * h + g * k)))
    return Pos, Vel

#########################################################
# Earth motion
#########################################################

def EarthPosition(t, ephem='builtin'):
    '''
    The position of Earth at time, t, w.r.t. the Sun 
    using astropy.coordinates.solar_system_ephemeris.
    '''
#    logger = logging.getLogger()
#    logger.critical('As of 1.3, astropy has changed the behavior of some functions, giving diverging answers from previous versions')
    
    T = Time(t, format='jd', scale='utc')

    # pos_bary, vel_bary = get_body_barycentric_posvel('earth', time=T)
    with solar_system_ephemeris.set(ephem):
        pos_bary = get_body_barycentric('earth', time=T)
    pos_ICRS = ICRS(pos_bary)

    hcrs_frame = HCRS(obstime=T)
    Pos_HCRS = np.vstack((SkyCoord(pos_ICRS).transform_to(hcrs_frame).
                            cartesian.xyz.to(u.meter).value))
    Pos_HCI = HCRS2HCI(Pos_HCRS)
    
    return Pos_HCI

def EarthVelocity(t, ephem='builtin'):
    '''
    The velocity of Earth at time, t, w.r.t. the Sun using central differencing.
    '''
    # TODO when astropy ends up implementing velocity transformations between frames, use that instead

    dt = 1.0 * 60 * 60  # 1 hour
    EarthPos1 = EarthPosition(t - dt / (24 * 60 * 60), ephem)  # This time is in JD
    EarthPos2 = EarthPosition(t + dt / (24 * 60 * 60), ephem)  # This time is in JD
    Vel_HCI = (EarthPos2 - EarthPos1) / (2 * dt)

    return Vel_HCI

#########################################################
# File queries
#########################################################
def find_events_folder(datadir, event_pattern="DN*"):
    """ return a list of event folders within 'datadir' that follow the event pattern given
    """
    find_comm = ['find',
                 datadir,
                 '-type', 'd',
                 '-name', event_pattern,
                 '-maxdepth', '1']
    
    list_results = subprocess.check_output(find_comm).decode().split('\n')
    
    return [it for it in list_results if it != '']

def find_file(path_to_event):
    """return the most recent trajectory folder for a given event codename
    """
   
    event_codename = os.path.basename(os.path.normpath(path_to_event))

    try:
        traj_folder = get_most_recent_traj_folder(path_to_event)
    except FileNotFoundError:
        return None

    return traj_folder

def get_most_recent_traj_folder(event_folder):
    '''
    Try to find a the most recent trajectory folder. Has to start with 'trajectory_20'
    Finds the most recent by sorting them lexicographically
    Parameters:
        event_folder: folder where to look
    Returns:
        the directory where the most recent full trajectory data is stored
    '''
    
    dir_list = os.listdir(event_folder)
    
    traj_folds = sorted([d for d in dir_list if d.startswith('trajectory_20')])
    
    if len(traj_folds) < 1:
        raise FileNotFoundError('Cannot find trajectory data for that event')
    
    return os.path.join(event_folder, traj_folds[-1])