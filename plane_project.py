from __future__ import division
from math import *
import copy, operator, random, datetime
import os, glob

# science
import numpy as np
import scipy 
from scipy import linalg, integrate, stats, interpolate
from numpy.linalg import norm
from scipy.optimize import minimize

# Astropy
from astropy.table import Table, Column, join, hstack, vstack
from astropy.time import Time, TimeDelta
from astropy import units as u
from astropy.coordinates import SkyCoord, EarthLocation, AltAz


import matplotlib as mpl

import matplotlib.pyplot as plt

working_dir = '/home/ellie/dfn_events/events_trello/DN160410_03/trajectory_20180522_ECI/Paper/'
# line_fname = '/home/ellie/dfn_events/events_trello/DN160410_03/trajectory_20180522_ECI/Paper/TOPS/30_2016-04-10_130858_DSC_0439-G_DN160410_03_2018-03-09_103726_martin_nocomment.ecsv'
# line_fname = '/home/ellie/dfn_events/events_trello/DN160410_03/trajectory_20180522_ECI/Paper/TOPS/32_2016-04-10_130858_S_DSC_0425-G_DN160410_03_2018-03-09_104707_martin_nocomment.ecsv'
line_fname = '/home/ellie/dfn_events/events_trello/DN160410_03/trajectory_20180522_ECI/Paper/ALL/27_2016-04-10_130858_DSC_5093-G_DN160410_03_2018-03-09_103401_martin_nocomment.ecsv'
line = Table.read(line_fname, format='ascii.ecsv', guess=False, delimiter=',')

points_fname1 = '/home/ellie/dfn_events/events_trello/DN160410_03/trajectory_20180522_ECI/pointwise_20180524_with_errors_run_0/27_27_13_30_27_32_13_32_30_30_32_13_PointTriangulation_ECEF.ecsv'
points1 = Table.read(points_fname1, format='ascii.ecsv', guess=False, delimiter=',')


# points_fname2 = '/home/ellie/dfn_events/events_search/DN160410_03_SA_MtBarry/3D_PF_trajectory/ECI/grav/32_2016-04-10_130858_S_DSC_0425-G_DN160410_03_2016-09-20_105334_hadry_nocomment.ecsv'
# points2 = Table.read(points_fname2, format='ascii.ecsv', guess=False, delimiter=',')



# working_dir = '/home/ellie/dfn_events/events_trello/DN151212_03/trajectory_20180524_ECI/'

# # line_fname = '/home/ellie/dfn_events/events_trello/DN151212_03/trajectory_20180524_ECI/39_2015-12-12_113628_DSC_0128-G_DN151212_03_2018-04-27_155537_martin_nocomment.ecsv'
# line_fname = '/home/ellie/dfn_events/events_trello/DN151212_03/trajectory_20180524_ECI/57_2015-12-12_113557_DSC_0182-G_DN151212_03_2016-09-15_173133_hadry_beginning_CUT_TOP.ecsv'
# line = Table.read(line_fname, format='ascii.ecsv', guess=False, delimiter=',')

# # line_fname2 = '/home/ellie/dfn_events/events_trello/DN151212_03/trajectory_20180524_ECI/57_2015-12-12_113557_DSC_0182-G_DN151212_03_2016-09-15_173133_hadry_beginning.ecsv'
# # line2 = Table.read(line_fname2, format='ascii.ecsv', guess=False, delimiter=',')

# points_fname1 = '/home/ellie/dfn_events/events_trello/DN151212_03/trajectory_20180524_ECI/pointwise_20180526_with_errors_run_0/57_26_29_39_57_29_30_PointTriangulation_ECEF.ecsv'
# points1 = Table.read(points_fname1, format='ascii.ecsv', guess=False, delimiter=',')

# points_fname2 = '/home/ellie/dfn_events/events_trello/DN151212_03/trajectory_20180524_ECI/DN151212_03_EKS_all_cams_all.csv'
# points2 = Table.read(points_fname2, format='ascii.csv', guess=False, delimiter=',')

# points_fname1_e = '/home/ellie/dfn_events/events_search/DN151212_03_SA_Birdsville_track_17s/3D_PF_trajectory/pointwise_20170808/pointwise_20170816/26_39_57_29_30_PointTriangulation_ECEF.ecsv'
# points1_end = Table.read(points_fname1_e, format='ascii.ecsv', guess=False, delimiter=',')

# points2_end = points2[points2['dash_start_end'] == 'E' ]



# working_dir = '/home/ellie/dfn_events/events_search/DN151212_03_SA_Birdsville_track_17s/3D_PF_trajectory/'

# line_fname = '/home/ellie/dfn_events/events_search/DN151212_03_SA_Birdsville_track_17s/3D_PF_trajectory/ECEF/grav/29_2015-12-12_113629_DSC_0146-G_DN151212_03_2016-09-07_110305_dfn-user_.ecsv'
# line = Table.read(line_fname, format='ascii.ecsv', guess=False, delimiter=',')

# points_fname1 = '/home/ellie/dfn_events/events_search/DN151212_03_SA_Birdsville_track_17s/3D_PF_trajectory/pointwise_20170808/pointwise_20170811/26_39_57_29_30_PointTriangulation_ECEF.ecsv'
# points1 = Table.read(points_fname1, format='ascii.ecsv', guess=False, delimiter=',')

# points_fname2 = '/home/ellie/dfn_events/events_search/DN151212_03_SA_Birdsville_track_17s/3D_PF_trajectory/ECI/grav/39_2015-12-12_113628_DSC_0128-G_DN151212_03_2016-09-07_111957_dfn-user_vrop.ecsv'
# points2 = Table.read(points_fname2, format='ascii.ecsv', guess=False, delimiter=',')

# points_fname1_e = '/home/ellie/dfn_events/events_search/DN151212_03_SA_Birdsville_track_17s/3D_PF_trajectory/pointwise_20170808/pointwise_20170816/26_39_57_29_30_PointTriangulation_ECEF.ecsv'
# points1_end = Table.read(points_fname1_e, format='ascii.ecsv', guess=False, delimiter=',')

# points2_end = points2[points2['dash_start_end'] == 'E' ]


# points_fname = '/home/ellie/dfn_events/events_search/DN151212_03_SA_Birdsville_track_17s/3D_PF_trajectory/pointwise_20170808/pointwise_20170811/26_39_57_29_30_PointTriangulation_ECEF.ecsv'
# points = Table.read(points_fname, format='ascii.ecsv', guess=False, delimiter=',')


# # check if ecsv files have times and use only those that do
# for f in all_ecsv_files:
#     if "notime" not in f:
#         filenames.append(str(f))
#     else:
#         print(f, 'does not contain timing data and will not be used')
# n_obs = len(filenames)

# data, t0, T0 = Geo_Fireball_Data(filenames, 'both', False)


# x [3,1]: is the non-straight line point that needs projecting

# UV_rad [3,1]: is the radiant unit vector along a straight line trajectory
# rad_0 [3,1]: is the first point on the straight line trajectory

# e1 [3,1]: is the x-axis of the plane that looks down the straight line
# e2 [3,1]: is the y-axis of the plane that looks down the straight line (analogous to up direction)

# t1 [scalar]: is the 2d coordinate along the e1 axis
# t2 [scalar]: is the 2d coordinate along the e2 axis
# s [scalar]: is the distance down the UV_rad axis (into the plane)

# Now, I will with the python code to transform a single x point:
rad = np.array([line['X_eci'][-1]-line['X_eci'][0],line['Y_eci'][-1]-line['Y_eci'][0],  line['Z_eci'][-1]-line['Z_eci'][0]])
UV_rad = rad/np.linalg.norm(rad)

rad_0 = np.array([line['X_eci'][0],line['Y_eci'][0],  line['Z_eci'][0]])
rad_0_UV = rad_0/np.linalg.norm(rad_0)

e1 = np.cross(UV_rad, rad_0) / np.linalg.norm(np.cross(UV_rad, rad_0))
e2 = np.cross(e1, UV_rad) / np.linalg.norm(np.cross(e1, UV_rad))

print(e1, e2)



points_all_pw = np.zeros((5, len(points1)))
# points_all_eci = np.zeros((3, len(points2)))

for i in range(len(points1)):
    X = np.array([points1['X_eci'][i], points1['Y_eci'][i], points1['Z_eci'][i]])
    X_err = np.array([sqrt(points1['cov_geo_xx'][i]), sqrt(points1['cov_geo_yy'][i]), sqrt(points1['cov_geo_zz'][i])])
    t1 = e1.dot(X - rad_0)
    t2 = e2.dot(X - rad_0)
    err_x = e1.dot(X_err)
    err_y = e2.dot(X_err)
    s = UV_rad.dot(X - rad_0) # not sure if you need this number, unless you wanna use it to colour scale
    points_all_pw[:, i] = [t1, t2, err_x, err_y, int(s)]


# for i in range(len(points2)):
#     X = np.array([points2['X_eci'][i], points2['Y_eci'][i], points2['Z_eci'][i]])
#     t1 = e1.dot(X - rad_0)
#     t2 = e2.dot(X - rad_0)
#     s = UV_rad.dot(X - rad_0) # not sure if you need this number, unless you wanna use it to colour scale
#     points_all_eci[:, i] = [t1, t2, s]
    
# data_out = Table(names=('x', 'y', 'd'), data= points_all.T)


# points_all_pw = np.zeros((3, len(points1_end)))
# points_all_eci = np.zeros((3, len(points2_end)))

# for i in range(len(points1_end)):
#     X = np.array([points1_end['X_eci'][i], points1_end['Y_eci'][i], points1_end['Z_eci'][i]])
#     t1 = e1.dot(X - rad_0)
#     t2 = e2.dot(X - rad_0)
#     s = UV_rad.dot(X - rad_0) # not sure if you need this number, unless you wanna use it to colour scale
#     points_all_pw[:, i] = [t1, t2, s]

# for i in range(len(points2_end)):
#     X = np.array([points2_end['X_eci'][i], points2_end['Y_eci'][i], points2_end['Z_eci'][i]])
#     t1 = e1.dot(X - rad_0)
#     t2 = e2.dot(X - rad_0)
#     s = UV_rad.dot(X - rad_0) # not sure if you need this number, unless you wanna use it to colour scale
#     points_all_eci[:, i] = [t1, t2, s]
    

f1 = plt.figure(figsize=(16, 9))

# plt.scatter(points_all_pw[0],points_all_pw[1], c=points_all_pw[2], linestyle= '-', marker='o')
# plt.scatter(points_all_eci[0],points_all_eci[1], c=points_all_eci[2], linestyle= '-', marker='o')
# plt.errorbar(points_all_pw[0],points_all_pw[1], c=points_all_pw[2], linestyle= '-', marker='o',
                 # yerr=(points1['err_plus_altitude'] + table['err_plus_azimuth'])*3.14159/ (2.0 * 180) * table[range_col])
# plt.errorbar(points_all_pw[0],points_all_pw[1] , linestyle=None, marker='o',
#                  xerr = points_all_pw[2], yerr=points_all_pw[3])
plt.scatter(points_all_pw[0],points_all_pw[1], c=-points_all_pw[4],  marker='o', zorder=2)
plt.errorbar(points_all_pw[0],points_all_pw[1] , c='#808080', fmt=None, marker=None, zorder=1, 
                 xerr = points_all_pw[2], yerr=points_all_pw[3])
plt.scatter(0, 0, c='red')
plt.xlabel('t1')
plt.ylabel('t2')
plt.show()


data_out_pw = Table(names=('x', 'y', 'x_err', 'y_err', 'd'), data= points_all_pw.T)
data_out_pw.add_column(points1['height'])
data_out_pw.add_column(points1['time'])
data_out_pw.add_column(points1['datetime'])

# data_out_eci = Table(names=('x', 'y', 'd'), data= points_all_eci.T)
all_eci_errrs = np.zeros((5, len(line)))
all_eci_errrs[2, :] = (line['err_plus_altitude'] + line['err_plus_azimuth'])*3.14159/ (2.0 * 180) * line['range']
data_out_ecef = Table(names=('x', 'y', 'x_err', 'y_err', 'd'), data= all_eci_errrs.T)

data_out_pw['x_roll'] = np.nan
data_out_pw['y_roll'] = np.nan
data_out_pw['x_roll'][1:-1] = (data_out_pw['x'][:-2] + data_out_pw['x'][1:-1] + data_out_pw['x'][2:])/3
data_out_pw['y_roll'][1:-1] = (data_out_pw['y'][:-2] + data_out_pw['y'][1:-1] + data_out_pw['y'][2:])/3
# data_out_pw['x_roll'][2:-2] = (data_out_pw['x'][:-4] + data_out_pw['x'][1:-3] + data_out_pw['x'][2:-2] + data_out_pw['x'][3:-1] + data_out_pw['x'][4:])/5
# data_out_pw['y_roll'][2:-2] = (data_out_pw['y'][:-4] + data_out_pw['y'][1:-3] + data_out_pw['y'][2:-2] + data_out_pw['x'][3:-1] + data_out_pw['x'][4:])/5

name_pw= os.path.join(working_dir ,'pw_projected.csv')
# name_eci= os.path.join(working_dir ,'39_eci_projected.csv')
name_ecef= os.path.join(working_dir ,'eci_projected.csv')

data_out_pw.write(name_pw, format='ascii.csv', delimiter=',')
print('{} has been written'.format(name_pw))
# data_out_eci.write(name_eci, format='ascii.csv', delimiter=',')
data_out_ecef.write(name_ecef, format='ascii.csv', delimiter=',')

print('{} has been written'.format(name_ecef))

