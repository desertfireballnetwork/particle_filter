# -*- coding: utf-8 -*-
"""
Created on Mon Nov 21 09:42:11 2016

@author: trent
"""
import argparse
import os
import numpy as np
from numpy.linalg import norm
import astropy.units as u 
from astropy.coordinates import EarthLocation
from astropy.table import Table, vstack, Column
from scipy.optimize import minimize
from astropy.time import Time
from trajectory_utilities import LLH2ECEF,ShortestMidPoint, \
        TotalAngSep, ECEF2LLH, ENU2ECEF, ECEF2ECI
from CSV2KML import Path, Points

def PointTriangulation(data, directory=None, mc=1):
    
    # Find the times with multiple data points
    [dt_vals, dt_counts] = np.unique(data['time'], return_counts=True)
    dt_vals_plus = dt_vals[dt_counts > 1]
    
    # Triangulate all paired points
    x = np.zeros((3, len(dt_vals_plus)))
    cov = np.zeros((3, 3, len(dt_vals_plus)))

    for k, dt_val in enumerate(dt_vals_plus):
        
        # Get all the observation data
        obs_ECEF_all = []
        for i in range(dt_counts[dt_vals == dt_val][0]):

            Times = np.array(data['time'])

            obs_lat = data['obs_lat'][Times == dt_val][i]
            obs_lon = data['obs_lon'][Times == dt_val][i]
            obs_hei = data['obs_hei'][Times == dt_val][i]
            
            obs_LLH = np.vstack((obs_lat, obs_lon, obs_hei))
            obs_ECEF = LLH2ECEF(obs_LLH)

            obs_ECEF_all.append( obs_ECEF )
        
        x_particles = np.zeros((3,mc))
        for particle in range(mc):

            # Calculate each line of sight in the paired points
            UV_ECEF_all = []
            for i in range(dt_counts[dt_vals == dt_val][0]):

                Times = np.array(data['time'])

                # Extract some raw data
                alt = np.deg2rad(data['altitude'][Times == dt_val][i])
                azi = np.deg2rad(data['azimuth'][Times == dt_val][i])
                
                if mc != 1:
                    alt_err = np.deg2rad(data['err_plus_altitude'][Times == dt_val][i])
                    azi_err = np.deg2rad(data['err_plus_azimuth'][Times == dt_val][i])
                    alt += np.random.normal(0, alt_err)
                    azi += np.random.normal(0, azi_err)

                # Convert from spherical to cartesian 
                UV_ENU = np.vstack((np.cos(alt) * np.sin(azi),
                                    np.cos(alt) * np.cos(azi),
                                    np.sin(alt)))
                                    
                # Convert from ENU to ECEF coordinates
                UV_ECEF = ENU2ECEF(data['obs_lon'][Times == dt_val][i], 
                    data['obs_lat'][Times == dt_val][i]).dot(UV_ENU)
                
                UV_ECEF_all.append( UV_ECEF )
                
            # Calculate the best position between the first two lines
            x_est = ShortestMidPoint(obs_ECEF_all, UV_ECEF_all)
            
            # Minimizing angular separation between all the lines
            result = minimize(TotalAngSep, x_est, 
                args=(obs_ECEF_all, UV_ECEF_all), method='Nelder-Mead')

            x_particles[:,particle] = result.x
        
        if mc > 1:
            PointsTable = Table()
            PointsTable['datetime'] = [data['datetime'][Times == dt_val][0]]*mc
            PointsTable['X_geo'] = x_particles[0]
            PointsTable['Y_geo'] = x_particles[1]
            PointsTable['Z_geo'] = x_particles[2]
            partices_LLH = ECEF2LLH(x_particles)
            PointsTable['latitude'] = np.rad2deg(partices_LLH[0])
            PointsTable['longitude'] = np.rad2deg(partices_LLH[1])
            PointsTable['height'] = partices_LLH[2]
            PointsFile = os.path.join(directory,'{:.3f}_particles.ecsv'.format(dt_val))
            PointsTable.write(PointsFile, format='ascii.ecsv', delimiter=',')
            print('\Points file written to: ' + PointsFile)
            Points(PointsFile)
            
        x[:,k] = np.mean(x_particles, axis=1)
        cov[:,:,k] = np.cov(x_particles)

        print('\nTime step: {0:.2f} sec'.format(dt_val))
        # print('Normalised Residuals Before: {0:8.4f}'.format(
        #     np.rad2deg(TotalAngSep(x_est, obs_ECEF_all, UV_ECEF_all))*3600))
        # print('Normalised Residuals After:  {0:8.4f}'.format(
        #     np.rad2deg(result.fun)*3600))

    if mc == 1:
        return x, dt_vals_plus, dt_counts[dt_counts > 1]
    else:
        return x, dt_vals_plus, dt_counts[dt_counts > 1], cov
        
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Pointwise Triangulate meteors')
    parser.add_argument("-d", "--inputdirectory", type=str,
            help="input directory for images with extension .ecsv" )
    parser.add_argument("-err", "--error_analysis", action="store_true", default=False,
                help="use this option if you want to analyse errors")
    parser.add_argument("-s","--pse",type=str, default='B',
                help="use start or end points? Type S or E or B for both. Default uses both")
    
    args = parser.parse_args()
    errors = args.error_analysis
    if args.inputdirectory and os.path.isdir(args.inputdirectory):
        directory = args.inputdirectory
    else:
        print( 'directory problem')
        exit(1)
    
    if args.pse == 'S' or args.pse == 's':
        pse = 'S' 
    elif args.pse == 'E' or args.pse == 'e':
        pse = 'E'
    elif args.pse == 'B' or args.pse == 'b':
        pse = 'both'
    else:
        print('-s input invalid, running with ends')
        pse = 'E'  

    Files = [os.path.join( directory, a) for a in os.listdir( directory)
             if a.endswith('.ecsv') ]

    # Create the point-wise triangulation file
    import datetime; date_str = datetime.datetime.now().strftime('%Y%m%d')
    PointDir = os.path.join(directory, 'pointwise_'+date_str)
    if errors: PointDir += '_with_errors'
    PointDir += '_run_0'; i = 1
    while os.path.isdir(PointDir): # Make sure the folder name is unique
        PointDir = '_'.join(PointDir.split('_')[:-1])+'_'+str(i); i += 1
    os.mkdir(PointDir)

            
    # Files = ['/home/trent/dfn_events/events_search/DN151212_03_SA_Birdsville_track_17s/3D_PF_trajectory/29_2015-12-12_113629_DSC_0146-G_DN151212_03_2016-09-07_110305_dfn-user_.ecsv',
             # '/home/trent/dfn_events/events_search/DN151212_03_SA_Birdsville_track_17s/3D_PF_trajectory/39_2015-12-12_113628_DSC_0128-G_DN151212_03_2016-09-07_111957_dfn-user_.ecsv',
             # '/home/trent/dfn_events/events_search/DN151212_03_SA_Birdsville_track_17s/3D_PF_trajectory/57_2015-12-12_113627_DSC_0183-G_DN151212_03_2016-09-07_105505_dfn-user_.ecsv',
             # '/home/trent/dfn_events/events_search/DN151212_03_SA_Birdsville_track_17s/3D_PF_trajectory/26_2015-12-12_113628_S_DSC_2511-G_DN151212_03_2016-09-09_115555_hadry_ending.ecsv']
    
    Data = []
    for file in Files:
        print(file)

        raw_data = Table.read( file, format='ascii.ecsv', guess=False, delimiter=',')
        if pse != 'both':
            data = raw_data[raw_data['dash_start_end'] == pse ]
        else:
            data = raw_data

        obs_lat = np.deg2rad(data.meta['obs_latitude'])
        obs_lon = np.deg2rad(data.meta['obs_longitude'])
        obs_hei = data.meta['obs_elevation']
        
        # # Extract some raw data
        # alt = np.deg2rad(data['altitude'])
        # azi = np.deg2rad(data['azimuth'])
        
        # # Convert from spherical to cartesian 
        # UV_ENU = np.vstack((np.cos(alt) * np.sin(azi),
        #                     np.cos(alt) * np.cos(azi),
        #                     np.sin(alt)))
                            
        # # Convert from ENU to ECEF coordinates
        # UV_ECEF = ENU2ECEF(obs_lon, obs_lat).dot(UV_ENU)
        
        # data['UV_i'] = UV_ECEF[0]
        # data['UV_j'] = UV_ECEF[1]
        # data['UV_k'] = UV_ECEF[2]

        data['obs_lat'] = obs_lat
        data['obs_lon'] = obs_lon
        data['obs_hei'] = obs_hei
        
        Data.append(data)
    
    # Stack the data packets 
    AllData = vstack(Data, metadata_conflicts='silent')
    
    # Sort the data in time
    AllData.sort('datetime')
    
    t0 = Time(AllData['datetime'][0], format='isot',scale='utc')
    AllData['time'] = np.round(((Time(AllData['datetime'], format='isot',scale='utc') - t0).value * 24*60*60), 2)
    
    if errors:
        mc = 1000 # Number of particles
        ParticleFolder = os.path.join(PointDir,str(mc)+'_particles_per_timestep')
        os.mkdir(ParticleFolder)

        [Pos_ECEF, t_rel, N_cams, Cov_ECEF] = PointTriangulation(AllData, ParticleFolder, mc)
    else:
        [Pos_ECEF, t_rel, N_cams] = PointTriangulation(AllData)

    # Coordinate transforms
    T_jd = t0.jd + t_rel / (24*60*60)
    Pos_LLH = ECEF2LLH(Pos_ECEF)
    Pos_ECI = ECEF2ECI(Pos_ECEF, Pos_ECEF, T_jd)[0]

    # Raw velocity calculations
    vel_eci = norm(Pos_ECI[:,1:] - Pos_ECI[:,:-1], axis=0) / (t_rel[1:]- t_rel[:-1])
    vel_geo = norm(Pos_ECEF[:,1:] - Pos_ECEF[:,:-1], axis=0) / (t_rel[1:]- t_rel[:-1])
    gamma = np.arcsin((Pos_LLH[2][1:] - Pos_LLH[2][:-1]) / (vel_geo * (t_rel[1:]-t_rel[:-1])))
    vel_eci = np.hstack((vel_eci, np.nan)); vel_geo = np.hstack((vel_geo, np.nan)); gamma = np.hstack((gamma, np.nan))

    t_isot_col = Column(name='datetime', data=Time(T_jd, format='jd', scale='utc').isot)
    t_rel_col = Column(name='time', data=t_rel*u.second)
    n_cams_col = Column(name='N_cams', data=N_cams)
    x_eci_col = Column(name='X_eci', data=Pos_ECI[0]*u.m)
    y_eci_col = Column(name='Y_eci', data=Pos_ECI[1]*u.m)
    z_eci_col = Column(name='Z_eci', data=Pos_ECI[2]*u.m)
    x_geo_col = Column(name='X_geo', data=Pos_ECEF[0]*u.m)
    y_geo_col = Column(name='Y_geo', data=Pos_ECEF[1]*u.m)
    z_geo_col = Column(name='Z_geo', data=Pos_ECEF[2]*u.m)
    lat_col = Column(name='latitude', data=np.rad2deg(Pos_LLH[0])*u.deg)
    lon_col = Column(name='longitude', data=np.rad2deg(Pos_LLH[1])*u.deg)
    hei_col = Column(name='height', data=Pos_LLH[2]*u.m)
    v_eci_col = Column(name='vel_eci', data=vel_eci*u.m/u.second)
    v_geo_col = Column(name='vel_geo', data=vel_geo*u.m/u.second)
    gam_col = Column(name='gamma', data=np.rad2deg(gamma)*u.deg)

    cam_numbers = [os.path.basename(file).split('_')[0] for file in Files]
    ofile = os.path.join(PointDir, '_'.join(cam_numbers) + '_PointTriangulation_ECEF.ecsv')
    TriTable = Table( [t_isot_col, t_rel_col, n_cams_col, x_eci_col, y_eci_col, z_eci_col, 
                x_geo_col, y_geo_col, z_geo_col, lat_col, lon_col, hei_col, 
                v_eci_col, v_geo_col, gam_col], meta={'t0': str(AllData['datetime'][0])} )
    
    if errors:
        TriTable['cov_geo_xx'] = Cov_ECEF[0,0]*u.m*u.m
        TriTable['cov_geo_yy'] = Cov_ECEF[1,1]*u.m*u.m
        TriTable['cov_geo_zz'] = Cov_ECEF[2,2]*u.m*u.m
        TriTable['cov_geo_xy'] = Cov_ECEF[0,1]*u.m*u.m
        TriTable['cov_geo_xz'] = Cov_ECEF[0,2]*u.m*u.m
        TriTable['cov_geo_yz'] = Cov_ECEF[1,2]*u.m*u.m
        TriTable['av_err'] = np.sqrt(Cov_ECEF[0,0]+Cov_ECEF[1,1]+Cov_ECEF[2,2])*u.m

    TriTable.write( ofile, format='ascii.ecsv', delimiter=',')

    print('Output has been written to: ' + ofile)

    # Create the KML's
    Points(ofile, colour='ff1400ff') # red points
    Path(ofile)

