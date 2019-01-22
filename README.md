# Particle filter for estimating fireball trajectories
Uses particle filter methodology outlined in
[Arulampalam et al., 2002](https://ieeexplore.ieee.org/abstract/document/978374/), and is detailed in 
[Sansom et al. 2019, 3D Meteoroid Trajectories, ICARUS, Volume 321, Pages 388-406](https://doi.org/10.1016/j.icarus.2018.09.026) (see also [arXiv](https://arxiv.org/abs/1802.02697) preprint).

Testing of resampling method using logarithmis weights is still ongoing, as is the method of combining positional and luminous weightings. 

As this software is in development, please consider contacting me to help determine the best run options for your data. 
I am also happy to help with interpretation and visualisation of results, as well as helping formatting input table data for compatability.


## Requirements:
- Python 3
- Astropy 2.0+
- MPI4py
- seaborn

## Running the program:
This code ip paralellised using MPI4py. 
Run in commandline as 

`$ mpirun -n <#processes> python main_MPI.py <userinputs>`

There are three run options depending on combination of data 
available. 

- 1D particle filter:            

1D analysis on a single, pre-triangulated trajectory file with X_geo, Y_geo and Z_geo information.

- 3D particle filter, cartesian: 

3D analysis on single or multiple pre-triangulated files with X_geo, Y_geo and Z_geo information.

- 3D particle filter, rays:      

3D analysis on calibrated astrometric observations (altitude and azimuth data required)




    Inputs: 
        required:
        -i --dimension: select from (1) (2) or (3) described 
             above.
        -d --inputdirectory: input directory of folder 
             containing files with extension .ecsv
        -p --numparts: number of particles to run, testing run 
             100; quick run try 1000; good run try 10,000+
        
        optional:
        -m --mass_option: Specify how you would like entry masses 
             to be initiated. (1) calculated using ballistic 
             coefficient in metadata; (2) using random logarithmic 
             distribution; (3) random uniform distribution. 
             Default is 3
        -s --pse: specify if you would like to use exclusively 
             start (S) or end (E) points, or use both (B). 
             Default uses ends
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
       
      
## Visualisation
We reccomend the use of TOPCAT table processing software for result visualisation. 
TOPCAT is an open source library for manipulating large tabular data. Please give credit by citing:

[Taylor, TOPCAT & STILTS, Astronomical Data Analysis Software and Systems XIV, 347, 2005](http://adsabs.harvard.edu/full/2005ASPC..347...29T).


## Credit

If you use this particle filter tool for your research please give credit by citing:

[Sansom et al., 3D Meteoroid Trajectories, ICARUS, submitted](https://arxiv.org/abs/1802.02697)


