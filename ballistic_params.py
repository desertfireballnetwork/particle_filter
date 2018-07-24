# import modules

# general
import configparser, sys, os, argparse, glob
import logging

# science
import numpy as np
import scipy 
from scipy.optimize import minimize, basinhopping

# Astropy
from astropy.table import Table, Column, join, hstack, vstack, QTable
from astropy.time import Time, TimeDelta
import astropy.units as u
from astropy.units.quantity import Quantity
from astropy.coordinates import EarthLocation

#plotting
import matplotlib.pyplot as plt
import itertools
import seaborn as sns

from scipy.interpolate import interp1d

#local
import dfn_utils
import trajectory_utilities as tu

class RandomDisplacementBounds(object):
    """random displacement with bounds"""
    def __init__(self, xmin, xmax, stepsize=0.00001):
        self.xmin = xmin
        self.xmax = xmax
        self.stepsize = stepsize

    def __call__(self, x):
        """take a random step but ensure the new position is within the bounds"""
        while True:
            # this could be done in a much more clever way, but it will work for example purposes
            xnew = x + np.random.uniform(-self.stepsize, self.stepsize, np.shape(x))
            if np.all(xnew < self.xmax) and np.all(xnew > self.xmin):
                break
        return xnew

def Triang_to_alpha_beta(input_data, gamma, v0_stackedmeta=0., plot=False):
    """
    Calculate ballistic parameters
    Updates input table metadata with calculated values
    Parameters:
        data: input astropy table
        gamma: ??
    Returns:
        dictionary of results
    """

    logger = logging.getLogger('trajectory')
    input_data.sort('datetime')
    
    masked_data = Table(input_data[~np.isnan(input_data['kang_vels'])])

    alt = np.asarray(masked_data['height'])
    vel = np.asarray(masked_data['kang_vels'])

    #Vels
    

    if v0_stackedmeta >100.:
        # print('3', v0_stackedmeta)
        v0 = v0_stackedmeta
    elif 'EKS_initial_velocity_all_cam' in masked_data.meta:
        # print('2', masked_data.meta['EKS_initial_velocity_all_cam'])
        v0 = float(masked_data.meta['EKS_initial_velocity_all_cam'])
    elif 'D_DT_EKS' in masked_data.colnames:
        # print('1', masked_data['D_DT_EKS'])
        v0 = masked_data['D_DT_EKS'][0]
    else:
        # print('4', vel[:5])
        v0 = np.median(vel[:5])


    if isinstance(v0, Quantity):
        v0 = v0.value
    # print(alt, vel, v0)
    # i = 0
    # n = len(vel)
    # while  i < n:
    #     if isinstance(vel[i], Quantity):
    #         test = vel[i].value
    #     else: test = vel[i]
    #     if vel[i] != vel[i] or test > 1.7 * v0:
    #         vel=np.delete(vel, i)
    #         alt=np.delete(alt, i) 
    #         n -=1
    #     else:
    #         i+=1

    if isinstance((vel/v0), Quantity):
        Vvalues = (vel/v0).data
        v0 = v0.value
    else:
        Vvalues = vel/v0      #creates a matrix of V/Ve to give a dimensionless parameter for velocity


    # Height
    h0 = 7160.
    Yvalues = alt/h0  #        %creates a matrix of h/h0 to give a dimensionless parameter for altitude

    Gparams= Q4_min(Vvalues, Yvalues)

    alpha = Gparams[0]
    beta = Gparams[1]
    sea_level_rho = 1.29
    cd = 1.#3
    A = [1.21, 1.3, 1.55]
    m_rho = [2700, 3500, 7000]
    sin_gamma = np.sin(gamma)
    mu = 2./3.

    me_sphere = pow(0.5 *  cd * 1.29 * 7160 * A[0] / (alpha * sin_gamma * m_rho[1]**(2/3.0)), 3.0)
    me_brick = pow(0.5 *  cd * 1.29 * 7160 * A[2] / (alpha * sin_gamma * m_rho[1]**(2/3.0)), 3.0)
            
    mass = []
    for i in A:
        mass.append([np.exp(-beta/(1-mu) * (1-Vvalues[-1]**2)) * pow(0.5 * cd * sea_level_rho * h0 * i / (pow(j, 2/3.) * sin_gamma * alpha), 3) for j in m_rho])
    #print('\nAlpha parameter:', alpha, '\nBeta parameter:', beta, '\npossible mass:', mass)
    logger.info('entry velocity: {3:.2f} - slope {4:2f} - Alpha: {0:.2f} - Beta parameter: {1:.2f} - Possible mass (@ 3500 kg/m3 - 1.3 drag): {2:.2f} kg'.format(alpha, beta, mass[1][1], v0, float(np.rad2deg(gamma))))
    
    input_data.meta['ballistic_alpha'] = float(alpha)
    input_data.meta['ballistic_beta'] = float(beta)
    # mass[1][1] corresponds to density 3500 and shape 1.3
    input_data.meta['ballistic_entry_mass'] = float(mass[1][1])
    input_data.meta['ballistic_reference_velocity'] = float(v0)
    input_data.meta['slope'] = float(np.rad2deg(gamma))
    
    # Eq 6. in Gritsevich 2007
    input_data.meta['ballistic_final_mass'] = mass[0][1]#np.exp(-input_data.meta['ballistic_beta'] * ((1-Vvalues[-1])/(1.-mu)))
    
    try:
        vel_table_update(input_data, v0, h0, alpha, beta)
    except ValueError:
        pass

    return {'Vvalues':Vvalues, 'Yvalues':Yvalues, 'alpha':alpha, 'beta':beta, 'mass':mass, 'A':A, 'm_rho':m_rho, 'reference_velocity': float(v0), 'gamma':float(np.rad2deg(gamma)), 'me_sphere': me_sphere, 'me_brick': me_brick, 'mf_sphere': mass[0][1], 'mf_brick':  mass[2][1]}


def vel_table_update(table, v_init, h0, alpha, beta, all_cams=False):
    """
    Interpolate Gritsevich curve
    """
    fit_min_vel = 1000./v_init
    vels = np.arange(fit_min_vel,0.9999999, 0.001)   

    y = np.log(alpha) +beta - np.log((scipy.special.expi(beta) - scipy.special.expi(beta * vels**2)) /2)

    vels_interp = interp1d(y, vels, fill_value='extrapolate')
    # print('this converts', v_init)
    if all_cams:
        table['D_DT_fitted_all'] = vels_interp(np.ma.filled(table['height']) / h0)*v_init
    else:
        table['D_DT_fitted'] = vels_interp(np.ma.filled(table['height']) / h0)*v_init

    #table.show_in_browser()




def plot_all(output_params_collection, event_codename='fireball', wdir='/tmp/', kwargs={'trajectory_segment':'all'}):
    """
    Plot a list of ballistic parameters runs
    """
    
    # initiate color palette
    palette = itertools.cycle(sns.color_palette())
    
    plt.close()

    for key in output_params_collection:
        plt.figure(1)

        res = output_params_collection[key]

        # new color for each set of observations
        color = next(palette)
    
        alpha = res['alpha']
        beta = res['beta']
        mass = res['mass']
        A = res['A']
        m_rho = res['m_rho']
        Yvalues = res['Yvalues']
        Vvalues = res['Vvalues']
        
        x = np.arange(0,1, 0.00005);                                                                                     #create a matrix of x values
        fun = lambda x:np.log(alpha) + beta - np.log((scipy.special.expi(beta) - scipy.special.expi(beta* x**2) )/2);           
                                                                                                        #(obtained from Q4 minimisation)
        y = [fun(i) for i in x]
        
        # Handle datetime axis
        
    
        if res['telescope'] != 'all':
            extra_text = '\nM_sp @ {0}, M0={1:.2f} >{2:.2f} kg \nM_br @ {0}, M0={3:.2f}>{4:.2f} kg'.format(m_rho[1], res['me_sphere'], res['mf_sphere'], res['me_brick'], res['mf_brick'])

        
            plt.scatter(Vvalues, Yvalues, color=color,
                        marker='x', label=None)
                        #label=data.meta['telescope'] + " " + data.meta['location'])

            plt.plot(x, y, color=color, 
                        label='{0: <10} : {1} {2:.3f} {3} {4:.3f}'.format(res['telescope'], r'$\alpha$ = ', alpha, r'$\beta$ = ', beta) + extra_text)
                        # label='{0} {1:.3f} {2} {3:.3f}'.format(r'$\alpha$ = ', alpha, r'$\beta$ = ', beta))
                        #label='{0} {1} {2:.3f} {3:.3f} {4:.4f}'.format(data.meta['telescope'], data.meta['location'], alpha, beta, mass))

        else:
            extra_text = '\nV0 used {7:.3f}, slope {5}{6:.1f}\nM_sp @ {0}, M0={1:.2f} >{2:.2f} kg \nM_br @ {0}, M0={3:.2f}>{4:.2f} kg'.format(m_rho[1], res['me_sphere'], res['mf_sphere'], res['me_brick'], res['mf_brick'], r'$\gamma$ = ', res['gamma'], res['reference_velocity'])

            plt.plot(x, y, color='k', 
                        label='{0: <10} : {1} {2:.3f} {3} {4:.3f}'.format(res['telescope'], r'$\alpha$ = ', alpha, r'$\beta$ = ', beta) + extra_text)
        
            plt.figure(2)
            plt.scatter(Vvalues, Yvalues, color='b',
                        marker='x', label=None)
                        #label=data.meta['telescope'] + " " + data.meta['location'])
            plt.plot(x, y, color='k', 
                        label='{0: <10} : {1} {2:.3f} {3} {4:.3f}'.format(res['telescope'], r'$\alpha$ = ', alpha, r'$\beta$ = ', beta) + extra_text)
        

            plt.title(event_codename + " - Ballistic Alpha-Beta plot - stacked")
    
            plt.xlabel("Normalised velocity")
            plt.ylabel("Normalised altitude")
            plt.legend(frameon=True, loc='best', fancybox=True, framealpha=0.5, fontsize='xx-small')

            fname = os.path.join(wdir, event_codename + "_alpha_beta_consistency_check_stacked_" + kwargs['trajectory_segment'] + ".png")
            #while os.path.isfile(fname):
                #fname = fname.split('.')[0] + '_alt.png'
            #plt.savefig(fname)
            plt.savefig(fname, dpi=150)
            plt.close()
        
        #plt.title(event_codename + " - Alpha-Beta Qc plot")
        #textstr = '$\mu=%.2f$\n$\mathrm{median}=%.2f$\n$\sigma=%.2f$'%(mu, median, sigma)
        
        # Plot bars and create text labels for the table
        # cell_text = []
        # for i in range(len(mass[0])):
            # cell_text.append(['%1.3f' % x for x in mass[i]])

        #the_table = plt.table(cellText=cell_text, TODO FIXME
                            #rowLabels=A,
                            #colLabels=m_rho,
                            #loc='top')

        #plt.subplots_adjust(left=0.2, bottom=0.2) TODO FIXME
        #plt.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=14,verticalalignment='top', bbox=props)
        
    plt.title(event_codename + " - Ballistic Alpha-Beta plot")
    
    plt.xlabel("Normalised velocity")
    plt.ylabel("Normalised altitude")
    plt.legend(frameon=True, loc='best', fancybox=True, framealpha=0.5, fontsize='xx-small')

    fname = os.path.join(wdir, event_codename + "_alpha_beta_consistency_check_" + kwargs['trajectory_segment'] + ".png")
    #while os.path.isfile(fname):
        #fname = fname.split('.')[0] + '_alt.png'
    #plt.savefig(fname)
    plt.savefig(fname, dpi=150)
    
def plot_one(output_params_collection, event_codename='fireball', wdir='/tmp/', kwargs={}):
    """
    Plot a list of ballistic parameters runs
    """
    
    # initiate color palette
    palette = itertools.cycle(sns.color_palette())
    
    plt.close()
    plt.figure()
    
    res = output_params_collection[0]
    # new color for each set of observations

    alpha = res['alpha']
    beta = res['beta']
    mass = res['mass']
    A = res['A']
    m_rho = res['m_rho']
    Yvalues = res['Yvalues']
    Vvalues = res['Vvalues']
    slope = res['slope']

    
    x = np.arange(0,1, 0.00005);                                                                                     #create a matrix of x values
    fun = lambda x:np.log(alpha) + beta - np.log((scipy.special.expi(beta) - scipy.special.expi(beta* x**2) )/2);           
                                                                                                    #(obtained from Q4 minimisation)
    y = [fun(i) for i in x]
    
    # Handle datetime axis
    plt.scatter(Vvalues, Yvalues, 
                marker='x', label=None)
                #label=data.meta['telescope'] + " " + data.meta['location'])
    
    plt.plot(x, y, 
                label='{0} {1:.3f} {2} {3:.3f} {4} {5:.3f} \n V0 used {10:.3f} \n Possible sphere mass (@ 3500 kg/m3, M0={8:.2f}): >{6:.2f} kg \n Possible brick mass (@ 3500 kg/m3, M0={9:.2f}): >{7:.2f} kg'.format(r'$\alpha$ = ', alpha, r'$\beta$ = ', beta, r'$\gamma$ = ', np.rad2deg(gamma), mass[0][1], mass[2][1], me_sphere, me_brick, v0))
             # label='{0} {1:.3f} {2} {3:.3f}'.format(r'$\alpha$ = ', alpha, r'$\beta$ = ', beta))
                #label='{0} {1} {2:.3f} {3:.3f} {4:.4f}'.format(data.meta['telescope'], data.meta['location'], alpha, beta, mass))


    plt.title(event_codename + " - Ballistic Alpha-Beta plot")
    
    plt.xlabel("Normalised velocity")
    plt.ylabel("Normalised altitude")
    plt.legend(frameon=True, loc='best', fancybox=True, framealpha=0.5, fontsize='xx-small')

    fname = os.path.join(wdir, event_codename + "_alpha_beta_consistency_check_.png")
    #while os.path.isfile(fname):
        #fname = fname.split('.')[0] + '_alt.png'
    #plt.savefig(fname)
    plt.savefig(fname, dpi=150)

def Q4_min(Vvalues, Yvalues):
    """ initiates and calls the Q4 minimisation given in Gritsevich 2007 -
        'Validity of the photometric formula for estimating the mass of a fireball projectile'
    """
    params = np.vstack((Vvalues, Yvalues))

    # x0 = [100,2]
    # bnds = ((0.001, 1000), (0.001, 1000))
    # res = minimize(min_fun, x0, args=(Vvalues, Yvalues),bounds=bnds)
    b0 = 0.001
    a0 = np.exp(Yvalues[-1])/(2. * b0)
    x0 = [a0, b0]
    xmin = [0.0001, 0.0001]
    xmax = [10000., 10.]

    # rewrite the bounds in the way required by L-BFGS-B
    bounds = [(low, high) for low, high in zip(xmin, xmax)]

    # bnds = ((0.001, 1000), (0.001, 100))

    # print('asdfghl', minimize(min_fun, x0, args=(Vvalues, Yvalues),bounds=bnds))

    # define the new step taking routine and pass it to basinhopping
    take_step = RandomDisplacementBounds(xmin=xmin, xmax=xmax)

    # use method L-BFGS-B because the problem is smooth and bounded
    # minimizer_kwargs = dict(method="Powell",  args=(Vvalues, Yvalues))
    minimizer_kwargs = dict(method="L-BFGS-B", bounds=bounds, args=(Vvalues, Yvalues))
    res = basinhopping(min_fun, x0, minimizer_kwargs=minimizer_kwargs, niter=100, take_step=take_step)

    return res.x    

                                      
def min_fun(x, vvals, yvals):
    """minimises equation 7 using Q4 minimisation given in equation 10 of 
       Gritsevich 2007 - 'Validity of the photometric formula for estimating 
       the mass of a fireball projectile'

    """ 
    res = 0.
    for i in range(len(vvals)):
        res += pow(2 * x[0] * np.exp(-yvals[i]) - (scipy.special.expi(x[1]) - scipy.special.expi(x[1]* vvals[i]**2) ) * np.exp(-x[1]) , 2)
    #       #sum...alpha*e^-y*2                     |__________________-del______________________________________|     *e^-beta    
        # res += (np.log(2 * x[0]) -yvals[i] - np.log(scipy.special.expi(x[1]) - scipy.special.expi(x[1]* vvals[i]**2) ) -x[1]) * 2

    return res

                                      
def min_fun_2(x, vvals, yvals):
    """minimises equation 7 using Q4 minimisation given in equation 10 of 
       Gritsevich 2007 - 'Validity of the photometric formula for estimating 
       the mass of a fireball projectile'

    """ 
    res = 0.
    x^(1 - mu) * (1 - np.exp(-he + ht + v))
    for i in range(len(vvals)):
        res += pow(2 * x[0] * np.exp(-yvals[i]) - (scipy.special.expi(x[1]) - scipy.special.expi(x[1]* vvals[i]**2) ) * np.exp(-x[1]) , 2)
    #       #sum...alpha*e^-y*2                     |__________________-del______________________________________|     *e^-beta    

    return res

def add_vel_column(table, eci_bool=False, tcolname='datetime', order=2, extrapol_edge=True):

    
    if order not in [1, 2]:
        raise VelocityAnalysisError("Only order 1 and 2 are implemented for velocities")
    
    n = len(table)
    
    t = Time(table[tcolname].data.tolist()) 
    relative_time = Time(t - t[0], format='sec').sec

    delta_X = np.empty(n)
    delta_T = np.empty(n)

    if 'X_eci' in table.colnames:
        xcolname = 'X_eci'
        ycolname = 'Y_eci' 
        zcolname = 'Z_eci' 
    else:
        xcolname = 'X_geo' 
        ycolname = 'Y_geo' 
        zcolname = 'Z_geo' 

    # Compute deltas in time and position
    # Order 1: indices [1,n] are comptuted
    # Order 2: indices [1,n-1] are comptuted, using a second order kangaroo method to mix Start with Starts and Ends with Ends

    delta_T[order:] = Time(t[order:] - t[0:n - order], format='sec').sec
    for i in range(order,n-order):
        dx =  np.sqrt(pow(table[xcolname][i]-table[xcolname][i-order], 2)+
                      pow(table[ycolname][i]-table[ycolname][i-order], 2)+
                      pow(table[zcolname][i]-table[zcolname][i-order], 2))

        if isinstance(dx, Quantity):
            delta_X[i] =  dx.value
        else:
            delta_X[i] =  dx

    # Compute velocities
    vel_X = delta_X / delta_T

    if order == 1:
        vel_X[0] = np.nan
        vel_X[-1] = np.nan
    elif order == 2:
        vel_X[0] = np.nan
        vel_X[1] = np.nan
        vel_X[-1] = np.nan
        vel_X[-2] = np.nan
    else:
        pass

    # Return the new velocity column
    theUnit = table[xcolname].unit / u.second
    if 'kang_vels' not in table.colnames:
        new_col = Column(name='kang_vels', data=vel_X * theUnit)
        table.add_column(new_col)
    else:
        table['kang_vels'] = vel_X * theUnit

    return 

def stack_observations(tables, eci_bool=False):

    if len(tables) < 2:
        return tables[0]
    
    table_list_no_mixin_cols = [Table(t) for t in tables]

    for t in table_list_no_mixin_cols:
        t['de_bruijn_sequence_element_index'] = t['de_bruijn_sequence_element_index'].astype(str)
        t['dash_start_end'] = t['dash_start_end'].astype(str)
        t['periodic_pulse'] = t['periodic_pulse'].astype(str)
        t['pick_flag'] = t['pick_flag'].astype(str)
        t['encoding_type'] = t['encoding_type'].astype(str)
        t['datetime'] = t['datetime'].astype(str)
        # if 'kang_vels' not in t.colnames:
        add_vel_column(t, eci_bool, order=2)
   

    # 'initial_velocity' does not necessarily exist yet
    if 'EKS_initial_velocity_all_cam' in table_list_no_mixin_cols[0].meta:
        v0s = [t.meta['EKS_initial_velocity_all_cam'] for t in table_list_no_mixin_cols]
        v_init = np.nanmean(np.asarray(v0s))
    else:
        v_init = 0.
    return QTable(vstack(table_list_no_mixin_cols, metadata_conflicts='silent')), v_init    #join(table_list_no_mixin_cols, join_type='left'), v0   #


def run_Triang_to_alpha_beta_all(table_list, smoothdata=None, cropped_smooth_data=False, event_codename='fireball', wdir='/tmp/', kwargs={}):
    # stack observations and sort by datetime
    stacked_data, v_init = stack_observations(table_list)
    stacked_data = Table(stacked_data)
    stacked_data.sort(['datetime'])
    logger = logging.getLogger('trajectory')
    # print(stacked_data)
    # print('asdf', stacked_data['D_DT_EKS'][0:5])
    # exit(1)

    # normalising parameters 
    h0 = 7160.

    if v_init == 0.:
        if 'D_DT_EKS' in stacked_data.colnames:
            a = np.nonzero(stacked_data['D_DT_EKS'])
            v_init = np.nanmean([stacked_data['D_DT_EKS'][i] for i in a[0][0:3]])
        else:
            a = np.nonzero(stacked_data['D_DT_geo'])
            v_init = np.nanmean([stacked_data['D_DT_geo'][i] for i in a[0][0:3]])


    # calculate ballistic params
    zenith, _ = tu.get_zenith_and_bearing(stacked_data, 'beg')
    gamma = (90*u.deg - zenith).to(u.rad).value

    output_params = Triang_to_alpha_beta(stacked_data, gamma, v0_stackedmeta=v_init)

    # save results in each original table
    for t in table_list:
        t.meta['ballistic_alpha_all'] = stacked_data.meta['ballistic_alpha']
        t.meta['ballistic_beta_all'] = stacked_data.meta['ballistic_beta']
        t.meta['ballistic_entry_mass_all'] = stacked_data.meta['ballistic_entry_mass']          # mass[1][1] corresponds to density 3500 and shape 1.3
        t.meta['slope'] = stacked_data.meta['slope']
        t.meta['ballistic_final_mass_all'] = stacked_data.meta['ballistic_final_mass']

        vel_table_update(t, v_init, h0, t.meta['ballistic_alpha_all'], t.meta['ballistic_beta_all'], all_cams=True)
     
    if smoothdata != None and kwargs['trajectory_segment'] == 'all':
        try:
            logger.info('running ballistic params on EKS_smoothed_vels_all')

            v0  = smoothdata['D_DT_EKS'][0]

            if cropped_smooth_data:
                # masked_data = Table(stacked_data[~np.isnan(stacked_data['D_DT_geo'])])
                # alt = masked_data['height']
                # vel = masked_data['D_DT_geo']

                masked_data = Table(smoothdata[~np.isnan(smoothdata['D_DT_EKS'])])
                alt = masked_data['height']
                vel = masked_data['D_DT_EKS']

            else:
                alt = smoothdata['height']
                vel = smoothdata['D_DT_EKS']
                

            if isinstance((vel/v0), Quantity):
                Vvalues = (vel/v0).data
                v0 = v0.value
            else:
                Vvalues = vel/v0      #creates a matrix of V/Ve to give a dimensionless parameter for velocity

            if isinstance((alt/h0), Quantity):
                Yvalues = (alt/h0).data
            else:
                Yvalues = alt/h0  #        %creates a matrix of h/h0 to give a dimensionless parameter for altitude



            # Height
            h0 = 7160.

            Gparams= Q4_min(Vvalues, Yvalues)
            alpha = Gparams[0]
            beta = Gparams[1]

            #print('\nAlpha parameter:', alpha, '\nBeta parameter:', beta, '\npossible mass:', mass)
            logger.info('entry velocity: {0:.2f} - Alpha: {1:.2f} - Beta parameter: {2:.2f} - slope {3:2f}'.format( v0, alpha, beta, float(np.rad2deg(gamma))))
            
            x = np.arange(0,1, 0.00005);                                                                                     #create a matrix of x values
            fun = lambda x:np.log(alpha) + beta - np.log((scipy.special.expi(beta) - scipy.special.expi(beta* x**2) )/2);           
                                                                                                            #(obtained from Q4 minimisation)
            y = [fun(i) for i in x]

            sea_level_rho = 1.29
            cd = 1.#3
            A = [1.21, 1.3, 1.55]
            m_rho = [2700, 3500, 7000]
            sin_gamma = np.sin(gamma)
            mu = 2./3.

            # A_sphere = 1.21
            # A_brick = 1.55
            # rho = 3500
            me_sphere = pow(0.5 *  cd * 1.29 * 7160 * A[0] / (alpha * sin_gamma * m_rho[1]**(2/3.0)), 3.0)
            me_brick = pow(0.5 *  cd * 1.29 * 7160 * A[2] / (alpha * sin_gamma * m_rho[1]**(2/3.0)), 3.0)
            # mu = 2/3.0

            # mf_sphere = me_sphere * np.exp(-beta / (1-mu) *(1-Vvalues[-1]**2))
            # mf_brick = me_brick * np.exp(-beta / (1-mu) *(1-Vvalues[-1]**2))

            mass = []
            for i in A:

                mass.append([np.exp(-beta/(1-mu) * (1-Vvalues[-1]**2)) * pow(0.5 * cd * sea_level_rho * h0 * i / (pow(j, 2/3.) * sin_gamma * alpha), 3) for j in m_rho])


            # print(mass[0][1], mass[2][1])
            # print(mf_sphere, mf_brick)
            # print(me_sphere, me_brick)
            # ppp


            plt.close()
            plt.scatter(Vvalues, Yvalues, color='k', 
                        marker='x', label=None)
                        #label=data.meta['telescope'] + " " + data.meta['location'])
            plt.plot(x, y, color='k', 
                        label='{0} {1:.3f} {2} {3:.3f} {4} {5:.3f} \n V0 used {10:.3f} \n Possible sphere mass (@ 3500 kg/m3, M0={8:.2f}): >{6:.2f} kg \n Possible brick mass (@ 3500 kg/m3, M0={9:.2f}): >{7:.2f} kg'.format(r'$\alpha$ = ', alpha, r'$\beta$ = ', beta, r'$\gamma$ = ', np.rad2deg(gamma), mass[0][1], mass[2][1], me_sphere, me_brick, v0))
             

            plt.title(event_codename + " - Ballistic Alpha-Beta plot for EKS velocities")

            plt.xlabel("Normalised velocity")
            plt.ylabel("Normalised altitude")
            plt.legend(frameon=True, loc='upper left', fancybox=True, framealpha=0.5, fontsize='xx-small')
            # plt.show()
            
            fname = os.path.join(wdir, event_codename + "_alpha_beta_EKS_vels.png")
            plt.savefig(fname, dpi=150)

            vel_col = Column(data=(np.asarray(Vvalues) * v0), name = 'D_DT_blstc')
            smoothdata.add_column(vel_col)

            fname = os.path.join(wdir, event_codename + "_EKS_all_cams_" + kwargs['trajectory_segment'] + ".csv")
            smoothdata.write(fname, format='ascii.csv', delimiter=',')
        except ValueError:
            pass

    return output_params


def grav(alt):
    G = 6.6726e-11    #N-m2/kg2
    M = 5.97237e24      #kg Mass of Earth
    mean_rad = 6371000.0
        
    return G * M / (mean_rad + alt) ** 2

def atm_roh(alt):
    p0_earth = 1.29
    scale_height_earth = 7160.0

    return p0_earth * exp(-alt / scale_height_earth)

def new_alt(alt, gamma, dt, v):
    dh = sin(gamma) * dt * v
    return alt - dh

def integrals_abl(X, h, param): 
    [kv, kr, pm] = param
    g = grav(h)
    [v, rad, energy, sin_gam] = X

    m = 4. / 3. * pi * (rad)**3 * pm

    Xdot=[0., 0., 0., 0.] 

    Xdot[3] = -g / v**2 * (1 - sin_gam) / sin_gam
    Xdot[0] = kv * atm_roh(h) * v / (rad * sin_gam) - g / v
    Xdot[1] = kr * atm_roh(h) * v**2 / (sin_gam)
    Xdot[2] = m * v * Xdot[0]
    # if h >62000:
    #     print(h, X[0], Xdot[0])

    return Xdot #

def mass_integ(X, h, param): 

    [ht, he, mu] = param
    h0 = 7160.0

    yt = ht / h0
    y = h / h0

    # Xdot = np.exp(-y + yt) / ((1 - mu) * pow(X, mu))
    
    return Xdot #

def ablation_fun(ht, mu):
    
    mt = [0.]
    ode_out = scipy.integrate.odeint(mass_integ, mt, [0, 1],  args = (param,)) 
        

    return M0

def plot_all_nonpipeline(output_params_collection, event_codename='fireball', wdir='/tmp/', kwargs={'trajectory_segment':'all'}):
    """
    Plot a list of ballistic parameters runs
    """
    
    # initiate color palette
    palette = itertools.cycle(sns.color_palette())
    
    plt.close()
    

    for key in range(len(output_params_collection)):
        plt.figure(1)

        res = output_params_collection[key]
        # new color for each set of observations
        color = next(palette)
    
        alpha = res['alpha']
        beta = res['beta']
        mass = res['mass']
        A = res['A']
        m_rho = res['m_rho']
        Yvalues = res['Yvalues']
        Vvalues = res['Vvalues']
        
        x = np.arange(0,1, 0.00005);                                                                                     #create a matrix of x values
        fun = lambda x:np.log(alpha) + beta - np.log((scipy.special.expi(beta) - scipy.special.expi(beta* x**2) )/2);           
                                                                                                        #(obtained from Q4 minimisation)
        y = [fun(i) for i in x]
        
        if not res['telescope'] == 'all':
            extra_text = '\nM_sp @ {0}, M0={1:.2f} >{2:.2f} kg \nM_br @ {0}, M0={3:.2f}>{4:.2f} kg'.format(m_rho[1], res['me_sphere'], res['mf_sphere'], res['me_brick'], res['mf_brick'])

            plt.scatter(Vvalues, Yvalues, color=color,
                        marker='x', label=None)
                        #label=data.meta['telescope'] + " " + data.meta['location'])
            plt.plot(x, y, color=color, 
                        label='{0: <10} : {1} {2:.3f} {3} {4:.3f}'.format(res['telescope'], r'$\alpha$ = ', alpha, r'$\beta$ = ', beta) + extra_text)
        else:
            extra_text = '\nV0 used {7:.3f} \nM_sp @ {0}, M0={1:.2f} >{2:.2f} kg \nM_br @ {0}, M0={3:.2f}>{4:.2f} kg'.format(m_rho[1], res['me_sphere'], res['mf_sphere'], res['me_brick'], res['mf_brick'], r'$\gamma$ = ', res['gamma'], res['reference_velocity'])
            
            plt.plot(x, y, color='k', 
                        label='{0: <10} : {1} {2:.3f} {3} {4:.3f}'.format(res['telescope'], r'$\alpha$ = ', alpha, r'$\beta$ = ', beta) + extra_text)
        
            plt.figure(2)
            plt.scatter(Vvalues, Yvalues, color='b',
                        marker='x', label=None)
                        #label=data.meta['telescope'] + " " + data.meta['location'])
            plt.plot(x, y, color='k', 
                        label='{0: <10} : {1} {2:.3f} {3} {4:.3f}'.format(res['telescope'], r'$\alpha$ = ', alpha, r'$\beta$ = ', beta) + extra_text)
        

            plt.title(event_codename + " - Ballistic Alpha-Beta plot - stacked")
    
            plt.xlabel("Normalised velocity")
            plt.ylabel("Normalised altitude")
            plt.legend(frameon=True, loc='best', fancybox=True, framealpha=0.5, fontsize='xx-small')

            fname = os.path.join(wdir, event_codename + "_alpha_beta_consistency_check_stacked_" + kwargs['trajectory_segment'] + ".png")
            #while os.path.isfile(fname):
                #fname = fname.split('.')[0] + '_alt.png'
            #plt.savefig(fname)
            plt.savefig(fname, dpi=150)
            plt.close()

    plt.title(event_codename + " - Ballistic Alpha-Beta plot")
    
    plt.xlabel("Normalised velocity")
    plt.ylabel("Normalised altitude")
    plt.legend(frameon=True, loc='best', fancybox=True, framealpha=0.5, fontsize='xx-small')

    fname = os.path.join(wdir, event_codename + "_alpha_beta_consistency_check_" + kwargs['trajectory_segment'] + ".png")
    #while os.path.isfile(fname):
        #fname = fname.split('.')[0] + '_alt.png'
    #plt.savefig(fname)
    plt.savefig(fname, dpi=150)
    plt.close()
    



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run ballistic parameters on raw camera files.')
    #inputgroup = parser.add_mutually_exclusive_group(required=True)
    parser.add_argument("-d","--inputthing",type=str,
            help="single file or folder containing triangulated files with extension .ecsv", required=True)
    parser.add_argument("-p", "--plot", action="store_true", default=True, 
            help="plot results")
    
    args = parser.parse_args()   

    ## check if the input directory given exists and pull all ecsv files, and extract data


    if (os.path.isdir(args.inputthing)):
        working_dir = args.inputthing
        # list all altaz files in this directory
        forbidden_kws=['CUT_','key_parameters','MC_orbits']

        all_potential_files = sorted(glob.glob(os.path.join(working_dir,"*.ecsv")))
    
        filenames = [x for x in all_potential_files
                    if not any(s in x for s in forbidden_kws)]
    
    else:        
        working_dir = os.path.dirname(args.inputthing)
        filenames = [args.inputthing]
    
    logger = logging.getLogger('trajectory')
    
    # read-in data, ignoring files with no reliable timing
    table_list = []
    for f in filenames:
        t = Table.read(f, format='ascii.ecsv', guess=False, delimiter=',')
        t.meta['self_disk_path'] = f
        if dfn_utils.is_type_pipeline(t, 'velocitic'):
            table_list += [t]
        else:
            logger.debug('Skiping {0} : does not contain velocity information'.format(os.path.basename(f)))


    output_params_collection = []
    # run on every single file
    for t in table_list:
        #gamma = get_entry_angle(t)
        # print(t.meta['self_disk_path'])
        zenith, _ = tu.get_zenith_and_bearing(t, 'beg')
        gamma = (90*u.deg - zenith).to(u.rad).value
        if 'X_eci' in t.colnames:
            eci_bool = True
        else: eci_bool =False
        # if 'kang_vels' not in t.colnames:
        add_vel_column(t, eci_bool, order=2)
        output_params = Triang_to_alpha_beta(t, gamma)
        output_params.update(t.meta)
        output_params_collection += [output_params]
        
    # run on all files
    if len(table_list) > 1:
        output_params = run_Triang_to_alpha_beta_all(table_list)
        output_params['telescope'] = 'all'
        output_params_collection = [output_params] + output_params_collection
        if args.plot:
            plot_all_nonpipeline(output_params_collection, event_codename=table_list[0].meta['event_codename'], wdir=working_dir)
    elif args.plot: 
        plot_all_nonpipeline(output_params_collection, event_codename=table_list[0].meta['event_codename'], wdir=working_dir)

    # write results
    for t in table_list:
        t.write(t.meta['self_disk_path'], format='ascii.ecsv', delimiter=',')
        
    
    '''
    TODO DEPRECATED TO DELETE AT SOME POINT
    ## get data
    for f in filenames:
        
        if "notime" not in f and "no_time" not in f:

            try:
                data = vstack([data, Table.read(f, format='ascii.ecsv', guess=False, delimiter=',')], metadata_conflicts='silent')
            except NameError:
                data = Table.read(f, format='ascii.ecsv', guess=False, delimiter=',')
                gamma = get_entry_angle(data)
                print('incoming entry angle:', gamma *360/(2 * 3.14159))

            #else:
            #    data = vstack([data, Table.read(f, format='ascii.ecsv', guess=False, delimiter=',')])

            if dfn_utils.is_type_pipeline(data, 'velocitic'):
                try:
                    event_codename = data.meta['event_codename']
                except:
                    event_codename = "fireball"
                    pass

            else:
                print('file', f, 'has no velocities')
        else: 
            print('file', f, 'has no time')

    

    data.sort(['datetime'])
    alpha, beta, mass = Triang_to_alpha_beta(data, gamma, event_codename, next(palette))

    for f in filenames:
        read_file = Table.read(f, format='ascii.ecsv', guess=False, delimiter=',')
        ### ADD meta
        read_file.meta['ballistic_coefficient_alpha'] = float(str(alpha))
        read_file.meta['mass_loss_param_beta'] = float(str(beta))
        read_file.meta['ballistic_entry_mass'] = float(str(mass[1][1]))


        read_file.write(f, format='ascii.ecsv', delimiter=',')
    
    fname = os.path.join(working_dir, event_codename + "_alpha_beta_consistency_check.png")
    plt.savefig(fname)
    
    '''

