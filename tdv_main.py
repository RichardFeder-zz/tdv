import numpy as np 
import matplotlib.pyplot as plt 
import astropy.stats
from astropy.modeling import models, fitting
from astropy.table import hstack
import scipy.signal as signal
import scipy.stats
from scipy.optimize import curve_fit
from astropy.table import Table
from scipy import special,misc
import heapq
import os
import sys

base_dir = '/Users/richardfeder/Documents/tdv_project/'

obj = str(sys.argv[1])
if len(sys.argv) > 2:
	verbtype = sys.argv[2]
else:
	verbtype = 0


class transit():

    def __init__(self):
        self.priod_bounds = [3000., 15000.]
        self.dsig_thresh = 0.1
        self.base_dir = base_dir
        self.dl = 20
        self.dr = 40
        self.min_counts = 4
        self.verbtype = verbtype
        self.obj = obj
        self.transits, self.times, self.errs, self.transit_counts = [[] for x in xrange(4)]
        self.transit_counts_clip, self.times_clip, self.transits_clip, self.errs_clip = [[] for x in xrange(4)]
        self.fig_dir = self.base_dir + '/figures/'+self.obj
        if not os.path.isdir(self.fig_dir):
        	os.makedir(self.fig_dir)
        self.fig_dir += '/'


    def sine_fit(x,a,b,c):
        return a*np.sin(6.14*x/b) + c

    def get_timeseries(self):
        self.name = 'Kepler '+str(self.obj)
        self.table_raw = Table.read(self.base_dir+'data/raw/k' + str(self.obj) + '_raw.tbl', format='ipac')
        self.table_err = Table.read(self.base_dir+'data/err/k' + str(self.obj) + '_err.tbl', format='ipac')

    def readin_transit_and_err(self):
        self.t1_time = self.table_raw['TIME'] 
        self.PDCSAP_FLUX = self.table_raw['PDCSAP_FLUX'] 
        self.PDCSAP_err = self.table_err['PDCSAP_FLUX_ERR']

    def get_sine_fit(xs, ys, bounds=None):
        if bounds is not None:
            popt, pcov= curve_fit(sine_fit, xd, yd, bounds=bounds)
        else:
            popt, pcov = curve_fit(sine_fit, xd, yd)

        perr = np.sqrt(np.diag(pcov)) #ASSOCIATED ERRORS
        return popt, p_err

    def plot_norm_flux_lcur(self):
        plt.figure()
        plt.title('Light Curve with Median Filter Normalization - '+ self.name)
        plt.xlabel('Time (BKJD)')
        plt.ylabel('Normalized Flux')
        plt.plot(self.clip_time_flux_err[:,0], self.clip_time_flux_err[:,1])
#         plt.plot(self.time_clip, self.norm_flux_clip)
        plt.show()

    def plot_best_fit_lcur(self, popt, f_err):

        linear_fit = np.poly1d(np.polyfit(self.t_min, self.depths, 1, w=self.weights))
        ts = np.linspace(np.amin(self.time_clip), np.amax(self.time_clip), 500) 

        plt.figure()
        plt.title('Best fit curves to Transit Depth Time Series  - '+self.name)
        plt.xlabel('Time (BKJD)')
        plt.ylabel('Transit Depth')
        plt.errorbar(self.t_min, self.depths, yerr=self.depths_err, linestyle="None", fmt='o')
        plt.plot(ts, linear_fit(ts), label='Linear Fit')
        plt.plot(ts, func(ts, *popt), label='Sinusoid Fit')
        plt.legend()
        plt.ylim(self.d_min-self.d_std, self.d_max+self.d_std)
        if save=='yes':
            plt.savefig(self.fig_dir+str(self.obj)+'/'+ str(self.obj) + '_tdv_sinusoid.png')
        plt.show()

    def normalize_light_curve(self, plot='yes'):

        n1 = signal.medfilt(self.PDCSAP_FLUX, 57) 

        norm_flux = self.PDCSAP_FLUX/n1
        norm_err = self.PDCSAP_err/n1

        stdclip, meanclip = np.std(norm_flux), np.mean(norm_flux)
        highclip = meanclip + stdclip

        self.clip_time_flux_err = np.array([[self.t1_time[i], norm_flux[i], norm_err[i]] for i in xrange(len(self.t1_time)) if norm_flux[i] < highclip])

        if plot=='yes':
            self.plot_norm_flux_lcur()


    def filter_transits(self):

        fluxes = self.clip_time_flux_err[:,1]
        sigf = np.std(fluxes)
        x=0
        while x < self.clip_time_flux_err.shape[1]:
            print fluxes[x], 1-2*sigf
            print self.clip_time_flux_err[(x-self.dl):(x+self.dr), 1]
            if fluxes[x] < 1-2*sigf:
                self.transits.append(self.clip_time_flux_err[(x-self.dl):(x+self.dr), 1])
                self.times.append(self.clip_time_flux_err[(x-self.dl):(x+self.dr), 0])
                self.errs.append(self.clip_time_flux_err[(x-self.dl):(x+self.dr), 2])
                x += self.dr
            else:
                x += 1

        for i in xrange(len(self.transits)):
            count = len([a for a in self.transits[i] if a < 1-2*sigf])
            self.transit_counts.append(count)

            if count > self.min_counts:
                self.transits_clip.append(transits[i])
                self.transit_counts_clip.append(count)
                self.times_clip.append(times[i])
                self.errs_clip.append(errs[i])

        if self.verbtype > 1:
            print(str(len(self.transits_clip)) + ' transits had more than five points in their transit light curves.')
            print('Mean, STD for filtered transit data points: ' + str(np.mean(self.transit_counts_clip)) + ', ' + str(np.std(self.transit_counts_clip)))


    def plot_depth_hist(self, depths, minmax, save=0):
        plt.figure()
        plt.title('Transit Depth Histogram')
        plt.xlabel('Depth')
        plt.ylabel('Number')
        plt.hist(self.depths, bins=int(len(self.depths)/3))
        for i in minmax:
            plt.axvline(i, linestyle='dashed', color='b')
        if save:
            plt.savefig(self.fig_dir+'depth_hist.png', bbox_inches='tight')
        plt.show()


    def plot_light_curve(self):
        # def plot_light_curve(table1, table2, save='no', verbype=0):
        self.normalize_light_curve(plot='no')
        self.filter_transits()
        # transit_counts_clip, times_clip, transits_clip, errs_clip = filter_transits(time_clip, norm_flux_clip, norm_err_clip)


        depths = np.array([np.max(i)-np.min(i) for i in transits_clip])

        b = np.histogram(depths, bins=40)
        depth_mode = b[1][np.argmax(b[0])]
        depth_std = np.std(depths)
        ratio = depth_std/depth_mode

        #approximate way to filter out two distinct populations of transits from multiple planets
        if ratio > self.dsig_thresh:
            minmax = [depth_mode - ratio*np.std(depths), depth_mode + ratio*np.std(depths)]  
        else:
            minmax = [np.median(depths) - 3*depth_std, np.median(depths) + 3*depth_std]


        plot_depth_hist(depths, minmax)

        if verbtype > 1:
            print('Standard Deviation / Mode of All Depths: ' + str(ratio))

        if ratio > self.dsig_thresh:
            transits_clip2, depths_clip2, times_clip2, errs_clip2 = [[] for x in range(4)]

            tdte_array = np.array([[self.transit_clip[i], self.depths[i], self.times_clip[i], self.errs_clip[i]] for i  \
            in xrange(len(self.depths)) if self.depths[i]<minmax[1] and self.depths[i]>minmax[0]]).transpose()

            hist_d2 = np.histogram(tdte_array[1,:], bins=int(0.3*tdte_array.shape[0]))
            depth2_std = np.std(tdte_array[1,:])

            depth2_mode = depth_mode = hist_d2[1][np.argmax(hist_d2[0])]
            minmax = [np.median(tdte_array[1,:]) - 3*depth2_std, np.median(tdte_array[1,:]) + 3*depth2_std]

            transits_clip = tdte_array[0,:]
            depths = tdte_array[1,:]
            times_clip = tdte_array[2,:]
            errs_clip = tdte_array[3,:]


        mins, mins_err, baselines, baselines_err, t_min = [[] for x in range(5)]


        for i in range(len(transits_clip)):
            tot_avg = np.mean(transits_clip[i])

            baseline_pts = np.array([[transit_clip[i][a], errs_clip[i][a]] for a in xrange(len(transit_clip[i])) if transit_clip[i][a] > tot_avg])
            baselines.append(np.mean(baseline_pts[:,0]))
            baselines_err.append(np.mean(baseline_pts[:,1]))

            n = np.argmin(transits_clip[i])
            t = times_clip[i][n]
            t_min.append(t)

            min_3 = np.array(transits_clip[i]).argsort()[:3]

            min_avg = (transits_clip[i][min_3[0]] + transits_clip[i][min_3[1]] + transits_clip[i][min_3[2]])/3
            min_err = (errs_clip[i][min_3[0]] + errs_clip[i][min_3[1]] + errs_clip[i][min_3[2]])/3

            mins.append(min_avg)
            mins_err.append(min_err)

        d_and_err = np.zeros(shape=(2, len(baselines))).astype(np.float32)

        d_and_err[0,:] = np.array(baselines) - np.array(mins)
        d_and_err[1,:] = np.sqrt(np.array(baselines_err)**2 + np.array(mins_err)**2)


        weights = 1/ d_and_err[0,:]

        d_std, d_avg, d_min, d_max = np.std(d_and_err[0,:]), np.mean(d_and_err[0,:]), np.amin(d_and_err[0,:]), np.amax(d_and_err[0,:])

        t_min = np.array(t_min)
        depths = np.array(depths)

        # these bounds restrict the parameter space explored by sinusoidal fits
        min_bounds = (0 , self.period_bounds[0], d_avg-2*d_std)
        max_bounds = (0.02*d_avg, self.period_bounds[1], d_avg+2*d_std)    


        popt, p_err = get_sine_fit(t_min, depths, bounds=(min_bounds, max_bounds))
        self.plot_best_fit_lcur(popt, p_err[0]) #takes in optimal parameters and errors on flux

        if verbtype > 1:
            print('depth average: ' + str(depth_avg) + '. depth_std: ' + str(depth_std))
            print('Optimal Parameter Values: a = ' + str(popt[0]) + ', b = ' + str(popt[1]) + ', c = ' + str(popt[2]))
            print('Parameter Errors: ' + str(perr))


b117_obj = transit()

b117_obj.get_timeseries()
b117_obj.readin_transit_and_err()
b117_obj.normalize_light_curve()
b117_obj.filter_transits()