import numpy as np 
# import matplotlib.pyplot as plt 
import matplotlib
matplotlib.use('TkAgg')
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

base_dir = '/Users/richardfeder/Documents/tdv/'

obj = str(sys.argv[1])
if len(sys.argv) > 2:
	verbtype = sys.argv[2]
else:
	verbtype = 0

save = 1


def sine_fit_wphase(x,a,b,c,d):
    return a*np.sin(6.14*x/b+d) + c
def sine_fit(x,a,b,c):
    return a*np.sin(6.14*x/b) + c

class transit():

    def __init__(self):
        self.period_bounds = [3000., 10000.]
        self.dsig_thresh = 0.1
        self.base_dir = base_dir
        self.dl = 20
        self.dr = 40
        self.min_counts = 4
        self.verbtype = verbtype
        self.obj = obj
        self.transits, self.times, self.errs, self.transit_counts = [[] for x in xrange(4)]
        self.transit_counts_clip, self.times_clip, self.transits_clip, self.errs_clip = [[] for x in xrange(4)]
        self.save = save
        self.fig_dir = self.base_dir + '/figures/'+self.obj
        if not os.path.isdir(self.fig_dir):
            os.makedirs(self.fig_dir)
        self.fig_dir += '/'

    def get_timeseries(self):
        self.name = 'Kepler '+str(self.obj)
        self.table_raw = Table.read(self.base_dir+'data/raw/k' + str(self.obj) + '_raw.tbl', format='ipac')
        self.table_err = Table.read(self.base_dir+'data/err/k' + str(self.obj) + '_err.tbl', format='ipac')

    def readin_transit_and_err(self):
        self.t1_time = self.table_raw['TIME'] 
        self.PDCSAP_FLUX = self.table_raw['PDCSAP_FLUX'] 
        self.PDCSAP_err = self.table_err['PDCSAP_FLUX_ERR']

    def get_sine_fit(self, xs, ys):

        # print self.min_bounds[0]
        # print self.max_bounds[0]

        # print np.average([float(self.min_bounds[0]), float(self.max_bounds[0])])
        # print np.average(self.min_bounds[1], self.max_bounds[1])
        # print np.average(self.min_bounds[2], self.max_bounds[2])

        first_guess = [np.average([self.min_bounds[0], self.max_bounds[0]]), np.average([self.min_bounds[1], self.max_bounds[1]]), np.average([self.min_bounds[2], self.max_bounds[2]])]

        print 'first guess'
        print first_guess
        if self.period_bounds is not None:
            popt, pcov= curve_fit(sine_fit, xs, ys, bounds=(self.min_bounds, self.max_bounds), p0=first_guess)
        else:
            popt, pcov = curve_fit(sine_fit, xs, ys, sigma=self.weights, p0=first_guess)
    
        p_err = np.sqrt(np.diag(pcov)) #ASSOCIATED ERRORS
        return popt, p_err

    def plot_norm_flux_lcur(self):
        plt.figure()
        plt.title('Light Curve with Median Filter Normalization - '+ self.name)
        plt.xlabel('Time (BKJD)')
        plt.ylabel('Normalized Flux')
        plt.plot(self.clip_time_flux_err[:,0], self.clip_time_flux_err[:,1])
        if self.save:
            plt.savefig(self.fig_dir+'norm_flux_lcur.png', bbox_inches='tight')
        # plt.show()
        plt.close()

    def plot_best_fit_lcur(self, popt, f_err):
        
        linear_fit = np.poly1d(np.polyfit(self.t_min, self.d_and_err[0,:], 1, w=self.weights))
        ts = np.linspace(np.amin(self.times_clip[0]), np.amax(self.times_clip[-1]), 500) 

        plt.figure()
        plt.title('Best fit curves to Transit Depth Time Series  - '+self.name)
        plt.xlabel('Time (BKJD)')
        plt.ylabel('Transit Depth')
        plt.errorbar(self.t_min, self.d_and_err[0,:], yerr=self.d_and_err[1,:], linestyle="None", fmt='o')
        plt.plot(ts, linear_fit(ts), label='Linear Fit')
        plt.plot(ts, sine_fit(ts, *popt), label='Sinusoid Fit')
        plt.ylim(np.mean(self.d_and_err[0,:])-3*self.d_std, np.mean(self.d_and_err[0,:])+3*self.d_std)
        plt.legend()
        if self.save:
            plt.savefig(self.fig_dir+ str(self.obj) + '_tdv_sinusoid.png', bbox_inches='tight')
        plt.close()
        # plt.show()

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
        while x < self.clip_time_flux_err.shape[0]:
            
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
                self.transits_clip.append(self.transits[i])
                self.transit_counts_clip.append(count)
                self.times_clip.append(self.times[i])
                self.errs_clip.append(self.errs[i])

        if self.verbtype > 1:
            print(str(len(self.transits_clip)) + ' transits had more than five points in their transit light curves.')
            print('Mean, STD for filtered transit data points: ' + str(np.mean(self.transit_counts_clip)) + ', ' + str(np.std(self.transit_counts_clip)))


    def plot_depth_hist(self):
        plt.figure()
        plt.title('Transit Depth Histogram')
        plt.xlabel('Depth')
        plt.ylabel('Number')
        plt.hist(self.depths, bins=int(len(self.depths)/3))
        for i in self.minmax:
            plt.axvline(i, linestyle='dashed', color='b')
        if self.save:
            plt.savefig(self.fig_dir+'depth_hist.png', bbox_inches='tight')
        # plt.show()
        plt.close()

    def main(self):
        self.get_timeseries()
        self.readin_transit_and_err()
        self.normalize_light_curve(plot='no')
        self.filter_transits()

        self.depths = np.array([np.max(i)-np.min(i) for i in self.transits_clip])

        b = np.histogram(self.depths, bins=40)
        depth_mode = b[1][np.argmax(b[0])]
        depth_std = np.std(self.depths)
        ratio = depth_std/depth_mode

        #approximate way to filter out two distinct populations of transits from multiple planets
        if ratio > self.dsig_thresh:
            self.minmax = [depth_mode - ratio*np.std(self.depths), depth_mode + ratio*np.std(self.depths)]  
        else:
            self.minmax = [np.median(self.depths) - 3*depth_std, np.median(self.depths) + 3*depth_std]


        self.plot_depth_hist()

        if verbtype > 1:
            print('Standard Deviation / Mode of All Depths: ' + str(ratio))

        if ratio > self.dsig_thresh:
            transits_clip2, depths_clip2, times_clip2, errs_clip2 = [[] for x in range(4)]

            tdte_array = np.array([[self.transits_clip[i], self.depths[i], self.times_clip[i], self.errs_clip[i]] for i  \
            in xrange(len(self.depths)) if self.depths[i]<self.minmax[1] and self.depths[i]>self.minmax[0]]).transpose()

            hist_d2 = np.histogram(tdte_array[1,:], bins=int(0.3*tdte_array.shape[0]))
            depth2_std = np.std(tdte_array[1,:])

            depth2_mode = depth_mode = hist_d2[1][np.argmax(hist_d2[0])]
            self.minmax = [np.median(tdte_array[1,:]) - 3*depth2_std, np.median(tdte_array[1,:]) + 3*depth2_std]

            self.transits_clip = tdte_array[0,:]
            self.depths = tdte_array[1,:]
            self.times_clip = tdte_array[2,:]
            self.errs_clip = tdte_array[3,:]
            
        mins, mins_err, baselines, baselines_err, t_min = [[] for x in range(5)]


        for i in range(len(self.transits_clip)):
            tot_avg = np.mean(self.transits_clip[i])
            
            baseline_pts = np.array([[self.transits_clip[i][a], self.errs_clip[i][a]] for a in xrange(len(self.transits_clip[i])) if self.transits_clip[i][a] > tot_avg])
            baselines.append(np.mean(baseline_pts[:,0]))
            baselines_err.append(np.mean(baseline_pts[:,1]))

            n = np.argmin(self.transits_clip[i])
            t = self.times_clip[i][n]
            t_min.append(t)

            min_3 = np.array(self.transits_clip[i]).argsort()[:3]

            min_avg = (self.transits_clip[i][min_3[0]] + self.transits_clip[i][min_3[1]] + self.transits_clip[i][min_3[2]])/3
            min_err = (self.errs_clip[i][min_3[0]] + self.errs_clip[i][min_3[1]] + self.errs_clip[i][min_3[2]])/3

            mins.append(min_avg)
            mins_err.append(min_err)

        self.d_and_err = np.zeros(shape=(2, len(baselines))).astype(np.float32)

        self.d_and_err[0,:] = np.array(baselines) - np.array(mins)
        self.d_and_err[1,:] = np.sqrt(np.array(baselines_err)**2 + np.array(mins_err)**2)
        

        self.weights = 1./ self.d_and_err[0,:]

        self.d_std, self.d_avg, self.d_min, self.d_max = np.std(self.d_and_err[0,:]), np.mean(self.d_and_err[0,:]), np.amin(self.d_and_err[0,:]), np.amax(self.d_and_err[0,:])

        
        self.t_min = np.array(t_min)
        self.depths = np.array(self.depths).astype(np.float32)
    
        # self.min_bounds = (0 , self.period_bounds[0], self.d_avg-2*self.d_std, 0)
        # self.max_bounds = (0.02*self.d_avg, self.period_bounds[1], self.d_avg+2*self.d_std, np.pi)
        self.min_bounds = (0. , self.period_bounds[0], self.d_avg-3*self.d_std)
        self.max_bounds = (0.05*self.d_avg, self.period_bounds[1], self.d_avg+3*self.d_std)
    
        popt, p_err = self.get_sine_fit(self.t_min, self.d_and_err[0,:])
        self.plot_best_fit_lcur(popt, p_err[0]) #takes in optimal parameters and errors on flux

        if verbtype > 1:
            print('depth average: ' + str(self.d_avg) + '. depth_std: ' + str(self.d_std))
            print('Optimal Parameter Values: a = ' + str(popt[0]) + ', b = ' + str(popt[1]) + ', c = ' + str(popt[2]))

            # print('Optimal Parameter Values: a = ' + str(popt[0]) + ', b = ' + str(popt[1]) + ', c = ' + str(popt[2])+ ' d='+ str(popt[3]))
            print('Parameter Errors: ' + str(p_err))


b117_obj = transit()

# b117_obj.get_timeseries()
# b117_obj.readin_transit_and_err()
# b117_obj.normalize_light_curve()
# b117_obj.filter_transits()

b117_obj.main()