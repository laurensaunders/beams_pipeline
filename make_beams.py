import numpy as np
import matplotlib.pyplot as plt
from numpy import genfromtxt
from scipy.interpolate import interp1d
import scipy.optimize as opt
import sys, os
import elliptical_beam

class AnalyticBeam:
    """
    Class to create analytic-only beams.

    Inputs
    ------
    params (dict): parameters that characterize the beam. Keys must include:
        'N' (int): number of pixels
        'pixel_size' (float): size of each pixel in degrees
        'beam_fwhm' (float): FWHM of the beam (depends on band)
        'ellipt' (float): ellipticity of the analytic beam
    """
    def __init__(self, params):
        self.params = params
        self.beam_matrix = self.make_beam_matrix()

    def make_beam_matrix(self):
        N = self.params['N']
        pixel_size = self.params['pixel_size']
        pix_size = pixel_size * 60.
        fwhm = self.params['beam_fwhm']
        ellipt = self.params['ellipt']

        betas = elliptical_beam.make_betas(N, pix_size, fwhm, ellipt)
        z = elliptical_beam.make_zeros(N, pix_size)

        s, d, an, bn = elliptical_beam.make_beam_params(betas[0], betas[1], betas[2], betas[3])
        beam_matrix = elliptical_beam.make_beam_matrix(s, d, an, z)

        return beam_matrix

class HFSSBeam:
    def __init__(self, params):
        self.params = params
        self.freq = np.linspace(self.params['fmin'], self.params['fmax'], self.params['numfreqs'])
        self.folders = {'IQ': self.params['folder']['I'],
                        'U': self.params['folder']['U']}
        self.stop_angle = self.params['stop_angle']
        self.mask_params = self.params['mask_params']
        self.npix = self.params['npix']

        self.phase_outputs_iq = self.run_phase_output(self.freq, self.folders['IQ'], self.stop_angle, self.mask_params, 'IQ')
        self.phase_outputs_u = self.run_phase_output(self.freq, self.folders['U'], self.stop_angle, self.mask_params, 'U')
        self.fits_iq = self.phase_fit_loop(self.freq, self.folders['IQ'], 'iq')
        self.fits_u = self.phase_fit_loop(self.freq, self.folders['U'], 'u')
        iq_avg = self.iqu_beam_avg(self.freq, self.folders['IQ'], self.npix, self.phase_outputs_iq, self.stop_angle)
        u_avg = self.iqu_beam_avg(self.freq, self.folders['U'], self.npix, self.phase_outputs_u, self.stop_angle)
        self.iqu_avg = {'Q': iq_avg[0],
                        'U': u_avg[1],
                        'I': iq_avg[1]}
        self.beam_matrix = {'QQ': self.iqu_avg['Q'],
                            'QI': self.iqu_avg['I'],
                            'QU': self.iqu_avg['U'],
                            'IQ': self.iqu_avg['Q'] - self.iqu_avg['Q'],
                            'II': self.iqu_avg['Q'],
                            'IU': self.iqu_avg['Q'] - self.iqu_avg['Q'],
                            'UQ': self.iqu_avg['Q'] - self.iqu_avg['Q'],
                            'UI': self.iqu_avg['Q'] - self.iqu_avg['Q'],
                            'UU': self.iqu_avg['Q']}

    def twodunwrapx(self, array):
        phase_offset = np.arange(-1000,1000) * 2.*np.pi
        i = 0
        while (i < (np.shape(array))[0]):
            j = 0
            while (j < (np.shape(array))[1]-1):
                current_val = [array[i,j]]
                next_val = array[i,j+1] + phase_offset
                diff = np.abs(next_val - current_val)
                best = np.where(diff == np.min(diff))
                array[i,j+1] = next_val[best]
                j+=1
            i+=1
        return(array)

    def twodunwrap(self, array):
        xunwraped = self.twodunwrapx(np.transpose(array))
        unwrapped = self.twodunwrapx(np.transpose(xunwraped))
        return(unwrapped)

    def defocus(self, A,B,x0,y0,x,y):
        r = np.sqrt((x-x0)**2 + (y-y0)**2)
        return(A + B* r**2)

    def min_fun(self, p, unwraped_phi,mask):
        A  = p[0]
        B  = p[1]
        x0 = p[2]
        y0 = p[3]
        ## make coordinates for the fit
        x,y = np.meshgrid(np.arange(np.shape(unwraped_phi)[0]),
                      np.arange(np.shape(unwraped_phi)[1]) )

        x = x - np.mean(x)
        y = y - np.mean(y)
        ##
        cleaned_phase = (unwraped_phi-mask* self.defocus(A,B,x0,y0,x,y))
        #plt.imshow(cleaned_phase)
        ##
        return(np.std(cleaned_phase))

    def unpack_csv(self, freqzz, folder):
        '''
        Per-frequency read in csv data and separate real/imaginary a,b,x,y
        '''
        data = genfromtxt(folder+'Rex_%d.csv' % freqzz, delimiter=',', skip_header=1)
        data2 = genfromtxt(folder+'Imx_%d.csv' % freqzz, delimiter=',', skip_header=1)
        data3 = genfromtxt(folder+'Rey_%d.csv' % freqzz, delimiter=',', skip_header=1)
        data4 = genfromtxt(folder+'Imy_%d.csv' % freqzz, delimiter=',', skip_header=1)
        theta = data[:,0]
        Rbax = data[:, 1:362]
        Rbay=data[:,362:723]
        Ibax=data2[:,1:362]
        Ibay=data2[:,362:723]
        Rbbx=data3[:,1:362]
        Rbby=data3[:,362:723]
        Ibbx=data4[:,1:362]
        Ibby=data4[:,362:723]
        return theta, Rbax, Rbay, Ibax, Ibay, Rbbx, Rbby, Ibbx, Ibby

    def first_regrid(self, iqu, theta, Rbax, Rbay, Ibax, Ibay, Rbbx, Rbby, Ibbx, Ibby, mask, folder, freq, ExEy=False):
        '''
        For a single frequency, regrid. Use in phase_output.
        The procedure for phase_output and iqu_mf_so_farfield_mod is the same,
        so can just use the same output. Procedure for ExEy is different.
        '''
        phi_1d = np.linspace(0, 360, 361)
        phi, r = np.meshgrid(phi_1d, theta)
        if ExEy:
            x = r * np.cos(np.radians(phi)) + 90.
            y = r * np.sin(np.radians(phi)) - 90.
            for ii in range(len(theta)):
                if x[ii,jj] > 90.:
                    x[ii,jj] = x[ii,jj] - 90.
                if y[ii,jj] < -90.:
                    y[ii,jj] = y[ii,jj] + 90.
# pick up here, remember to differentiate between iq and u
            image_rax = np.zeros((90, 90))
            image_ray = np.zeros((90, 90))
            image_rbx = np.zeros((90, 90))
            image_rby = np.zeros((90, 90))
            image_iax = np.zeros((90, 90))
            image_iay = np.zeros((90, 90))
            image_ibx = np.zeros((90, 90))
            image_iby = np.zeros((90, 90))
            for ii in range(90):
                for jj in range(90):
                    hits = 0
                    val_d = 0.
                    val_s = 0.
                    val_b = 0.
                    val_di = 0.
                    val_si = 0.
                    val_ai = 0.
                    val_bi = 0.
                    for kk in range(len(theta)):
                        for mm in range(len(phi_1d)):
                            if (x[kk,mm]>=jj and x[kk,mm]<=jj+1) and (y[kk,mm]<=-ii and y[kk,mm]>=-ii-1):
                                hits += 1
                                val_d += Rbax[kk,mm]
                                val_s += Rbay[kk,mm]
                                val_a += Rbbx[kk,mm]
                                val_b += Rbby[kk,mm]
                                val_di += Ibax[kk,mm]
                                val_si += Ibay[kk,mm]
                                val_ai += Ibbx[kk,mm]
                                val_bi += Ibby[kk,mm]
                            if hits != 0:
                                image_rax[ii,jj] = val_d / hits
                                image_ray[ii,jj] = val_s / hits
                                image_rbx[ii,jj] = val_a / hits
                                image_rby[ii,jj] = val_b / hits
                                image_iax[ii,jj] = val_di / hits
                                image_iay[ii,jj] = val_si / hits
                                image_ibx[ii,jj] = val_ai / hits
                                image_iby[ii,jj] = val_bi / hits
        else:
            if iqu == 'u' or iqu == 'U':
                x = r * np.cos(np.radians(phi - 45)) + 45.
                y = r * np.sin(np.radians(phi - 45)) - 45.
            elif iqu == 'iq' or iqu == 'IQ':
                x = r * np.cos(np.radians(phi)) + 45.
                y = r * np.sin(np.radians(phi)) - 45.
            else:
                print('iqu set incorrectly! it is set to ' + str(iqu))
                return
            sum_rbax = np.zeros((90, 90))
            hits_rbax = np.zeros((90, 90))
            sum_rbay = np.zeros((90, 90))
            hits_rbay = np.zeros((90, 90))
            sum_rbbx = np.zeros((90, 90))
            hits_rbbx = np.zeros((90, 90))
            sum_rbby = np.zeros((90, 90))
            hits_rbby = np.zeros((90, 90))
            sum_ibax = np.zeros((90, 90))
            hits_ibax = np.zeros((90, 90))
            sum_ibay = np.zeros((90, 90))
            hits_ibay = np.zeros((90, 90))
            sum_ibbx = np.zeros((90, 90))
            hits_ibbx = np.zeros((90, 90))
            sum_ibby = np.zeros((90, 90))
            hits_ibby = np.zeros((90, 90))

            for kk in range(len(theta)):
                for mm in range(len(phi_1d)):
                    i_x = int(np.floor((x[kk, mm] - 0))) #???
                    i_y = int(np.floor((y[kk, mm] + 90)))
                    if i_x == 90:
                        i_x = 89
                    if i_y == 90:
                        i_y = 89
 #                   print(i_y, i_x)
#                    print(Rbax[kk,mm])
                    sum_rbax[i_y, i_x] += Rbax[kk,mm]
                    hits_rbax[i_y, i_x] += 1.
                    sum_rbay[i_y, i_x] += Rbay[kk,mm]
                    hits_rbay[i_y, i_x] += 1.
                    sum_rbbx[i_y, i_x] += Rbbx[kk,mm]
                    hits_rbbx[i_y, i_x] += 1.
                    sum_rbby[i_y, i_x] += Rbby[kk,mm]
                    hits_rbby[i_y, i_x] += 1.
                    sum_ibax[i_y, i_x] += Ibax[kk,mm]
                    hits_ibax[i_y, i_x] += 1.
                    sum_ibay[i_y, i_x] += Ibay[kk,mm]
                    hits_ibay[i_y, i_x] += 1.
                    sum_ibbx[i_y, i_x] += Ibbx[kk,mm]
                    hits_ibbx[i_y, i_x] += 1.
                    sum_ibby[i_y, i_x] += Ibby[kk,mm]
                    hits_ibby[i_y, i_x] += 1.
            image_rax=np.nan_to_num(np.divide(sum_rbax,hits_rbax))
            image_ray=np.nan_to_num(np.divide(sum_rbay,hits_rbay))
            image_rbx=np.nan_to_num(np.divide(sum_rbbx,hits_rbbx))
            image_rby=np.nan_to_num(np.divide(sum_rbby,hits_rbby))
            image_iax=np.nan_to_num(np.divide(sum_ibax,hits_ibax))
            image_iay=np.nan_to_num(np.divide(sum_ibay,hits_ibay))
            image_ibx=np.nan_to_num(np.divide(sum_ibbx,hits_ibbx))
            image_iby=np.nan_to_num(np.divide(sum_ibby,hits_ibby))

            image_rax=np.fft.fftshift(image_rax*mask)
            image_ray=np.fft.fftshift(image_ray*mask)
            image_rbx=np.fft.fftshift(image_rbx*mask)
            image_rby=np.fft.fftshift(image_rby*mask)
            image_iax=np.fft.fftshift(image_iax*mask)
            image_iay=np.fft.fftshift(image_iay*mask)
            image_ibx=np.fft.fftshift(image_ibx*mask)
            image_iby=np.fft.fftshift(image_iby*mask)

            #Calculate phase and magnitude
            mag_ax=np.sqrt((image_rax**2 + image_iax**2))
            mag_ay=np.sqrt((image_ray**2 + image_iay**2))
            mag_bx=np.sqrt((image_rbx**2 + image_ibx**2))
            mag_by=np.sqrt((image_rby**2 + image_iby**2))
            phi_ax=np.arctan2(image_rax,image_iax)
            phi_ay=np.arctan2(image_ray,image_iay)
            phi_bx=np.arctan2(image_rbx,image_ibx)
            phi_by=np.arctan2(image_rby,image_iby)
            #unwrap the phase of the co-polar beams
            #make mask
            phi_ax[np.where(phi_ax == np.pi)] = 0
            phi_by[np.where(phi_by == np.pi)] = 0
            phi_bx[np.where(phi_bx == np.pi)] = 0
            phi_ay[np.where(phi_ay == np.pi)] = 0
            phi_ax[np.where(phi_ax == -np.pi)] = 0
            phi_by[np.where(phi_by == -np.pi)] = 0
            phi_bx[np.where(phi_bx == -np.pi)] = 0
            phi_ay[np.where(phi_ay == -np.pi)] = 0
            #center
            phi_ax = np.fft.fftshift(phi_ax)
            phi_by = np.fft.fftshift(phi_by)
            phi_ay = np.fft.fftshift(phi_ay)
            phi_bx = np.fft.fftshift(phi_bx)
            #make mask
            mask = np.ones(np.shape(phi_ax))
            mask[np.where(phi_ax ==0)] = 0.

            #subtract off best fit
            x, y = np.meshgrid(np.arange(np.shape(phi_ax)[0]),
                               np.arange(np.shape(phi_ax)[1]) )

            x = x - np.mean(x)
            y = y - np.mean(y)
            best_fit = mask * self.defocus(0.0, 5.763159317368588e-05*freq,0.,0.,x,y) #need to change factor??
        #v1_13_5p6 7.06187276741e-05
        # CQ 5.97682669731e-06
        #5.76686889983e-05 v11_3_500um_deep
        #7.45674589335e-05 v6_22
        #  resid_ax=phi_ax_unwrap-best_fit
        #  resid_by=phi_ax_unwrap-best_fit
            resid_ax=phi_ax-best_fit
            resid_by=phi_by-best_fit
            resid_ay=phi_ay-best_fit
            resid_bx=phi_bx-best_fit
            resid_ax=np.fft.fftshift(resid_ax)
            resid_by=np.fft.fftshift(resid_by)
            resid_ay=np.fft.fftshift(resid_ay)
            resid_bx=np.fft.fftshift(resid_bx)

        out_dict = {'images': {'rax': image_rax,
                               'ray': image_ray,
                               'rbx': image_rbx,
                               'rby': image_rby,
                               'iax': image_iax,
                               'iay': image_iay,
                               'ibx': image_ibx,
                               'iby': image_iby,
                               },
                    'resids': {'ax': resid_ax,
                               'ay': resid_ay,
                               'bx': resid_bx,
                               'by': resid_by,
                               },
                    }

        return out_dict

    def phase_output_perfreq(self, freqzz, folder, image_rax, image_ray, image_rbx, image_rby, image_iax, image_iay, image_ibx, image_iby):
        phase_xx=np.arctan2(image_rax,image_iax)
        phase_yy=np.arctan2(image_rby,image_iby)
        phase_xy=np.arctan2(image_ray,image_iay)
        phase_yx=np.arctan2(image_rbx,image_ibx)

        np.savetxt(folder+'phase/%d_phase_xx.txt' % freqzz,phase_xx)
        np.savetxt(folder+'phase/%d_phase_yy.txt' % freqzz,phase_yy)
        np.savetxt(folder+'phase/%d_phase_xy.txt' % freqzz,phase_xy)
        np.savetxt(folder+'phase/%d_phase_yx.txt' % freqzz,phase_yx)

    def run_phase_output(self, freqs, folder, stop_angle, mask_params, iqu):
        telecentricity = mask_params['telecentricity']
        cen = mask_params['cen']
        center = [cen-telecentricity, cen]
        radius = stop_angle
        Y, X = np.ogrid[:90, :90]
        dist_from_center = np.sqrt((X - center[0])**2 + (Y - center[1])**2)
        mask = (dist_from_center <= radius)*1
        np.savetxt(mask_params['savename'], mask)
        outputs = {}
        for zz in range(len(freqs)):
            #print(freqs[zz])
            theta, Rbax, Rbay, Ibax, Ibay, Rbbx, Rbby, Ibbx, Ibby = self.unpack_csv(freqs[zz], folder)
            first_regrid_out = self.first_regrid(iqu, theta, Rbax, Rbay, Ibax, Ibay, Rbbx, Rbby, Ibbx, Ibby, mask, folder, freqs[zz], ExEy=False)
            images = first_regrid_out['images']
            self.phase_output_perfreq(freqs[zz], folder, images['rax'], images['ray'], images['rbx'], images['rby'], images['iax'], images['iay'], images['ibx'], images['iby'])
            outputs[freqs[zz]] = {'unpack_output': (theta, Rbax, Rbay, Ibax, Ibay, Rbbx, Rbby, Ibbx, Ibby),
                                  'first_regrid_output': first_regrid_out,
                                  }
        return outputs

    def phase_mag(self, first_regrid_output):
        images = first_regrid_output['images']
        mag_ax=np.sqrt((images['rax']**2 + images['iax']**2))
        mag_ay=np.sqrt((images['ray']**2 + images['iay']**2))
        mag_bx=np.sqrt((images['rbx']**2 + images['ibx']**2))
        mag_by=np.sqrt((images['rby']**2 + images['iby']**2))
        phi_ax=np.arctan2(images['rax'],images['iax'])
        phi_ay=np.arctan2(images['ray'],images['iay'])
        phi_bx=np.arctan2(images['rbx'],images['ibx'])
        phi_by=np.arctan2(images['rby'],images['iby'])
        magphi = {'mag': {'mag_ax': mag_ax,
                         'mag_ay': mag_ay,
                         'mag_bx': mag_bx,
                         'mag_by': mag_by,
                        },
                 'phi': {'phi_ax': phi_ax,
                         'phi_ay': phi_ay,
                         'phi_bx': phi_bx,
                         'phi_by': phi_by,
                        },
                  }
        return magphi

    def apply_mag_resid(self, magphi, first_regrid_output, amr_params={'n_pix':180, 'n_half': 90, 'n_edge':180-45}):
        #zero pad, npix must be even! and an integer
        n_pix = amr_params['n_pix']
        n_half = amr_params['n_half']
        n_edge = amr_params['n_edge']

        mags = magphi['mag']
        phis = magphi['phi']

        mag_ax = mags['mag_ax']
        mag_ay = mags['mag_ay']
        mag_bx = mags['mag_bx']
        mag_by = mags['mag_by']

        phi_ax = phis['phi_ax']
        phi_ay = phis['phi_ay']
        phi_bx = phis['phi_bx']
        phi_by = phis['phi_by']

        resid = first_regrid_output
        resid_ax = resid['ax']
        resid_ay = resid['ay']
        resid_bx = resid['bx']
        resid_by = resid['by']

        #print(n_half, n_edge)
        magax = np.zeros((n_pix,n_pix))
        magay = np.zeros((n_pix,n_pix))
        magbx = np.zeros((n_pix,n_pix))
        magby = np.zeros((n_pix,n_pix))
        phiax = np.zeros((n_pix,n_pix))
        phiay = np.zeros((n_pix,n_pix))
        phibx = np.zeros((n_pix,n_pix))
        phiby = np.zeros((n_pix,n_pix))

        magax[0:45,0:45] = mag_ax[0:45,0:45]
        magax[0:45,n_edge:n_pix] = mag_ax[0:45,45:90]
        magax[n_edge:n_pix,0:45] = mag_ax[45:90,0:45]
        magax[n_edge:n_pix,n_edge:n_pix] = mag_ax[45:90,45:90]

        magay[0:45,0:45] = mag_ay[0:45,0:45]
        magay[0:45,n_edge:n_pix] = mag_ay[0:45,45:90]
        magay[n_edge:n_pix,0:45] = mag_ay[45:90,0:45]
        magay[n_edge:n_pix,n_edge:n_pix] = mag_ay[45:90,45:90]

        magbx[0:45,0:45] = mag_bx[0:45,0:45]
        magbx[0:45,n_edge:n_pix] = mag_bx[0:45,45:90]
        magbx[n_edge:n_pix,0:45] = mag_bx[45:90,0:45]
        magbx[n_edge:n_pix,n_edge:n_pix] = mag_bx[45:90,45:90]

        magby[0:45,0:45] = mag_by[0:45,0:45]
        magby[0:45,n_edge:n_pix] = mag_by[0:45,45:90]
        magby[n_edge:n_pix,0:45] = mag_by[45:90,0:45]
        magby[n_edge:n_pix,n_edge:n_pix] = mag_by[45:90,45:90]

        phiax[0:45,0:45] = resid_ax[0:45,0:45]
        phiax[0:45,n_edge:n_pix] = resid_ax[0:45,45:90]
        phiax[n_edge:n_pix,0:45] = resid_ax[45:90,0:45]
        phiax[n_edge:n_pix,n_edge:n_pix] = resid_ax[45:90,45:90]

        phiay[0:45,0:45] = resid_ay[0:45,0:45]
        phiay[0:45,n_edge:n_pix] = resid_ay[0:45,45:90]
        phiay[n_edge:n_pix,0:45] = resid_ay[45:90,0:45]
        phiay[n_edge:n_pix,n_edge:n_pix] = resid_ay[45:90,45:90]

        phibx[0:45,0:45] = resid_bx[0:45,0:45]
        phibx[0:45,n_edge:n_pix] = resid_bx[0:45,45:90]
        phibx[n_edge:n_pix,0:45] = resid_bx[45:90,0:45]
        phibx[n_edge:n_pix,n_edge:n_pix] = resid_bx[45:90,45:90]

        phiby[0:45,0:45] = resid_by[0:45,0:45]
        phiby[0:45,n_edge:n_pix] = resid_by[0:45,45:90]
        phiby[n_edge:n_pix,0:45] = resid_by[45:90,0:45]
        phiby[n_edge:n_pix,n_edge:n_pix] = resid_by[45:90,45:90]

        Bax = magax * np.exp(1j * phiax)
        Bay = magay * np.exp(1j * phiay)
        Bbx = magbx * np.exp(1j * phibx)
        Bby = magby * np.exp(1j * phiby)

        fft_Bax = np.fft.fft2(Bax)
        fft_Bay = np.fft.fft2(Bay)
        fft_Bbx = np.fft.fft2(Bbx)
        fft_Bby = np.fft.fft2(Bby)

        Rbax = np.zeros((n_pix,n_pix))
        Ibax = np.zeros((n_pix,n_pix))
        Rbay = np.zeros((n_pix,n_pix))
        Ibay = np.zeros((n_pix,n_pix))
        Rbbx = np.zeros((n_pix,n_pix))
        Ibbx = np.zeros((n_pix,n_pix))
        Rbby = np.zeros((n_pix,n_pix))
        Ibby = np.zeros((n_pix,n_pix))

        Rbax[0:n_half,0:n_half] = np.real(fft_Bax[n_half:n_pix,n_half:n_pix])
        Rbax[n_half:n_pix,n_half:n_pix] = np.real(fft_Bax[0:n_half,0:n_half])
        Rbax[0:n_half,n_half:n_pix] = np.real(fft_Bax[n_half:n_pix,0:n_half])
        Rbax[n_half:n_pix,0:n_half] = np.real(fft_Bax[0:n_half,n_half:n_pix])

        Ibax[0:n_half,0:n_half] = np.imag(fft_Bax[n_half:n_pix,n_half:n_pix])
        Ibax[n_half:n_pix,n_half:n_pix] = np.imag(fft_Bax[0:n_half,0:n_half])
        Ibax[0:n_half,n_half:n_pix] = np.imag(fft_Bax[n_half:n_pix,0:n_half])
        Ibax[n_half:n_pix,0:n_half] = np.imag(fft_Bax[0:n_half,n_half:n_pix])

        Rbay[0:n_half,0:n_half] = np.real(fft_Bay[n_half:n_pix,n_half:n_pix])
        Rbay[n_half:n_pix,n_half:n_pix] = np.real(fft_Bay[0:n_half,0:n_half])
        Rbay[0:n_half,n_half:n_pix] = np.real(fft_Bay[n_half:n_pix,0:n_half])
        Rbay[n_half:n_pix,0:n_half] = np.real(fft_Bay[0:n_half,n_half:n_pix])

        Ibay[0:n_half,0:n_half] = np.imag(fft_Bay[n_half:n_pix,n_half:n_pix])
        Ibay[n_half:n_pix,n_half:n_pix] = np.imag(fft_Bay[0:n_half,0:n_half])
        Ibay[0:n_half,n_half:n_pix] = np.imag(fft_Bay[n_half:n_pix,0:n_half])
        Ibay[n_half:n_pix,0:n_half] = np.imag(fft_Bay[0:n_half,n_half:n_pix])

        Rbbx[0:n_half,0:n_half] = np.real(fft_Bbx[n_half:n_pix,n_half:n_pix])
        Rbbx[n_half:n_pix,n_half:n_pix] = np.real(fft_Bbx[0:n_half,0:n_half])
        Rbbx[0:n_half,n_half:n_pix] = np.real(fft_Bbx[n_half:n_pix,0:n_half])
        Rbbx[n_half:n_pix,0:n_half] = np.real(fft_Bbx[0:n_half,n_half:n_pix])

        Ibbx[0:n_half,0:n_half] = np.imag(fft_Bbx[n_half:n_pix,n_half:n_pix])
        Ibbx[n_half:n_pix,n_half:n_pix] = np.imag(fft_Bbx[0:n_half,0:n_half])
        Ibbx[0:n_half,n_half:n_pix] = np.imag(fft_Bbx[n_half:n_pix,0:n_half])
        Ibbx[n_half:n_pix,0:n_half] = np.imag(fft_Bbx[0:n_half,n_half:n_pix])

        Rbby[0:n_half,0:n_half] = np.real(fft_Bby[n_half:n_pix,n_half:n_pix])
        Rbby[n_half:n_pix,n_half:n_pix] = np.real(fft_Bby[0:n_half,0:n_half])
        Rbby[0:n_half,n_half:n_pix] = np.real(fft_Bby[n_half:n_pix,0:n_half])
        Rbby[n_half:n_pix,0:n_half] = np.real(fft_Bby[0:n_half,n_half:n_pix])

        Ibby[0:n_half,0:n_half] = np.imag(fft_Bby[n_half:n_pix,n_half:n_pix])
        Ibby[n_half:n_pix,n_half:n_pix] = np.imag(fft_Bby[0:n_half,0:n_half])
        Ibby[0:n_half,n_half:n_pix] = np.imag(fft_Bby[n_half:n_pix,0:n_half])
        Ibby[n_half:n_pix,0:n_half] = np.imag(fft_Bby[0:n_half,n_half:n_pix])

        ret_dict = {'mags': {'ax': magax,
                             'ay': magay,
                             'bx': magbx,
                             'by': magby,
                            },
                    'phis': {'ax': phiax,
                             'ay': phiay,
                             'bx': phibx,
                             'by': phiby,
                            },
                    'reim': {'Rbax': Rbax,
                             'Rbay': Rbay,
                             'Ibax': Ibax,
                             'Ibay': Ibay,
                             'Rbbx': Rbbx,
                             'Rbby': Rbby,
                             'Ibbx': Ibbx,
                             'Ibby': Ibby,
                            },
                   }
        return ret_dict

    def iqu_farfield_perfreq(self, phase_out_return, freq, stop_angle):
#        theta, Rbax, Rbay, Ibax, Ibay, Rbbx, Rbby, Ibbx, Ibby = phase_out_return[freq]['unpack_output']
#        print('shape of Rbax: ' + str(np.shape(Rbax)))
        first_regrid_output = phase_out_return[freq]['first_regrid_output']
#        print(first_regrid_output.keys())
        magphi = self.phase_mag(first_regrid_output)
        resid = first_regrid_output['resids']
#        print('npix = ' + str(self.npix)) 
        ret_dict = self.apply_mag_resid(magphi, resid, amr_params={'n_pix':self.npix, 'n_half': int(self.npix/2), 'n_edge':self.npix-45})
        #print(ret_dict)
        Rbax = ret_dict['reim']['Rbax']
        Rbay = ret_dict['reim']['Rbay']
        Ibax = ret_dict['reim']['Ibax']
        Ibay = ret_dict['reim']['Ibay']
        Rbbx = ret_dict['reim']['Rbbx']
        Rbby = ret_dict['reim']['Rbby']
        Ibbx = ret_dict['reim']['Ibbx']
        Ibby = ret_dict['reim']['Ibby']
        sigma = 0.5 * (Rbax**2 + Ibax**2 + Rbay**2 + Ibay**2 - Rbbx**2 - Ibbx**2 - Rbby**2 - Ibby**2)
        delta = 0.5 * (Rbax**2 + Ibax**2 - Rbay**2 - Ibay**2 - Rbbx**2 - Ibbx**2 + Rbby**2 + Ibby**2)
        a = Rbax * Rbay + Ibax * Ibay - Rbbx * Rbby - Ibbx * Ibby
        b = Ibay * Rbax - Ibax * Rbay + Ibbx * Rbby - Ibby * Rbbx
        #Normalize beam
        s = sigma / np.max(delta)
        d = delta / np.max(delta)
        a_n = a / np.max(delta)
        b_n = b / np.max(delta)
        return s, d, a_n, b_n, ret_dict

    def iqu_beam_avg(self, freqs, folder, n_pix, phase_output_return, stop_angle):
        Q_low = np.zeros((n_pix, n_pix))
        U_low = np.zeros((n_pix, n_pix))
        I_low = np.zeros((n_pix, n_pix))
        V_low = np.zeros((n_pix, n_pix))
        for zz in range(len(freqs)):
            s, d, a_n, b_n, ret_dict = self.iqu_farfield_perfreq(phase_output_return, freqs[zz], stop_angle)
            np.savetxt(folder + '%d_GHz_Q.txt' % freqs[zz], d)
            np.savetxt(folder + '%d_GHz_I.txt' % freqs[zz], s)
            np.savetxt(folder + '%d_GHz_U.txt' % freqs[zz], a_n)
            np.savetxt(folder + '%d_GHz_V.txt' % freqs[zz], b_n)
            
            Q_low += d
            I_low += s
            U_low += a_n
            V_low += b_n
        Q_avg = Q_low / np.max(Q_low)
        I_avg = I_low / np.max(Q_low)
        U_avg = U_low / np.max(Q_low)
        V_avg = V_low / np.max(Q_low)
        return Q_avg, I_avg, U_avg, V_avg

    def phase_fit_loop(self, freq, folder, plot=True):
        N = len(freq)
        a_fit = np.zeros(N)
        b_fit = np.zeros(N)
        x0_fit = np.zeros(N)
        y0_fit = np.zeros(N)
        for zz in range(N):
            phi = np.loadtxt(folder+"phase/%d_phase_xx.txt" % freq[zz])
            phi[np.where(phi == -np.pi)] = 0
            phi[np.where(phi == np.pi)] = 0
            phi = np.fft.fftshift(phi)

            mask = np.ones(np.shape(phi))
            mask[np.where(phi == 0)] = 0

            unwrapped_phi = self.twodunwrap(phi)
            unwrapped_phi = unwrapped_phi[0,0] # there might be a line missing here
            unwrapped_phi = mask * unwrapped_phi

            best_fit = opt.fmin_powell(self.min_fun, x0=(np.min(unwrapped_phi), 0.014*freq[zz]/150., 0., 0.), args=(unwrapped_phi, mask))
#            print(freq[zz], best_fit)
            a_fit[zz] = best_fit[0]
            b_fit[zz] = best_fit[1]
            x0_fit[zz] = best_fit[2]
            y0_fit[zz] = best_fit[3]
#        print("a: ", str(np.mean(a_fit)), str(np.median(a_fit)))
        b_fit_scaled = b_fit / freq
#        print("b_scaled: ", str(np.mean(b_fit_scaled)), str(np.median(b_fit_scaled)))
#        print("x0: ", str(np.mean(x0_fit)), str(np.median(x0_fit)))
#        print("y0: ", str(np.mean(y0_fit)), str(np.median(y0_fit)))

        if plot:
            plt.plot(freq, b_fit_scaled)
            plt.ylabel('b_fit_scaled')
            plt.xlabel('freq')
            plt.title('b_fit_scaled vs. freq')
            plt.show()

            plt.plot(freq, a_fit)
            plt.ylabel('a_fit')
            plt.xlabel('freq')
            plt.title('a_fit vs. freq')
            plt.show()
        fits = {'a': {'full': a_fit, 'mean': np.mean(a_fit), 'median': np.median(a_fit)},
                'b_scaled': {'full': b_fit_scaled, 'mean': np.mean(b_fit_scaled), 'median': np.median(b_fit_scaled)},
                'x0': {'full': x0_fit, 'mean': np.mean(x0_fit), 'median': np.median(x0_fit)},
                'y0': {'full': y0_fit, 'mean': np.mean(y0_fit), 'median': np.median(y0_fit)}
               }
        return fits

    def iqu_farfield_all(self, freq, phase_out_return, stop_angle):
        N = len(freq)
        iqu_results = {}
        for zz in range(N):
            s, d, a_n, b_n, magresid = iqu_farfield_perfreq(phase_out_return, freq[zz], stop_angle)
            iqu_results[freq[zz]] = {'s': s,
                                     'd': d,
                                     'a_n': a_n,
                                     'b_n': b_n,
                                     'magresid': magresid,
                                     }
        return iqu_results
