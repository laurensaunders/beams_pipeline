import numpy as np
import matplotlib.pyplot as plt

def make_2d_elliptical_beam(N, pix_size, beam_size_fwhm, ellipt, theta=np.pi/4, offset_x=0.0, offset_y=0.0):
    '''
    Alteration of make_2d_gaussian_beam to make elliptical beams
    '''
    N = int(N)
    ones = np.ones(N)
    inds = (np.arange(N) + 0.5 - N / 2.) * pix_size
    X = np.outer(ones, inds)
    Y = np.transpose(X)
    sigma_x = beam_size_fwhm * (1 + ellipt)
    sigma_y = beam_size_fwhm * (1 - ellipt)
    gaussian = np.exp(-0.5 * (((X - offset_x) / sigma_x) ** 2 + ((Y - offset_y) / sigma_y) ** 2))
    gaussian = gaussian / np.sum(gaussian)
    return gaussian

def make_2d_elliptical_beam_rot(N, pix_size, beam_size_fwhm, ellipt, theta=-np.pi/4, offset_x=0.0, offset_y=0.0):
    '''
    Alteration of make_2d_gaussian_beam to make elliptical beams
    '''
    N = int(N)
    ones = np.ones(N)
    inds = (np.arange(N) + 0.5 - N / 2.) * pix_size
    X = np.outer(ones, inds)
    Y = np.transpose(X)
    x = X - offset_x
    y = Y - offset_y
    sigma_x = beam_size_fwhm * (1 + ellipt)
    sigma_y = beam_size_fwhm * (1 - ellipt)
    a = (np.cos(theta) ** 2 / (2 * sigma_x ** 2)) + (np.sin(theta) ** 2 / (2 * sigma_y ** 2))
    b = (np.sin(2 * theta) / (4 * sigma_y ** 2)) - (np.sin(2 * theta) / (4 * sigma_x ** 2))
#    b = 0
    c = (np.sin(theta) ** 2 / (2 * sigma_x ** 2)) + (np.cos(theta) ** 2 / (2 * sigma_y ** 2))
    gaussian = np.exp(-((a * x ** 2) + (2 * b * x * y) + c * y ** 2))
    gaussian = gaussian / np.sum(gaussian)
    return gaussian

def make_zeros(N, pix_size):
    '''
    Alteration of make_2d_gaussian_beam to make elliptical beams
    '''
    N = int(N)
    ones = np.ones(N)
    inds = (np.arange(N) + 0.5 - N / 2.) * pix_size
    X = np.outer(ones, inds)
    return X - X

def make_betas(N, pix_size, beam_size_fwhm, ellipt):
    beta_by = make_2d_elliptical_beam(N, pix_size, beam_size_fwhm, ellipt)
    beta_ax = np.transpose(beta_by)
    beta_bx = make_2d_elliptical_beam_rot(N, pix_size, beam_size_fwhm, ellipt, theta=np.pi/4)
    beta_ay = make_2d_elliptical_beam_rot(N, pix_size, beam_size_fwhm, ellipt, theta=-np.pi/4)
    return beta_ax, beta_ay, beta_bx, beta_by

def make_beam_params(beta_ax, beta_ay, beta_bx, beta_by):
    sig = 0.5 * (beta_ax **2 - beta_by ** 2)
    delta = 0.5 * (beta_ax ** 2 + beta_by ** 2)
    s = sig / np.max(delta)
    d = delta / np.max(delta)

    a = 0.5 * (beta_bx ** 2 - beta_ay ** 2)
    b = 0.5 * (beta_ay ** 2 + beta_bx ** 2)

    a_n = a / np.max(delta)
    b_n = b / np.max(delta)

    return s, d, a_n, b_n

def make_beam_matrix(s, d, a_n, zeros):
    beam_matrix = {}
    beam_matrix['II'] = d
    beam_matrix['IQ'] = zeros
    beam_matrix['IU'] = zeros
    beam_matrix['QI'] = s
    beam_matrix['QQ'] = d
    beam_matrix['QU'] = a_n # MMT says this should be UI
    beam_matrix['UI'] = zeros # MMT says this should be QU
    beam_matrix['UQ'] = zeros
    beam_matrix['UU'] = d
    return beam_matrix

def make_3x3_plots(N, beam_size_fwhm, s, d, a_n, savefig=None):
    fig, ax = plt.subplots(3,3, figsize=(20,20))
    ulim = (N / 2) + (beam_size_fwhm * 15)
    llim = (N / 2) - (beam_size_fwhm * 15)
    z = make_zeros(N, pix_size)
    
    vmin = -1
    vmax = 1
    
    cmap1 = 'magma'
    cmap2 = 'viridis'
    
    imii = ax[0,0].imshow(beam_matrix['II'], vmin=0, vmax=vmax, cmap=cmap1)
    ax[0,0].set_title('II')
    ax[0,0].set_ylim(ulim, llim)
    ax[0,0].set_xlim(llim, ulim)
    plt.colorbar(imii, ax=ax[0,0])
    
    imqi = ax[1,0].imshow(beam_matrix['QI'], vmin=0.05*vmin, vmax=0.05*vmax, cmap=cmap2)
    ax[1,0].set_title('I -> Q')
    ax[1,0].set_ylim(ulim, llim)
    ax[1,0].set_xlim(llim, ulim)
    plt.colorbar(imqi, ax=ax[1,0])
    
    imui = ax[2,0].imshow(beam_matrix['UI'], vmin=0.05*vmin, vmax=0.05*vmax, cmap=cmap2)
    ax[2,0].set_title('I -> U')
    ax[2,0].set_ylim(ulim, llim)
    ax[2,0].set_xlim(llim, ulim)
    plt.colorbar(imui, ax=ax[2,0])
    
    imiq = ax[0,1].imshow(beam_matrix['IQ'], vmin=0.05*vmin, vmax=0.05*vmax, cmap=cmap2)
    ax[0,1].set_title('Q -> I')
    ax[0,1].set_ylim(ulim, llim)
    ax[0,1].set_xlim(llim, ulim)
    plt.colorbar(imiq, ax=ax[0,1])
    
    imqq = ax[1,1].imshow(beam_matrix['QQ'], vmin=0, vmax=vmax, cmap=cmap1)
    ax[1,1].set_title('QQ')
    ax[1,1].set_ylim(ulim, llim)
    ax[1,1].set_xlim(llim, ulim)
    plt.colorbar(imqq, ax=ax[1,1])
    
    imuq = ax[2,1].imshow(beam_matrix['UQ'], vmin=0.05*vmin, vmax=0.05*vmax, cmap=cmap2)
    ax[2,1].set_title('Q -> U')
    ax[2,1].set_ylim(ulim, llim)
    ax[2,1].set_xlim(llim, ulim)
    plt.colorbar(imuq, ax=ax[2,1])
    
    imiu = ax[0,2].imshow(beam_matrix['IU'], vmin=0.05*vmin, vmax=0.05*vmax, cmap=cmap2)
    ax[0,2].set_title('U -> I')
    ax[0,2].set_ylim(ulim, llim)
    ax[0,2].set_xlim(llim, ulim)
    plt.colorbar(imiu, ax=ax[0,2])
    
    imqu = ax[1,2].imshow(beam_matrix['QU'], vmin=0.05*vmin, vmax=0.05*vmax, cmap=cmap2)
    ax[1,2].set_title('U -> Q')
    ax[1,2].set_ylim(ulim, llim)
    ax[1,2].set_xlim(llim, ulim)
    plt.colorbar(imqu, ax=ax[1,2])
    
    imuu = ax[2,2].imshow(beam_matrix['UU'], vmin=0, vmax=vmax, cmap=cmap1)
    ax[2,2].set_title('UU')
    ax[2,2].set_ylim(ulim, llim)
    ax[2,2].set_xlim(llim, ulim)
    plt.colorbar(imuu, ax=ax[2,2])
    
    if savefig:
        plt.savefig(savefig)
    
    plt.show()
