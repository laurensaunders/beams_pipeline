import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import os, sys
path_prefix = '/Users/laurensaunders/' # change this to give your code the path to map_multi_tool on your system
sys.path.append(path_prefix)
import map_multi_tool.MMTModules as mmt
beams_pipeline_prefix = path_prefix # change this if the path to beams_pipeline is not the same as to map_multi_tool

def get_mmt_leakage(Beam, spectrum_params):
    """
    Wrapper for using one of the beam classes (AnalyticBeam or HFSSBeam)
    with MMT to find the leakage spectra.

    Inputs
    ------
    Beam (class inst): instance of AnalyticBeam or HFSSBeam with correct params
    spectrum_params (dict): dictionary of parameters characterizing the spectra
        you want to find. Dictionary keys must include:
        'sky_decomp' (list): decomposition into IQU components. Usually use [1,0,0]
        'delta_ell' (int/float): minimum ell bin size (may be changed if too small)
        'ell_max' (int/float): maximum ell value for calculating spectra
        'choose_normalization' (str or int): which spectrum you want to normalize to.
            Options are 'TT', 'EE', 'BB', or 0 (each normalized to its own max)

    Outputs
    -------
    binned_ell (np.array): array of binned ell values
    binned_spectra (dict): dictionary of binned spectra arrays, with keys 'TT', 'EE', 'BB', 'TE', 'TB', 'EB', 'beam_type'
    """

    beam_params = Beam.params
    beam_matrix = Beam.beam_matrix

    binned_ell, binned_spectra = mmt.get_leakage_spectra(beam_matrix, beam_params['pixel_size'], beam_params['N'], beam_params['beam_fwhm'], spectrum_params['sky_decomp'], spectrum_params['delta_ell'], spectrum_params['ell_max'], spectrum_params['choose_normalization'])
    binned_spectra['beam_type'] = beam_params['beam_type']
    return binned_ell, binned_spectra

def sample_cmb(binned_ell, beams_pipeline_prefix=beams_pipeline_prefix):
    """
    Interpolate CMB TT, EE, and BB spectra to the binned_ell values.

    Inputs
    ------
    binned_ell (np.array): array of binned ell values from get_mmt_leakage
    cmb_file_prefix (str): path to beams_pipeline. If get_mmt_leakage worked, you shouldn't need to do anything here.
    """
    t_cmb = np.loadtxt(beams_pipeline_prefix + 'beams_pipeline/cmb_spectra/CMB_t.txt')
    t_ell_cmb = np.loadtxt(beams_pipeline_prefix + 'beams_pipeline/cmb_spectra/CMB_t_ell.txt')
    e_cmb = np.loadtxt(beams_pipeline_prefix + 'beams_pipeline/cmb_spectra/CMB_e.txt')
    e_ell_cmb = np.loadtxt(beams_pipeline_prefix + 'beams_pipeline/cmb_spectra/CMB_e_ell.txt')
    b_cmb = np.loadtxt(beams_pipeline_prefix + 'beams_pipeline/cmb_spectra/CMB_b.txt')
    b_ell_cmb = np.loadtxt(beams_pipeline_prefix + 'beams_pipeline/cmb_spectra/CMB_b_ell.txt')

    cmb_t_fn = interp1d(t_ell_cmb, t_cmb)
    binned_cmb_t = cmb_t_fn(binned_ell)
    cmb_e_fn = interp1d(e_ell_cmb, e_cmb)
    binned_cmb_e = cmb_e_fn(binned_ell)
#    cmb_b_fn = interp1d(b_ell_cmb, b_cmb)
#    binned_cmb_b = cmb_b_fn(binned_ell)

    CMB_binned = {'tt_ell': binned_ell,
                  'ee_ell': binned_ell,
                  'bb_ell': b_ell_cmb,
                  'TT': binned_cmb_t,
                  'EE': binned_cmb_e,
                  'BB': b_cmb}
    
    return CMB_binned

def make_leakage_plot(binned_ell, binned_spectra, title, leakage, CMB_compare=True, showBB=False, beams_pipeline_prefix=beams_pipeline_prefix, savefig=False):
    """
    Simple plotter for binned leakage spectra.

    Inputs
    ______
    binned_ell (np.array): ell values output by get_mmt_leakage
    binned_spectra (dict): dictionary of leakage spectra output by get_mmt_leakage
    title (str): title for your plot
    leakage (list): list of which window functions you want to display
    CMB_compare (bool): choose whether to plot CMB TT and EE spectra along with your
                        leakage spectrum. Default is True
    showBB (bool): choose whether to display CMB BB spectrum along with your leakage
                   spectra. Default is False (auto set to True if CMB_compare=True and 
                   'TB' or 'EB' in leakage)
    beams_pipeline_prefix (str): See docstring for sample_cmb.
    savefig (str or False): If you want to save the plot, use savefig='/full/path/to/save/file.'
                            If not, set to False. Default is False.
    """
    if binned_spectra['beam_type'] == 'analytic':
        available = ['TT', 'TE', 'TB', 'EB']
    elif binned_spectra['beam_type'] == 'hfss':
        available = ['TT', 'TE']
    else:
        print('incorrect beam type')
        return
    CMB_binned = sample_cmb(binned_ell, beams_pipeline_prefix)
    plt.figure()
    if CMB_compare:
        plt.loglog(CMB_binned['tt_ell'], CMB_binned['TT'], label='CMB TT')
        plt.loglog(CMB_binned['ee_ell'], CMB_binned['EE'], label='CMB EE')
        if 'TE' in leakage and 'TE' in available:
            plt.loglog(binned_ell[np.where(binned_spectra['EE']>1e-120)], binned_spectra['EE'][np.where(binned_spectra['EE']>1e-120)] * CMB_binned['TT'][np.where(binned_spectra['EE']>1e-120)], label='T->E leakage', linestyle='-', marker='')
        if 'TT' in leakage and 'TT' in available:
            plt.semilogy(binned_ell[np.where(binned_spectra['TT']>1e-120)], binned_spectra['TT'][np.where(binned_spectra['TT']>1e-120)] * CMB_binned['TT'][np.where(binned_spectra['TT']>1e-120)], label='TT Window Function', linestyle='-', marker='')
        if 'TB' in leakage and 'TB' in available:
            showBB = True
            plt.semilogy(binned_ell[np.where(binned_spectra['TT']>1e-120)], binned_spectra['TB'][np.where(binned_spectra['TT']>1e-120)] * CMB_binned['TT'][np.where(binned_spectra['TT']>1e-120)], label='T->B leakage', linestyle='-', marker='')
        elif 'TB' in leakage and 'TB' not in available:
            print("'TB' not available for this beam type")
        if 'EB' in leakage and 'EB' in available:
            showBB = True
            plt.semilogy(binned_ell[np.where(binned_spectra['EB']>1e-120)], binned_spectra['EB'][np.where(binned_spectra['EB']>1e-120)] * CMB_binned['EE'][np.where(binned_spectra['EB']>1e-120)], label='E->B leakage', linestyle='-', marker='')
        elif 'EB' in leakage and 'EB' not in available:
            print("'EB' not available for this beam type")
    else:
        if 'TE' in leakage and 'TE' in available:
            plt.semilogy(binned_ell[np.where(binned_spectra['EE']>1e-120)], binned_spectra['EE'][np.where(binned_spectra['EE']>1e-120)], label='T->E leakage', linestyle='-', marker='')
        if 'TT' in leakage and 'TT' in available:
            plt.semilogy(binned_ell[np.where(binned_spectra['TT']>1e-120)], binned_spectra['TT'][np.where(binned_spectra['TT']>1e-120)], label='TT Window Function', linestyle='-', marker='')
        if 'TB' in leakage and 'TB' in available:
            plt.semilogy(binned_ell[np.where(binned_spectra['TB']>1e-120)], binned_spectra['TB'][np.where(binned_spectra['TB']>1e-120)], label='T->B leakage', linestyle='-', marker='')
        elif 'TB' in leakage and 'TB' not in available:
            print("'TB' not available for this beam type")
        if 'EB' in leakage and 'EB' in available:
            plt.semilogy(binned_ell[np.where(binned_spectra['EB']>1e-120)], binned_spectra['EB'][np.where(binned_spectra['EB']>1e-120)], label='E->B leakage', linestyle='-', marker='')
        elif 'EB' in leakage and 'EB' not in available:
            print("'EB' not available for this beam type")
    if showBB:
        plt.loglog(CMB_binned['bb_ell'], CMB_binned['BB'], label='CMB BB')
    plt.xlabel('$\ell$')
    plt.ylabel('$C_{\ell}$')
    plt.legend()
    plt.xlim(min(binned_ell), max(binned_ell))
  #  plt.ylim(1e-12, 1e-5)
    plt.title(title)
    if savefig:
        plt.savefig(savefig)
    plt.show()
