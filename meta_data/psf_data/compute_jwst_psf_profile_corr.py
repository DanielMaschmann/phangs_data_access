"""
Script to compute profile fitting correction values for JWST
"""

import os.path
import pickle

import numpy as np
import matplotlib.pyplot as plt

from phangs_data_access import phot_tools, phangs_info, phys_params

from astropy.convolution import Gaussian2DKernel
from astropy.convolution import convolve


miri_band_list = phys_params.miri_bands
nircam_band_list = phys_params.nircam_bands

miri_psf_correction_dict = {}

for band in miri_band_list:
    print(band)
    psf_dict = phot_tools.PSFTools.load_obs_psf_dict(band=band, instrument='miri')
    data = psf_dict['over_sampled_psf']
    psf_fwhm_pix = psf_dict['gaussian_fwhm'] / psf_dict['pixel_scale_psf_over_sampled']
    psf_std_pix = psf_dict['gaussian_std'] / psf_dict['pixel_scale_psf_over_sampled']


    # get maximal convolution
    max_convolution_for_psf_img = np.min(data.shape)/10
    # check if this is feasible:
    if max_convolution_for_psf_img > (10 * psf_std_pix):
        max_convolution = 10 * psf_std_pix
    else:
        max_convolution = max_convolution_for_psf_img

    list_convolutions = np.linspace(0, max_convolution, 20)

    measured_std_values = []
    measured_ee_value = []

    for conv in list_convolutions:
        x_stddev = conv
        y_stddev = conv
        theta = 0
        # now convolve data with a gaussian
        if conv == 0:
            data_convolve = data
        else:
            kernel = Gaussian2DKernel(x_stddev=x_stddev, y_stddev=y_stddev, theta=theta)
            data_convolve = convolve(data, kernel)


        # norm to 1
        data_convolve = data_convolve/np.sum(data_convolve)

        data_err = np.ones(data_convolve.shape) * np.max(data_convolve) / 100

        dao_src = phot_tools.SrcTools.detect_star_like_src_in_band_cutout(data=data,
                                                                          detection_threshold=np.max(data) / 2,
                                                                          psf_fwhm_pix=psf_fwhm_pix)
        profile_dict = phot_tools.ProfileTools.compute_axis_profiles(data=data_convolve,
                                                                     x_pos=dao_src['xcentroid'].value[0],
                                                                     y_pos=dao_src['ycentroid'].value[0], n_slits=12,
                                                                     err=data_err)
        # rad_profile_dict, std_pix, data, data_err, x_pos, y_pos, upper_sig_fact=10
        morph_dict = phot_tools.ProfileTools.measure_morph_photometry(rad_profile_dict=profile_dict,
                                                                      std_pix=psf_std_pix,
                                                                      data=data_convolve, data_err=data_err,
                                                                      x_pos=dao_src['xcentroid'].value[0],
                                                                      y_pos=dao_src['ycentroid'].value[0],
                                                                      upper_sig_fact=20)
        measured_std_values.append(morph_dict['mean_sig'] * psf_dict['pixel_scale_psf_over_sampled'])
        measured_ee_value.append(morph_dict['flux'] / np.sum(data_convolve))

    miri_psf_correction_dict.update({band: {'measured_std_values': measured_std_values, 'measured_ee_value': measured_ee_value}})

    plt.plot(measured_std_values, measured_ee_value, label=band)
plt.legend()
plt.xlabel('measured $\sigma$, [arcsec]')
plt.ylabel('measured EE, [arcsec]')
plt.savefig('plot_output/miri_gaussian_ee_corr_fraction.png')
plt.close()

# save dictionary
if not os.path.isdir('data_output'):
    os.makedirs('data_output')

with open('data_output/miri_psf_correction_dict.pickle', 'wb') as file_name:
    pickle.dump(miri_psf_correction_dict, file_name)



nircam_psf_correction_dict = {}

for band in nircam_band_list:
    print(band)
    psf_dict = phot_tools.PSFTools.load_obs_psf_dict(band=band, instrument='nircam')
    data = psf_dict['over_sampled_psf']
    psf_fwhm_pix = psf_dict['gaussian_fwhm'] / psf_dict['pixel_scale_psf_over_sampled']
    psf_std_pix = psf_dict['gaussian_std'] / psf_dict['pixel_scale_psf_over_sampled']


    # get maximal convolution
    max_convolution_for_psf_img = np.min(data.shape)/10
    # check if this is feasible:
    if max_convolution_for_psf_img > (10 * psf_std_pix):
        max_convolution = 10 * psf_std_pix
    else:
        max_convolution = max_convolution_for_psf_img

    list_convolutions = np.linspace(0, max_convolution, 20)

    measured_std_values = []
    measured_ee_value = []

    for conv in list_convolutions:
        x_stddev = conv
        y_stddev = conv
        theta = 0
        # now convolve data with a gaussian
        if conv == 0:
            data_convolve = data
        else:
            kernel = Gaussian2DKernel(x_stddev=x_stddev, y_stddev=y_stddev, theta=theta)
            data_convolve = convolve(data, kernel)


        # norm to 1
        data_convolve = data_convolve/np.sum(data_convolve)

        data_err = np.ones(data_convolve.shape) * np.max(data_convolve) / 100

        dao_src = phot_tools.SrcTools.detect_star_like_src_in_band_cutout(data=data,
                                                                          detection_threshold=np.max(data) / 2,
                                                                          psf_fwhm_pix=psf_fwhm_pix)
        profile_dict = phot_tools.ProfileTools.compute_axis_profiles(data=data_convolve,
                                                                     x_pos=dao_src['xcentroid'].value[0],
                                                                     y_pos=dao_src['ycentroid'].value[0], n_slits=12,
                                                                     err=data_err)
        # rad_profile_dict, std_pix, data, data_err, x_pos, y_pos, upper_sig_fact=10
        morph_dict = phot_tools.ProfileTools.measure_morph_photometry(rad_profile_dict=profile_dict,
                                                                      std_pix=psf_std_pix,
                                                                      data=data_convolve, data_err=data_err,
                                                                      x_pos=dao_src['xcentroid'].value[0],
                                                                      y_pos=dao_src['ycentroid'].value[0],
                                                                      upper_sig_fact=20)
        measured_std_values.append(morph_dict['mean_sig'] * psf_dict['pixel_scale_psf_over_sampled'])
        measured_ee_value.append(morph_dict['flux'] / np.sum(data_convolve))

    nircam_psf_correction_dict.update({band: {'measured_std_values': measured_std_values, 'measured_ee_value': measured_ee_value}})

    plt.plot(measured_std_values, measured_ee_value, label=band)
plt.legend()
plt.xlabel('measured $\sigma$, [arcsec]')
plt.ylabel('measured EE, [arcsec]')
plt.savefig('plot_output/nircam_gaussian_ee_corr_fraction.png')
plt.close()

# save dictionary
if not os.path.isdir('data_output'):
    os.makedirs('data_output')

with open('data_output/nircam_psf_correction_dict.pickle', 'wb') as file_name:
    pickle.dump(nircam_psf_correction_dict, file_name)


