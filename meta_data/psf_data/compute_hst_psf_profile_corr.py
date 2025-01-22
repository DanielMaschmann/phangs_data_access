"""
Script to compute profile fitting correction values for HST
"""

import os.path
import pickle

import numpy as np
import matplotlib.pyplot as plt

from phangs_data_access import phot_tools, phangs_info

from astropy.convolution import Gaussian2DKernel
from astropy.convolution import convolve


acs_wfc_band_list = phangs_info.acs_wfc_psf_band_list
wfc3_uv_band_list = phangs_info.wfc3_uv_psf_band_list
wfc3_ir_band_list = phangs_info.wfc3_ir_psf_band_list


acs_wfc_psf_correction_dict = {}

for band in acs_wfc_band_list:
    print(band)
    psf_dict = phot_tools.PSFTools.load_obs_psf_dict(band=band, instrument='acs')
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
        # print('conv ', conv)
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
        # get the center
        dao_src = phot_tools.SrcTools.detect_star_like_src_in_band_cutout(data=data,
                                                                          detection_threshold=np.max(data) / 2,
                                                                          psf_fwhm_pix=psf_fwhm_pix)
        profile_dict = phot_tools.ProfileTools.compute_axis_profiles(data=data_convolve,
                                                                     x_pos=dao_src['xcentroid'].value[0],
                                                                     y_pos=dao_src['ycentroid'].value[0], n_slits=12,
                                                                     err=data_err)
        morph_dict = phot_tools.ProfileTools.measure_morph_photometry(rad_profile_dict=profile_dict,
                                                                      std_pix=psf_std_pix,
                                                                      data=data_convolve, data_err=data_err,
                                                                      x_pos=dao_src['xcentroid'].value[0],
                                                                      y_pos=dao_src['ycentroid'].value[0],
                                                                      upper_sig_fact=20)
        measured_std_values.append(morph_dict['mean_sig'] * psf_dict['pixel_scale_psf_over_sampled'])
        measured_ee_value.append(morph_dict['flux'] / np.sum(data_convolve))

    acs_wfc_psf_correction_dict.update({band: {'measured_std_values': measured_std_values, 'measured_ee_value': measured_ee_value}})

    plt.plot(measured_std_values, measured_ee_value, label=band)
plt.legend()
plt.xlabel('measured $\sigma$, [arcsec]')
plt.ylabel('measured EE, [arcsec]')
plt.savefig('plot_output/acs_wfc_gaussian_ee_corr_fraction.png')
plt.close()

# save dictionary
if not os.path.isdir('data_output'):
    os.makedirs('data_output')

with open('data_output/acs_wfc_psf_correction_dict.pickle', 'wb') as file_name:
    pickle.dump(acs_wfc_psf_correction_dict, file_name)




wfc3_uv_psf_correction_dict = {}

for band in wfc3_uv_band_list:
    print(band)
    psf_dict = phot_tools.PSFTools.load_obs_psf_dict(band=band, instrument='uvis')
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
        # print('conv ', conv)
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
        # get the center
        dao_src = phot_tools.SrcTools.detect_star_like_src_in_band_cutout(data=data,
                                                                          detection_threshold=np.max(data) / 2,
                                                                          psf_fwhm_pix=psf_fwhm_pix)
        profile_dict = phot_tools.ProfileTools.compute_axis_profiles(data=data_convolve,
                                                                     x_pos=dao_src['xcentroid'].value[0],
                                                                     y_pos=dao_src['ycentroid'].value[0], n_slits=12,
                                                                     err=data_err)
        morph_dict = phot_tools.ProfileTools.measure_morph_photometry(rad_profile_dict=profile_dict,
                                                                      std_pix=psf_std_pix,
                                                                      data=data_convolve, data_err=data_err,
                                                                      x_pos=dao_src['xcentroid'].value[0],
                                                                      y_pos=dao_src['ycentroid'].value[0],
                                                                      upper_sig_fact=20)
        measured_std_values.append(morph_dict['mean_sig'] * psf_dict['pixel_scale_psf_over_sampled'])
        measured_ee_value.append(morph_dict['flux'] / np.sum(data_convolve))

    wfc3_uv_psf_correction_dict.update({band: {'measured_std_values': measured_std_values, 'measured_ee_value': measured_ee_value}})
    plt.plot(measured_std_values, measured_ee_value, label=band)
plt.legend()
plt.xlabel('measured $\sigma$, [arcsec]')
plt.ylabel('measured EE, [arcsec]')
plt.savefig('plot_output/wfc3_uv_gaussian_ee_corr_fraction.png')
plt.close()

# save dictionary
if not os.path.isdir('data_output'):
    os.makedirs('data_output')

with open('data_output/wfc3_uv_psf_correction_dict.pickle', 'wb') as file_name:
    pickle.dump(wfc3_uv_psf_correction_dict, file_name)




wfc3_ir_psf_correction_dict = {}

for band in wfc3_ir_band_list:
    print(band)
    psf_dict = phot_tools.PSFTools.load_obs_psf_dict(band=band, instrument='ir')
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
        # print('conv ', conv)
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
        # get the center
        dao_src = phot_tools.SrcTools.detect_star_like_src_in_band_cutout(data=data,
                                                                          detection_threshold=np.max(data) / 2,
                                                                          psf_fwhm_pix=psf_fwhm_pix)
        profile_dict = phot_tools.ProfileTools.compute_axis_profiles(data=data_convolve,
                                                                     x_pos=dao_src['xcentroid'].value[0],
                                                                     y_pos=dao_src['ycentroid'].value[0], n_slits=12,
                                                                     err=data_err)
        morph_dict = phot_tools.ProfileTools.measure_morph_photometry(rad_profile_dict=profile_dict,
                                                                      std_pix=psf_std_pix,
                                                                      data=data_convolve, data_err=data_err,
                                                                      x_pos=dao_src['xcentroid'].value[0],
                                                                      y_pos=dao_src['ycentroid'].value[0],
                                                                      upper_sig_fact=20)
        measured_std_values.append(morph_dict['mean_sig'] * psf_dict['pixel_scale_psf_over_sampled'])
        measured_ee_value.append(morph_dict['flux'] / np.sum(data_convolve))

    wfc3_ir_psf_correction_dict.update({band: {'measured_std_values': measured_std_values, 'measured_ee_value': measured_ee_value}})
    plt.plot(measured_std_values, measured_ee_value, label=band)
plt.legend()
plt.xlabel('measured $\sigma$, [arcsec]')
plt.ylabel('measured EE, [arcsec]')
plt.savefig('plot_output/wfc3_ir_gaussian_ee_corr_fraction.png')
plt.close()

# save dictionary
if not os.path.isdir('data_output'):
    os.makedirs('data_output')

with open('data_output/wfc3_ir_psf_correction_dict.pickle', 'wb') as file_name:
    pickle.dump(wfc3_ir_psf_correction_dict, file_name)

