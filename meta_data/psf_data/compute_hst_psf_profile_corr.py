"""
Script to compute profile fitting correction values for HST
"""

import os.path
import pickle

import numpy as np
import matplotlib.pyplot as plt

from phangs_data_access import phot_tools, phangs_info, helper_func

from astropy.convolution import Gaussian2DKernel
from astropy.convolution import convolve


acs_wfc_band_list = phangs_info.acs_wfc_psf_band_list
wfc3_uv_band_list = phangs_info.wfc3_uv_psf_band_list
wfc3_ir_band_list = phangs_info.wfc3_ir_psf_band_list

acs_wfc_fact_img_size = 5
wfc3_uv_fact_img_size = 5
wfc3_ir_fact_img_size = 5

# acs_wfc_psf_correction_dict = {}
#
# for band in acs_wfc_band_list:
#     if band in ['F150W2', 'F322W2']:
#         continue
#     print(band)
#     psf_dict = phot_tools.PSFTools.load_obs_psf_dict(band=band, instrument='acs')
#
#     psf_data = psf_dict['over_sampled_psf']
#     psf_fwhm_pix = psf_dict['gaussian_fwhm'] / psf_dict['pixel_scale_psf_over_sampled']
#     psf_std_pix = psf_dict['gaussian_std'] / psf_dict['pixel_scale_psf_over_sampled']
#
#     # get maximal convolution
#     max_convolution_for_psf_img = np.min(psf_data.shape)/5
#     # check if this is feasible:
#     if max_convolution_for_psf_img > (10 * psf_std_pix):
#         max_convolution = 10 * psf_std_pix
#     else:
#         max_convolution = max_convolution_for_psf_img
#
#     list_convolutions = np.linspace(0, max_convolution, 40)
#     list_convolutions[0] = list_convolutions[1]/100
#
#     measured_std_values = []
#     measured_ee_value = []
#
#     for conv in list_convolutions:
#
#
#         # now get a gaussian function as a primarry source
#         x_stddev = conv
#         y_stddev = conv
#         theta = 0
#
#         data_width = np.rint(np.max((conv, psf_std_pix)) * acs_wfc_fact_img_size)
#         n_pixels = data_width * 2 + 1
#         data_width_pix = data_width
#         x_bins = np.linspace(-data_width_pix, data_width_pix, int(n_pixels))
#         y_bins = np.linspace(-data_width_pix, data_width_pix, int(n_pixels))
#         x_mesh, y_mesh = np.meshgrid(x_bins, y_bins)  # get 2D variables instead of 1D
#         x0 = (x_bins[0] + x_bins[-1]) / 2
#         y0 = (y_bins[0] + y_bins[-1]) / 2
#         gauss_data = helper_func.FuncAndModels.gauss2d_rot(
#             x=x_mesh, y=y_mesh, amp=1, x0=x0, y0=y0, sig_x=conv, sig_y=conv, theta=theta)
#         data_convolve = convolve(gauss_data, psf_data)
#         # norm to 1
#         data_convolve = data_convolve/np.sum(data_convolve)
#
#         # plt.imshow(gauss_data)
#         # plt.show()
#         # plt.imshow(psf_data)
#         # plt.show()
#         # plt.imshow(data_convolve)
#         # plt.show()
#
#
#
#         data_err = np.ones(data_convolve.shape) * np.max(data_convolve) / 100
#
#         x_center = data_convolve.shape[0] / 2
#         y_center = data_convolve.shape[1] / 2
#         profile_dict = phot_tools.ProfileTools.compute_axis_profiles(data=data_convolve,
#                                                                      x_pos=x_center,
#                                                                      y_pos=y_center, n_slits=12,
#                                                                      err=data_err)
#
#         amp_list, mu_list, sig_list, amp_err_list, mu_err_list, sig_err_list = \
#             phot_tools.ProfileTools.fit_gauss2rad_profiles(rad_profile_dict=profile_dict, std_pix=psf_std_pix,
#                                                 upper_sig_fact=10, central_rad_fact=3)
#         amp_list = np.array(amp_list)
#         mu_list = np.array(mu_list)
#         sig_list = np.array(sig_list)
#         amp_err_list = np.array(amp_err_list)
#         sig_err_list = np.array(sig_err_list)
#
#         # get all the gaussian functions that make sense
#         # then need to be central
#         mask_mu = np.abs(mu_list) < psf_std_pix * 3
#         # they must have a positive amplitude
#         mask_amp = (amp_list > 0)
#         array_flux = amp_list[mask_mu * mask_amp] * 2 * np.pi * (sig_list[mask_mu * mask_amp] ** 2)
#         measured_sig = np.nanmean(sig_list) * psf_dict['pixel_scale_psf_over_sampled']
#         measured_flux = np.nanmean(array_flux)
#
#         print('measured_sig ', measured_sig, ' std ', psf_dict['gaussian_std'])
#         print(' measured_flux ', measured_flux, ' fraction ', measured_flux / np.sum(data_convolve))
#         measured_std_values.append(measured_sig)
#         measured_ee_value.append(measured_flux / np.sum(data_convolve))
#
#     acs_wfc_psf_correction_dict.update({band: {'measured_std_values': measured_std_values, 'measured_ee_value': measured_ee_value}})
#
#     plt.plot(measured_std_values, measured_ee_value, label=band)
#
# plt.legend()
# plt.xlabel('measured $\sigma$, [arcsec]')
# plt.ylabel('measured EE, [arcsec]')
# plt.savefig('plot_output/acs_wfc_gaussian_ee_corr_fraction.png')
# plt.close()
#
# # save dictionary
# if not os.path.isdir('data_output'):
#     os.makedirs('data_output')
#
# with open('data_output/acs_wfc_psf_correction_dict.pickle', 'wb') as file_name:
#     pickle.dump(acs_wfc_psf_correction_dict, file_name)
#




wfc3_uv_psf_correction_dict = {}

for band in wfc3_uv_band_list:
    if band in ['F150W2', 'F322W2']:
        continue
    print(band)
    psf_dict = phot_tools.PSFTools.load_obs_psf_dict(band=band, instrument='uvis')

    psf_data = psf_dict['over_sampled_psf']
    psf_fwhm_pix = psf_dict['gaussian_fwhm'] / psf_dict['pixel_scale_psf_over_sampled']
    psf_std_pix = psf_dict['gaussian_std'] / psf_dict['pixel_scale_psf_over_sampled']

    # get maximal convolution
    max_convolution_for_psf_img = np.min(psf_data.shape)/5
    # check if this is feasible:
    if max_convolution_for_psf_img > (10 * psf_std_pix):
        max_convolution = 10 * psf_std_pix
    else:
        max_convolution = max_convolution_for_psf_img

    list_convolutions = np.linspace(0, max_convolution, 40)
    list_convolutions[0] = list_convolutions[1]/100

    measured_std_values = []
    measured_ee_value = []

    for conv in list_convolutions:


        # now get a gaussian function as a primarry source
        x_stddev = conv
        y_stddev = conv
        theta = 0

        data_width = np.rint(np.max((conv, psf_std_pix)) * wfc3_uv_fact_img_size)
        n_pixels = data_width * 2 + 1
        data_width_pix = data_width
        x_bins = np.linspace(-data_width_pix, data_width_pix, int(n_pixels))
        y_bins = np.linspace(-data_width_pix, data_width_pix, int(n_pixels))
        x_mesh, y_mesh = np.meshgrid(x_bins, y_bins)  # get 2D variables instead of 1D
        x0 = (x_bins[0] + x_bins[-1]) / 2
        y0 = (y_bins[0] + y_bins[-1]) / 2
        gauss_data = helper_func.FuncAndModels.gauss2d_rot(
            x=x_mesh, y=y_mesh, amp=1, x0=x0, y0=y0, sig_x=conv, sig_y=conv, theta=theta)
        data_convolve = convolve(gauss_data, psf_data)
        # norm to 1
        data_convolve = data_convolve/np.sum(data_convolve)

        # plt.imshow(gauss_data)
        # plt.show()
        # plt.imshow(psf_data)
        # plt.show()
        # plt.imshow(data_convolve)
        # plt.show()



        data_err = np.ones(data_convolve.shape) * np.max(data_convolve) / 100

        x_center = data_convolve.shape[0] / 2
        y_center = data_convolve.shape[1] / 2
        profile_dict = phot_tools.ProfileTools.compute_axis_profiles(data=data_convolve,
                                                                     x_pos=x_center,
                                                                     y_pos=y_center, n_slits=12,
                                                                     err=data_err)

        amp_list, mu_list, sig_list, amp_err_list, mu_err_list, sig_err_list = \
            phot_tools.ProfileTools.fit_gauss2rad_profiles(rad_profile_dict=profile_dict, std_pix=psf_std_pix,
                                                upper_sig_fact=10, central_rad_fact=5)
        amp_list = np.array(amp_list)
        mu_list = np.array(mu_list)
        sig_list = np.array(sig_list)
        amp_err_list = np.array(amp_err_list)
        sig_err_list = np.array(sig_err_list)

        # get all the gaussian functions that make sense
        # then need to be central
        mask_mu = np.abs(mu_list) < psf_std_pix * 3
        # they must have a positive amplitude
        mask_amp = (amp_list > 0)
        array_flux = amp_list[mask_mu * mask_amp] * 2 * np.pi * (sig_list[mask_mu * mask_amp] ** 2)
        measured_sig = np.nanmean(sig_list) * psf_dict['pixel_scale_psf_over_sampled']
        measured_flux = np.nanmean(array_flux)

        print('measured_sig ', measured_sig, ' std ', psf_dict['gaussian_std'])
        print(' measured_flux ', measured_flux, ' fraction ', measured_flux / np.sum(data_convolve))
        measured_std_values.append(measured_sig)
        measured_ee_value.append(measured_flux / np.sum(data_convolve))

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
    if band in ['F150W2', 'F322W2']:
        continue
    print(band)
    psf_dict = phot_tools.PSFTools.load_obs_psf_dict(band=band, instrument='ir')

    psf_data = psf_dict['over_sampled_psf']
    psf_fwhm_pix = psf_dict['gaussian_fwhm'] / psf_dict['pixel_scale_psf_over_sampled']
    psf_std_pix = psf_dict['gaussian_std'] / psf_dict['pixel_scale_psf_over_sampled']

    # get maximal convolution
    max_convolution_for_psf_img = np.min(psf_data.shape)/5
    # check if this is feasible:
    if max_convolution_for_psf_img > (10 * psf_std_pix):
        max_convolution = 10 * psf_std_pix
    else:
        max_convolution = max_convolution_for_psf_img

    list_convolutions = np.linspace(0, max_convolution, 40)
    list_convolutions[0] = list_convolutions[1]/100

    measured_std_values = []
    measured_ee_value = []

    for conv in list_convolutions:


        # now get a gaussian function as a primarry source
        x_stddev = conv
        y_stddev = conv
        theta = 0

        data_width = np.rint(np.max((conv, psf_std_pix)) * wfc3_ir_fact_img_size)
        n_pixels = data_width * 2 + 1
        data_width_pix = data_width
        x_bins = np.linspace(-data_width_pix, data_width_pix, int(n_pixels))
        y_bins = np.linspace(-data_width_pix, data_width_pix, int(n_pixels))
        x_mesh, y_mesh = np.meshgrid(x_bins, y_bins)  # get 2D variables instead of 1D
        x0 = (x_bins[0] + x_bins[-1]) / 2
        y0 = (y_bins[0] + y_bins[-1]) / 2
        gauss_data = helper_func.FuncAndModels.gauss2d_rot(
            x=x_mesh, y=y_mesh, amp=1, x0=x0, y0=y0, sig_x=conv, sig_y=conv, theta=theta)
        data_convolve = convolve(gauss_data, psf_data)
        # norm to 1
        data_convolve = data_convolve/np.sum(data_convolve)

        # plt.imshow(gauss_data)
        # plt.show()
        # plt.imshow(psf_data)
        # plt.show()
        # plt.imshow(data_convolve)
        # plt.show()



        data_err = np.ones(data_convolve.shape) * np.max(data_convolve) / 100

        x_center = data_convolve.shape[0] / 2
        y_center = data_convolve.shape[1] / 2
        profile_dict = phot_tools.ProfileTools.compute_axis_profiles(data=data_convolve,
                                                                     x_pos=x_center,
                                                                     y_pos=y_center, n_slits=12,
                                                                     err=data_err)

        amp_list, mu_list, sig_list, amp_err_list, mu_err_list, sig_err_list = \
            phot_tools.ProfileTools.fit_gauss2rad_profiles(rad_profile_dict=profile_dict, std_pix=psf_std_pix,
                                                upper_sig_fact=10, central_rad_fact=3)
        amp_list = np.array(amp_list)
        mu_list = np.array(mu_list)
        sig_list = np.array(sig_list)
        amp_err_list = np.array(amp_err_list)
        sig_err_list = np.array(sig_err_list)

        # get all the gaussian functions that make sense
        # then need to be central
        mask_mu = np.abs(mu_list) < psf_std_pix * 3
        # they must have a positive amplitude
        mask_amp = (amp_list > 0)
        array_flux = amp_list[mask_mu * mask_amp] * 2 * np.pi * (sig_list[mask_mu * mask_amp] ** 2)
        measured_sig = np.nanmean(sig_list) * psf_dict['pixel_scale_psf_over_sampled']
        measured_flux = np.nanmean(array_flux)

        print('measured_sig ', measured_sig, ' std ', psf_dict['gaussian_std'])
        print(' measured_flux ', measured_flux, ' fraction ', measured_flux / np.sum(data_convolve))
        measured_std_values.append(measured_sig)
        measured_ee_value.append(measured_flux / np.sum(data_convolve))

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









exit()



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

