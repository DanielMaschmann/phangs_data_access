"""
Script to compute profile fitting correction values for JWST
"""

import os.path
import pickle

import numpy as np
import matplotlib.pyplot as plt

from phangs_data_access import phot_tools, phangs_info, phys_params, helper_func
from phangs_visualizer import plotting_tools

from astropy.convolution import Gaussian2DKernel
from astropy.convolution import convolve

from scipy.interpolate import RegularGridInterpolator


nircam_fact_img_size = 5
miri_fact_img_size = 5

miri_band_list = phys_params.miri_bands
nircam_band_list = phys_params.nircam_bands





miri_apert_gauss_corr_dict = {}

for band in miri_band_list:
    if band in ['F1065C', 'F1140C', 'F1550C', 'F2300C']:
        continue
    print(band)
    psf_dict = phot_tools.PSFTools.load_obs_psf_dict(band=band, instrument='miri')

    psf_data = psf_dict['over_sampled_psf']
    psf_fwhm_pix = psf_dict['gaussian_fwhm'] / psf_dict['pixel_scale_psf_over_sampled']
    psf_std_pix = psf_dict['gaussian_std'] / psf_dict['pixel_scale_psf_over_sampled']

    print('psf_std_arcsec ', psf_dict['gaussian_std'])
    print('psf_std_pix ', psf_std_pix)

    apert_radii_arcsec_list = np.linspace(psf_dict['gaussian_std'], 2.0, 50)
    apert_radii_pix_list = apert_radii_arcsec_list / psf_dict['pixel_scale_psf_over_sampled']

    # get maximal convolution
    max_convolution_for_psf_img = np.min(psf_data.shape)/5
    print('max_convolution_for_psf_img ', max_convolution_for_psf_img)

    # check if this is feasible:
    if max_convolution_for_psf_img > (10 * psf_std_pix):
        max_convolution = 10 * psf_std_pix
    else:
        max_convolution = max_convolution_for_psf_img
    print('max_convolution ', max_convolution)

    list_convolutions = np.linspace(0, max_convolution, 20)
    list_convolutions[0] = list_convolutions[1]/100

    print('list_convolutions ', list_convolutions)

    measured_std_values = np.zeros(len(list_convolutions))
    measured_flux_frac_in_apert = np.zeros((len(apert_radii_arcsec_list), len(list_convolutions)))

    for conv_idx, conv in enumerate(list_convolutions):

        print('conv_idx, conv ', conv_idx, conv)

        # now get a gaussian function as a primarry source
        x_stddev = conv
        y_stddev = conv
        theta = 0

        data_width = np.rint(np.max((conv, psf_std_pix)) * miri_fact_img_size)
        n_pixels = data_width * 2 + 1
        data_width_pix = data_width
        x_bins = np.linspace(-data_width_pix, data_width_pix, int(n_pixels))
        y_bins = np.linspace(-data_width_pix, data_width_pix, int(n_pixels))
        x_mesh, y_mesh = np.meshgrid(x_bins, y_bins)  # get 2D variables instead of 1D
        x0 = (x_bins[0] + x_bins[-1]) / 2
        y0 = (y_bins[0] + y_bins[-1]) / 2
        moffat_data = helper_func.FuncAndModels.star_cluster_moffat2d(
            x=x_mesh, y=y_mesh, x0=x0, y0=y0, mu_0=1, nu=1.3, fwhm=conv)

        # gauss_data = helper_func.FuncAndModels.gauss2d_rot(
        #     x=x_mesh, y=y_mesh, amp=1, x0=x0, y0=y0, sig_x=conv, sig_y=conv, theta=theta)
        data_convolve = convolve(psf_data, moffat_data)
        # norm to 1
        data_convolve = data_convolve / np.sum(data_convolve)

        data_err = np.ones(data_convolve.shape) * np.max(data_convolve) / 100

        x_center_pix = data_convolve.shape[0] / 2
        y_center_pix = data_convolve.shape[1] / 2

        # measure photometry in all radii
        for apert_idx in range(len(apert_radii_pix_list)):
            apert_stats = phot_tools.ApertTools.get_apert_stats(
            data=data_convolve, data_err=data_err, x_pos=x_center_pix, y_pos=y_center_pix,
            aperture_rad_pix=apert_radii_pix_list[apert_idx], mask=None, sig_clip=None, sum_method='exact')

            measured_flux_frac_in_apert[apert_idx, conv_idx] = apert_stats.sum

        profile_dict = phot_tools.ProfileTools.compute_axis_profiles(data=data_convolve,
                                                                     x_pos=x_center_pix,
                                                                     y_pos=y_center_pix, n_slits=12,
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
        mean_sigma = np.nanmean(sig_list[mask_mu * mask_amp]) * psf_dict['pixel_scale_psf_over_sampled']

        measured_std_values[conv_idx] = mean_sigma


    miri_apert_gauss_corr_dict.update(
        {
            band:
                {
                    'apert_radii_arcsec_list': apert_radii_arcsec_list,
                    'measured_std_values': measured_std_values,
                    'measured_flux_frac_in_apert': measured_flux_frac_in_apert
                }})

    # plot_the output
    # interpolate function
    interp_func = helper_func.InterpTools.interp2dgrid(
        x_bins=apert_radii_arcsec_list, y_bins=measured_std_values,func_values=measured_flux_frac_in_apert,
        method='cubic')

    fine_interpolated = helper_func.InterpTools.get2dinterp_fine_grid(
        interp_func=interp_func, x_min=np.min(apert_radii_arcsec_list), x_max=np.max(apert_radii_arcsec_list),
        y_min=np.min(measured_std_values), y_max=np.max(measured_std_values), n_x_bins=100, n_y_bins=100)

    fig = plt.figure(figsize=(30, 12))
    fontsize = 30
    ax_corr_val = fig.add_axes((0.07, 0.07, 0.8, 0.92))
    ax_cbar = fig.add_axes((0.9, 0.3, 0.02, 0.4))
    norm = plotting_tools.ColorBarTools.compute_cbar_norm(vmin_vmax=(np.min(measured_flux_frac_in_apert), np.max(measured_flux_frac_in_apert)))
    ax_corr_val.imshow(fine_interpolated,
               extent=(np.min(apert_radii_arcsec_list), np.max(apert_radii_arcsec_list),
                       np.min(measured_std_values), np.max(measured_std_values)),
               origin='lower',
               norm=norm, cmap='plasma')
    # plot colorbar
    plotting_tools.ColorBarTools.create_cbar(ax_cbar=ax_cbar, cmap='plasma', norm=norm, cbar_label='EE in Aperture', fontsize=fontsize, ticks=None, labelpad=2, tick_width=2,
                    orientation='vertical', top_lable=True, label_color='k',
                    extend='neither')

    # plot also the dots from where we got the interpolation
    x_mesh, y_mesh = np.meshgrid(apert_radii_arcsec_list, measured_std_values)

    ax_corr_val.scatter(x_mesh, y_mesh, c=measured_flux_frac_in_apert.T, norm=norm, cmap='plasma', edgecolors='k')
    ax_corr_val.set_xlabel('Aperture [arcsec]', fontsize=fontsize,)
    ax_corr_val.set_ylabel(r'Measured Gauss. $\sigma$ [arcsec]', fontsize=fontsize,)
    ax_corr_val.tick_params(axis='both', which='both', width=3, length=15, right=False, top=True, direction='in',
                   labelsize=fontsize)
    if not os.path.isdir('plot_output/apert_profile_corr/miri'): os.makedirs('plot_output/apert_profile_corr/miri')
    fig.savefig('plot_output/apert_profile_corr/miri/apert_profile_corr_%s.png' % band)
    plt.clf()
    plt.cla()

    # # lastly check that everything has correct values
    # print('x_val=0.25, y_val=0.1 ', helper_func.InterpTools.get2dinterp_value(interp_func=interp_func, x_val=0.25, y_val=0.1))
    # print('x_val=0.40, y_val=0.1 ', helper_func.InterpTools.get2dinterp_value(interp_func=interp_func, x_val=0.40, y_val=0.1))
    # print('x_val=0.59, y_val=0.1 ', helper_func.InterpTools.get2dinterp_value(interp_func=interp_func, x_val=0.59, y_val=0.1))
    #
    # print('x_val=0.25, y_val=0.2 ', helper_func.InterpTools.get2dinterp_value(interp_func=interp_func, x_val=0.25, y_val=0.25))
    # print('x_val=0.25, y_val=0.3 ', helper_func.InterpTools.get2dinterp_value(interp_func=interp_func, x_val=0.25, y_val=0.35))
    # print('x_val=0.25, y_val=0.4 ', helper_func.InterpTools.get2dinterp_value(interp_func=interp_func, x_val=0.25, y_val=0.41))


# save dictionary
if not os.path.isdir('data_output/apert_gauss_corr'):
    os.makedirs('data_output/apert_gauss_corr')

with open('data_output/apert_gauss_corr/miri_apert_gauss_corr_dict.pickle', 'wb') as file_name:
    pickle.dump(miri_apert_gauss_corr_dict, file_name)



nircam_apert_gauss_corr_dict = {}

for band in nircam_band_list:
    if band in ['F150W2', 'F322W2']:
        continue
    print(band)
    psf_dict = phot_tools.PSFTools.load_obs_psf_dict(band=band, instrument='nircam')

    psf_data = psf_dict['over_sampled_psf']
    psf_fwhm_pix = psf_dict['gaussian_fwhm'] / psf_dict['pixel_scale_psf_over_sampled']
    psf_std_pix = psf_dict['gaussian_std'] / psf_dict['pixel_scale_psf_over_sampled']

    print('psf_std_arcsec ', psf_dict['gaussian_std'])
    print('psf_std_pix ', psf_std_pix)

    apert_radii_arcsec_list = np.linspace(psf_dict['gaussian_std'], 2.0, 50)
    apert_radii_pix_list = apert_radii_arcsec_list / psf_dict['pixel_scale_psf_over_sampled']

    # get maximal convolution
    max_convolution_for_psf_img = np.min(psf_data.shape)/5
    print('max_convolution_for_psf_img ', max_convolution_for_psf_img)

    # check if this is feasible:
    if max_convolution_for_psf_img > (10 * psf_std_pix):
        max_convolution = 10 * psf_std_pix
    else:
        max_convolution = max_convolution_for_psf_img
    print('max_convolution ', max_convolution)

    list_convolutions = np.linspace(0, max_convolution, 20)
    list_convolutions[0] = list_convolutions[1]/100

    print('list_convolutions ', list_convolutions)

    measured_std_values = np.zeros(len(list_convolutions))
    measured_flux_frac_in_apert = np.zeros((len(apert_radii_arcsec_list), len(list_convolutions)))

    for conv_idx, conv in enumerate(list_convolutions):

        print('conv_idx, conv ', conv_idx, conv)

        # now get a gaussian function as a primarry source
        x_stddev = conv
        y_stddev = conv
        theta = 0

        data_width = np.rint(np.max((conv, psf_std_pix)) * nircam_fact_img_size)
        n_pixels = data_width * 2 + 1
        data_width_pix = data_width
        x_bins = np.linspace(-data_width_pix, data_width_pix, int(n_pixels))
        y_bins = np.linspace(-data_width_pix, data_width_pix, int(n_pixels))
        x_mesh, y_mesh = np.meshgrid(x_bins, y_bins)  # get 2D variables instead of 1D
        x0 = (x_bins[0] + x_bins[-1]) / 2
        y0 = (y_bins[0] + y_bins[-1]) / 2
        moffat_data = helper_func.FuncAndModels.star_cluster_moffat2d(
            x=x_mesh, y=y_mesh, x0=x0, y0=y0, mu_0=1, nu=1.3, fwhm=conv)

        # gauss_data = helper_func.FuncAndModels.gauss2d_rot(
        #     x=x_mesh, y=y_mesh, amp=1, x0=x0, y0=y0, sig_x=conv, sig_y=conv, theta=theta)
        data_convolve = convolve(psf_data, moffat_data)
        # norm to 1
        data_convolve = data_convolve / np.sum(data_convolve)

        data_err = np.ones(data_convolve.shape) * np.max(data_convolve) / 100

        x_center_pix = data_convolve.shape[0] / 2
        y_center_pix = data_convolve.shape[1] / 2

        # measure photometry in all radii
        for apert_idx in range(len(apert_radii_pix_list)):
            apert_stats = phot_tools.ApertTools.get_apert_stats(
            data=data_convolve, data_err=data_err, x_pos=x_center_pix, y_pos=y_center_pix,
            aperture_rad_pix=apert_radii_pix_list[apert_idx], mask=None, sig_clip=None, sum_method='exact')

            measured_flux_frac_in_apert[apert_idx, conv_idx] = apert_stats.sum

        profile_dict = phot_tools.ProfileTools.compute_axis_profiles(data=data_convolve,
                                                                     x_pos=x_center_pix,
                                                                     y_pos=y_center_pix, n_slits=12,
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
        mean_sigma = np.nanmean(sig_list[mask_mu * mask_amp]) * psf_dict['pixel_scale_psf_over_sampled']

        measured_std_values[conv_idx] = mean_sigma


    nircam_apert_gauss_corr_dict.update(
        {
            band:
                {
                    'apert_radii_arcsec_list': apert_radii_arcsec_list,
                    'measured_std_values': measured_std_values,
                    'measured_flux_frac_in_apert': measured_flux_frac_in_apert
                }})

    # plot_the output
    # interpolate function
    interp_func = helper_func.InterpTools.interp2dgrid(
        x_bins=apert_radii_arcsec_list, y_bins=measured_std_values,func_values=measured_flux_frac_in_apert,
        method='cubic')

    fine_interpolated = helper_func.InterpTools.get2dinterp_fine_grid(
        interp_func=interp_func, x_min=np.min(apert_radii_arcsec_list), x_max=np.max(apert_radii_arcsec_list),
        y_min=np.min(measured_std_values), y_max=np.max(measured_std_values), n_x_bins=100, n_y_bins=100)

    fig = plt.figure(figsize=(30, 12))
    fontsize = 30
    ax_corr_val = fig.add_axes((0.07, 0.07, 0.8, 0.92))
    ax_cbar = fig.add_axes((0.9, 0.3, 0.02, 0.4))
    norm = plotting_tools.ColorBarTools.compute_cbar_norm(vmin_vmax=(np.min(measured_flux_frac_in_apert), np.max(measured_flux_frac_in_apert)))
    ax_corr_val.imshow(fine_interpolated,
               extent=(np.min(apert_radii_arcsec_list), np.max(apert_radii_arcsec_list),
                       np.min(measured_std_values), np.max(measured_std_values)),
               origin='lower',
               norm=norm, cmap='plasma')
    # plot colorbar
    plotting_tools.ColorBarTools.create_cbar(ax_cbar=ax_cbar, cmap='plasma', norm=norm, cbar_label='EE in Aperture', fontsize=fontsize, ticks=None, labelpad=2, tick_width=2,
                    orientation='vertical', top_lable=True, label_color='k',
                    extend='neither')

    # plot also the dots from where we got the interpolation
    x_mesh, y_mesh = np.meshgrid(apert_radii_arcsec_list, measured_std_values)

    ax_corr_val.scatter(x_mesh, y_mesh, c=measured_flux_frac_in_apert.T, norm=norm, cmap='plasma', edgecolors='k')
    ax_corr_val.set_xlabel('Aperture [arcsec]', fontsize=fontsize,)
    ax_corr_val.set_ylabel(r'Measured Gauss. $\sigma$ [arcsec]', fontsize=fontsize,)
    ax_corr_val.tick_params(axis='both', which='both', width=3, length=15, right=False, top=True, direction='in',
                   labelsize=fontsize)
    if not os.path.isdir('plot_output/apert_profile_corr/nircam'): os.makedirs('plot_output/apert_profile_corr/nircam')
    fig.savefig('plot_output/apert_profile_corr/nircam/apert_profile_corr_%s.png' % band)
    plt.clf()
    plt.cla()

    # # lastly check that everything has correct values
    # print('x_val=0.25, y_val=0.1 ', helper_func.InterpTools.get2dinterp_value(interp_func=interp_func, x_val=0.25, y_val=0.1))
    # print('x_val=0.40, y_val=0.1 ', helper_func.InterpTools.get2dinterp_value(interp_func=interp_func, x_val=0.40, y_val=0.1))
    # print('x_val=0.59, y_val=0.1 ', helper_func.InterpTools.get2dinterp_value(interp_func=interp_func, x_val=0.59, y_val=0.1))
    #
    # print('x_val=0.25, y_val=0.2 ', helper_func.InterpTools.get2dinterp_value(interp_func=interp_func, x_val=0.25, y_val=0.25))
    # print('x_val=0.25, y_val=0.3 ', helper_func.InterpTools.get2dinterp_value(interp_func=interp_func, x_val=0.25, y_val=0.35))
    # print('x_val=0.25, y_val=0.4 ', helper_func.InterpTools.get2dinterp_value(interp_func=interp_func, x_val=0.25, y_val=0.41))


# save dictionary
if not os.path.isdir('data_output/apert_gauss_corr'):
    os.makedirs('data_output/apert_gauss_corr')

with open('data_output/apert_gauss_corr/nircam_apert_gauss_corr_dict.pickle', 'wb') as file_name:
    pickle.dump(nircam_apert_gauss_corr_dict, file_name)




