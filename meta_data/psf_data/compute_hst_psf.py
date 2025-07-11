"""
create all needed data for the HST PSFs
"""
import numpy as np
import os
from astropy.io import fits
from phangs_data_access import phot_tools, phangs_info
import matplotlib.pyplot as plt
import pickle

# taken from https://www.stsci.edu/hst/instrumentation/acs/instrument-design
pixel_size_acs_wfc = 0.05
# taken from
# https://hst-docs.stsci.edu/wfc3ihb/chapter-2-wfc3-instrument-description/2-2-field-of-view-and-geometric-distortions
pixel_size_wfc3_uvis = 0.0395
pixel_size_wfc3_ir = 0.13

# how many supersampling the PSF has is described in
# https://www.stsci.edu/hst/instrumentation/wfc3/data-analysis/psf
super_sample_factor_acs_wfc = 4
super_sample_factor_wfc3_uvis = 4
super_sample_factor_wfc3_ir = 4


# encirceled energy values
ee_values = [0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.99]
ee_str = ['25', '30', '35', '40', '45', '50', '55', '60', '65', '70', '75', '80', '85', '90', '95', '99']



acs_wfc_band_list = phangs_info.acs_wfc_psf_band_list
wfc3_uv_band_list = phangs_info.wfc3_uv_psf_band_list
wfc3_ir_band_list = phangs_info.wfc3_ir_psf_band_list




psf_dict_hst_acs_wfc = {}
for band in acs_wfc_band_list:
    # getting empirical PSF. This PSF is oversampeled by a factor of 4
    if os.path.isfile(f'data/ACSWFC/STDPSF_ACSWFC_{band}.fits'):
        file_name = f'data/ACSWFC/STDPSF_ACSWFC_{band}.fits'
    elif os.path.isfile(f'data/ACSWFC/STDPSF_ACSWFC_{band}_SM4.fits'):
        file_name = f'data/ACSWFC/STDPSF_ACSWFC_{band}_SM4.fits'
    elif os.path.isfile(f'data/ACSWFC/STDPSF_ACSWFC_{band}_SM3.fits'):
        file_name = f'data/ACSWFC/STDPSF_ACSWFC_{band}_SM3.fits'
    else:
        raise FileNotFoundError(' file not found ')
    hdu = fits.open(file_name)
    data = hdu[0].data
    mean_psf = np.mean(data, axis=0)





    central_x_pos = mean_psf.shape[0]/2
    central_y_pos = mean_psf.shape[1]/2
    max_rad = np.min(mean_psf.shape) / 2

    # get radial profile stats:
    rad_profile_stat_dict = phot_tools.ProfileTools.get_rad_profile(
        data=mean_psf, x_pos=central_x_pos, y_pos=central_y_pos, max_rad=max_rad, err=None,
        pix_scale=(pixel_size_acs_wfc / super_sample_factor_acs_wfc), method='exact')


    plt.plot(rad_profile_stat_dict['rad'], rad_profile_stat_dict['profile'] / max(rad_profile_stat_dict['profile']), color='green')

    ee_values_arcsec = phot_tools.ProfileTools.get_src_ee(data=mean_psf, x_pos=central_x_pos, y_pos=central_y_pos,
                                                max_rad=max_rad, ee_values=ee_values,
                                                pix_scale=(pixel_size_acs_wfc / super_sample_factor_acs_wfc),
                                                err=None)
    ee_values_pix= phot_tools.ProfileTools.get_src_ee(data=mean_psf, x_pos=central_x_pos, y_pos=central_y_pos,
                                                max_rad=max_rad, ee_values=ee_values,
                                                pix_scale=1/super_sample_factor_acs_wfc,
                                                err=None)

    print(band, ee_values_arcsec[5], ee_values_arcsec[11], rad_profile_stat_dict['gaussian_fwhm'])
    psf_dict_hst_acs_wfc.update({
        '%s' % band: {
            # the PSF itself
            'over_sampled_psf': mean_psf,
            'pixel_scale_psf_over_sampled': (pixel_size_acs_wfc / super_sample_factor_acs_wfc),
            'n_over_sampled' : super_sample_factor_acs_wfc,
            # parametrization of the radial profile
            'radius_arcsec': rad_profile_stat_dict['rad'],
            'psf_profile': rad_profile_stat_dict['profile'],
            'gaussian_profile': rad_profile_stat_dict['gaussian_profile'],
            'gaussian_fwhm': rad_profile_stat_dict['gaussian_fwhm'],
            'gaussian_amp': rad_profile_stat_dict['gaussian_amp'],
            'gaussian_mean': rad_profile_stat_dict['gaussian_mean'],
            'gaussian_std': rad_profile_stat_dict['gaussian_std'],
        }
    })
    # encircled energy values
    for idx_ee, ee in enumerate(ee_str):
        psf_dict_hst_acs_wfc[band].update({'ee_%s_arcsec' % ee: ee_values_arcsec[idx_ee]})
        psf_dict_hst_acs_wfc[band].update({'ee_%s_pix' % ee: ee_values_pix[idx_ee]})


# save dictionary
if not os.path.isdir('data_output'):
    os.makedirs('data_output')

with open('data_output/hst_acs_wfc_psf_dict.npy', 'wb') as file_name:
    pickle.dump(psf_dict_hst_acs_wfc, file_name)



psf_dict_hst_wfc3_uv = {}
for band in wfc3_uv_band_list:

    hdu = fits.open(f'data/WFC3UV/STDPSF_WFC3UV_{band}.fits')
    data = hdu[0].data
    mean_psf = np.mean(data, axis=0)


    central_x_pos = mean_psf.shape[0]/2
    central_y_pos = mean_psf.shape[1]/2
    max_rad = np.min(mean_psf.shape) / 2

    # get radial profile stats:
    rad_profile_stat_dict = phot_tools.ProfileTools.get_rad_profile(
        data=mean_psf, x_pos=central_x_pos, y_pos=central_y_pos, max_rad=max_rad, err=None,
        pix_scale=(pixel_size_wfc3_uvis / super_sample_factor_wfc3_uvis), method='exact')

    plt.plot(rad_profile_stat_dict['rad'], rad_profile_stat_dict['profile'] / max(rad_profile_stat_dict['profile']),
             color='blue')

    ee_values_arcsec = phot_tools.ProfileTools.get_src_ee(data=mean_psf, x_pos=central_x_pos, y_pos=central_y_pos,
                                                max_rad=max_rad, ee_values=ee_values,
                                                pix_scale=(pixel_size_wfc3_uvis / super_sample_factor_wfc3_uvis),
                                                err=None)
    ee_values_pix= phot_tools.ProfileTools.get_src_ee(data=mean_psf, x_pos=central_x_pos, y_pos=central_y_pos,
                                                max_rad=max_rad, ee_values=ee_values,
                                                pix_scale=1/super_sample_factor_wfc3_uvis,
                                                err=None)

    print(band, ee_values_arcsec[5], ee_values_arcsec[11], rad_profile_stat_dict['gaussian_fwhm'])
    psf_dict_hst_wfc3_uv.update({
        '%s' % band: {
            # the PSF itself
            'over_sampled_psf': mean_psf,
            'pixel_scale_psf_over_sampled': (pixel_size_wfc3_uvis / super_sample_factor_wfc3_uvis),
            'n_over_sampled' : super_sample_factor_wfc3_uvis,
            # parametrization of the radial profile
            'radius_arcsec': rad_profile_stat_dict['rad'],
            'psf_profile': rad_profile_stat_dict['profile'],
            'gaussian_profile': rad_profile_stat_dict['gaussian_profile'],
            'gaussian_fwhm': rad_profile_stat_dict['gaussian_fwhm'],
            'gaussian_amp': rad_profile_stat_dict['gaussian_amp'],
            'gaussian_mean': rad_profile_stat_dict['gaussian_mean'],
            'gaussian_std': rad_profile_stat_dict['gaussian_std'],
        }
    })
    # encircled energy values
    for idx_ee, ee in enumerate(ee_str):
        psf_dict_hst_wfc3_uv[band].update({'ee_%s_arcsec' % ee: ee_values_arcsec[idx_ee]})
        psf_dict_hst_wfc3_uv[band].update({'ee_%s_pix' % ee: ee_values_pix[idx_ee]})

with open('data_output/hst_wfc3_uv_psf_dict.npy', 'wb') as file_name:
    pickle.dump(psf_dict_hst_wfc3_uv, file_name)


psf_dict_hst_wfc3_ir = {}
for band in wfc3_ir_band_list:

    hdu = fits.open(f'data/WFC3IR/STDPSF_WFC3IR_{band}.fits')
    data = hdu[0].data
    mean_psf = np.mean(data, axis=0)

    central_x_pos = mean_psf.shape[0]/2
    central_y_pos = mean_psf.shape[1]/2
    max_rad = np.min(mean_psf.shape) / 2

    # get radial profile stats:
    rad_profile_stat_dict = phot_tools.ProfileTools.get_rad_profile(
        data=mean_psf, x_pos=central_x_pos, y_pos=central_y_pos, max_rad=max_rad, err=None,
        pix_scale=(pixel_size_wfc3_ir / super_sample_factor_wfc3_ir), method='exact')

    plt.plot(rad_profile_stat_dict['rad'], rad_profile_stat_dict['profile'] / max(rad_profile_stat_dict['profile']),
             color='red')

    ee_values_arcsec = phot_tools.ProfileTools.get_src_ee(data=mean_psf, x_pos=central_x_pos, y_pos=central_y_pos,
                                                max_rad=max_rad, ee_values=ee_values,
                                                pix_scale=(pixel_size_wfc3_ir / super_sample_factor_wfc3_ir),
                                                err=None)
    ee_values_pix= phot_tools.ProfileTools.get_src_ee(data=mean_psf, x_pos=central_x_pos, y_pos=central_y_pos,
                                                max_rad=max_rad, ee_values=ee_values,
                                                pix_scale=1/super_sample_factor_wfc3_ir,
                                                err=None)

    print(band, ee_values_arcsec[5], ee_values_arcsec[11], rad_profile_stat_dict['gaussian_fwhm'])
    psf_dict_hst_wfc3_ir.update({
        '%s' % band: {
            # the PSF itself
            'over_sampled_psf': mean_psf,
            'pixel_scale_psf_over_sampled': (pixel_size_wfc3_ir / super_sample_factor_wfc3_ir),
            'n_over_sampled' : super_sample_factor_wfc3_ir,
            # parametrization of the radial profile
            'radius_arcsec': rad_profile_stat_dict['rad'],
            'psf_profile': rad_profile_stat_dict['profile'],
            'gaussian_profile': rad_profile_stat_dict['gaussian_profile'],
            'gaussian_fwhm': rad_profile_stat_dict['gaussian_fwhm'],
            'gaussian_amp': rad_profile_stat_dict['gaussian_amp'],
            'gaussian_mean': rad_profile_stat_dict['gaussian_mean'],
            'gaussian_std': rad_profile_stat_dict['gaussian_std'],
        }
    })
    # encircled energy values
    for idx_ee, ee in enumerate(ee_str):
        psf_dict_hst_wfc3_ir[band].update({'ee_%s_arcsec' % ee: ee_values_arcsec[idx_ee]})
        psf_dict_hst_wfc3_ir[band].update({'ee_%s_pix' % ee: ee_values_pix[idx_ee]})

with open('data_output/hst_wfc3_ir_psf_dict.npy', 'wb') as file_name:
    pickle.dump(psf_dict_hst_wfc3_ir, file_name)


plt.show()