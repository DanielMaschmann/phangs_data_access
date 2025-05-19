"""
all needed data of JWST PSFs
"""
import numpy as np
import os

from phangs_data_access import phot_tools, phys_params
import matplotlib.pyplot as plt
import webbpsf
import pickle



nircam_band_list = phys_params.nircam_bands
miri_band_list = phys_params.miri_bands
super_sample_factor_nircam = 5
super_sample_factor_miri = 5
psf_scaling_size = 60


psf_dict_jwst_nircam = {}
for band in nircam_band_list:
    if band in ['F150W2', 'F322W2']:
        continue

    nrc = webbpsf.NIRCam()
    nrc.filter = band

    empirical_fwhm_pix = phys_params.nircam_empirical_fwhm[band]['fwhm_pix']
    # compute fov pixel size
    fov_pixels = np.rint(empirical_fwhm_pix * psf_scaling_size)
    # make sure the number is odd
    if fov_pixels % 2 == 0:
        fov_pixels += 1
    # compute psf
    psf = nrc.calc_psf(oversample=super_sample_factor_nircam,
                            fov_pixels=fov_pixels)
    print('shape over sampeled ', psf[2].data.shape)
    print('shape native scale ', psf[3].data.shape)
    pixel_scale = psf[3].header['PIXELSCL']
    pixel_scale_super_sampled = psf[2].header['PIXELSCL']
    fwhm = webbpsf.measure_fwhm(psf, ext=0)
    central_x_pos = psf[2].data.shape[0] / 2
    central_y_pos = psf[2].data.shape[1] / 2
    max_rad = np.min(psf[2].data.shape) / 2
    # get radial profile stats:
    rad_profile_stat_dict = phot_tools.ProfileTools.get_rad_profile(
        data=psf[2].data, x_pos=central_x_pos, y_pos=central_y_pos, max_rad=max_rad, err=None,
        pix_scale=pixel_scale_super_sampled, method='exact')

    plt.plot(rad_profile_stat_dict['rad'], rad_profile_stat_dict['profile'] / max(rad_profile_stat_dict['profile']),
             color='blue')

    ee_values = phot_tools.ProfileTools.get_src_ee(data=psf[2].data, x_pos=central_x_pos, y_pos=central_y_pos,
                                                max_rad=max_rad, ee_values=[0.25, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99],
                                                pix_scale=pixel_scale_super_sampled,
                                                err=None)
    print(band, ee_values[1], ee_values[4], rad_profile_stat_dict['gaussian_fwhm'])

    psf_dict_jwst_nircam.update({
        '%s' % band: {
            # the PSF itself
            'psf': psf[3].data,
            'over_sampled_psf': psf[2].data,
            'pixel_scale_psf': pixel_scale,
            'pixel_scale_psf_over_sampled': pixel_scale_super_sampled,
            'n_over_sampled': super_sample_factor_nircam,
            # parametrization of the radial profile
            'radius_arcsec': rad_profile_stat_dict['rad'],
            'psf_profile': rad_profile_stat_dict['profile'],
            'gaussian_profile': rad_profile_stat_dict['gaussian_profile'],
            'gaussian_fwhm': rad_profile_stat_dict['gaussian_fwhm'],
            'gaussian_amp': rad_profile_stat_dict['gaussian_amp'],
            'gaussian_mean': rad_profile_stat_dict['gaussian_mean'],
            'gaussian_std': rad_profile_stat_dict['gaussian_std'],
            # encircled energy values
            'ee_25percent': ee_values[0],
            'ee_50percent': ee_values[1],
            'ee_60percent': ee_values[2],
            'ee_70percent': ee_values[3],
            'ee_80percent': ee_values[4],
            'ee_90percent': ee_values[5],
            'ee_95percent': ee_values[6],
            'ee_99percent': ee_values[7],
        }
    })

# save dictionary
if not os.path.isdir('data_output'):
    os.makedirs('data_output')

with open('data_output/nircam_psf_dict.npy', 'wb') as file_name:
    pickle.dump(psf_dict_jwst_nircam, file_name)


psf_dict_jwst_miri = {}
for band in miri_band_list:
    if band in ['F1065C', 'F1140C', 'F1550C', 'F2300C']:
        continue
    nrc = webbpsf.MIRI()
    nrc.filter = band


    empirical_fwhm_pix = phys_params.miri_empirical_fwhm[band]['fwhm_pix']
    # compute fov pixel size
    fov_pixels = np.rint(empirical_fwhm_pix * psf_scaling_size)
    # make sure the number is odd
    if fov_pixels % 2 == 0:
        fov_pixels += 1
    # compute psf
    psf = nrc.calc_psf(oversample=super_sample_factor_miri,
                            fov_pixels=fov_pixels)
    print('shape over sampeled ', psf[2].data.shape)
    print('shape native scale ', psf[3].data.shape)

    pixel_scale = psf[3].header['PIXELSCL']
    pixel_scale_super_sampled = psf[2].header['PIXELSCL']
    fwhm = webbpsf.measure_fwhm(psf, ext=0)
    central_x_pos = psf[2].data.shape[0] / 2
    central_y_pos = psf[2].data.shape[1] / 2
    max_rad = np.min(psf[2].data.shape) / 2

    # get radial profile stats:
    rad_profile_stat_dict = phot_tools.ProfileTools.get_rad_profile(
        data=psf[2].data, x_pos=central_x_pos, y_pos=central_y_pos, max_rad=max_rad, err=None,
        pix_scale=pixel_scale_super_sampled, method='exact')

    plt.plot(rad_profile_stat_dict['rad'], rad_profile_stat_dict['profile'] / max(rad_profile_stat_dict['profile']),
             color='blue')

    ee_values = phot_tools.ProfileTools.get_src_ee(data=psf[2].data, x_pos=central_x_pos, y_pos=central_y_pos,
                                                max_rad=max_rad, ee_values=[0.25, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99],
                                                pix_scale=pixel_scale_super_sampled,
                                                err=None)
    print(band, ee_values[1], ee_values[4], rad_profile_stat_dict['gaussian_fwhm'])

    psf_dict_jwst_miri.update({
        '%s' % band: {
            # the PSF itself
            'psf': psf[3].data,
            'over_sampled_psf': psf[2].data,
            'pixel_scale_psf': pixel_scale,
            'pixel_scale_psf_over_sampled': pixel_scale_super_sampled,
            'n_over_sampled': super_sample_factor_miri,
            # parametrization of the radial profile
            'radius_arcsec': rad_profile_stat_dict['rad'],
            'psf_profile': rad_profile_stat_dict['profile'],
            'gaussian_profile': rad_profile_stat_dict['gaussian_profile'],
            'gaussian_fwhm': rad_profile_stat_dict['gaussian_fwhm'],
            'gaussian_amp': rad_profile_stat_dict['gaussian_amp'],
            'gaussian_mean': rad_profile_stat_dict['gaussian_mean'],
            'gaussian_std': rad_profile_stat_dict['gaussian_std'],
            # encircled energy values
            'ee_25percent': ee_values[0],
            'ee_50percent': ee_values[1],
            'ee_60percent': ee_values[2],
            'ee_70percent': ee_values[3],
            'ee_80percent': ee_values[4],
            'ee_90percent': ee_values[5],
            'ee_95percent': ee_values[6],
            'ee_99percent': ee_values[7],
        }
    })

with open('data_output/miri_psf_dict.npy', 'wb') as file_name:
    pickle.dump(psf_dict_jwst_miri, file_name)

plt.show()