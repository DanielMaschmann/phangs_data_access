"""
all needed data of JWST PSFs
"""
import numpy as np
import os

from phangs_data_access import phot_tools, phys_params
import matplotlib.pyplot as plt
# import webbpsf
import stpsf
import pickle



nircam_band_list = phys_params.nircam_bands
miri_band_list = phys_params.miri_bands
super_sample_factor_nircam = 5
super_sample_factor_miri = 5

# we chose 60 as an arbitrary FWHM factor to simulate the PSF
psf_scaling_size_nircam = 60

# in order to get the same EE values as in the Miri documentation we use the same normalization of the psf with a
# radius of 5 arcsec. See also:
# https://jwst-docs.stsci.edu/jwst-mid-infrared-instrument/miri-performance/miri-point-spread-functions#gsc.tab=0
fov_arcsec_miri = 10


# encirceled energy values
ee_values = [0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.99]
ee_str = ['25', '30', '35', '40', '45', '50', '55', '60', '65', '70', '75', '80', '85', '90', '95', '99']


psf_dict_jwst_nircam = {}
for band in nircam_band_list:
    if band in ['F150W2', 'F322W2']:
        continue

    nrc = stpsf.NIRCam()
    nrc.filter = band

    empirical_fwhm_pix = phys_params.nircam_empirical_fwhm[band]['fwhm_pix']
    # compute fov pixel size
    fov_pixels = np.rint(empirical_fwhm_pix * psf_scaling_size_nircam)
    # make sure the number is odd
    if fov_pixels % 2 == 0: fov_pixels += 1
    # compute psf

    psf = nrc.calc_psf(oversample=super_sample_factor_nircam,
                            fov_pixels=fov_pixels)
    print('shape over sampeled ', psf[2].data.shape)
    print('shape native scale ', psf[3].data.shape)
    pixel_scale = psf[3].header['PIXELSCL']
    pixel_scale_super_sampled = psf[2].header['PIXELSCL']
    fwhm = stpsf.measure_fwhm(psf, ext=0)
    central_x_pos = psf[2].data.shape[0] / 2
    central_y_pos = psf[2].data.shape[1] / 2
    max_rad = np.min(psf[2].data.shape) / 2
    # get radial profile stats:
    rad_profile_stat_dict = phot_tools.ProfileTools.get_rad_profile(
        data=psf[2].data, x_pos=central_x_pos, y_pos=central_y_pos, max_rad=max_rad, err=None,
        pix_scale=pixel_scale_super_sampled, method='exact')

    plt.plot(rad_profile_stat_dict['rad'], rad_profile_stat_dict['profile'] / max(rad_profile_stat_dict['profile']),
             color='blue')

    ee_values_arcsec = phot_tools.ProfileTools.get_src_ee(data=psf[2].data, x_pos=central_x_pos, y_pos=central_y_pos,
                                                max_rad=max_rad, ee_values=ee_values,
                                                pix_scale=pixel_scale_super_sampled,
                                                err=None)
    ee_values_pix= phot_tools.ProfileTools.get_src_ee(data=psf[2].data, x_pos=central_x_pos, y_pos=central_y_pos,
                                                max_rad=max_rad, ee_values=ee_values,
                                                pix_scale=1/super_sample_factor_nircam,
                                                err=None)

    print(band, ee_values_arcsec[5], ee_values_arcsec[11], rad_profile_stat_dict['gaussian_fwhm'])
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
        }
    })
    # encircled energy values
    for idx_ee, ee in enumerate(ee_str):
        psf_dict_jwst_nircam[band].update({'ee_%s_arcsec' % ee: ee_values_arcsec[idx_ee]})
        psf_dict_jwst_nircam[band].update({'ee_%s_pix' % ee: ee_values_pix[idx_ee]})


# save dictionary
if not os.path.isdir('data_output'):
    os.makedirs('data_output')

with open('data_output/nircam_psf_dict.npy', 'wb') as file_name:
    pickle.dump(psf_dict_jwst_nircam, file_name)


psf_dict_jwst_miri = {}
for band in miri_band_list:
    if band in ['F1065C', 'F1140C', 'F1550C', 'F2300C']:
        continue
    nrc = stpsf.MIRI()
    nrc.filter = band

    # compute psf
    psf = nrc.calc_psf(oversample=super_sample_factor_miri, fov_arcsec=fov_arcsec_miri)

    print('shape over sampeled ', psf[2].data.shape)
    print('shape native scale ', psf[3].data.shape)


    pixel_scale = psf[3].header['PIXELSCL']
    pixel_scale_super_sampled = psf[2].header['PIXELSCL']
    fwhm = stpsf.measure_fwhm(psf, ext=0)
    central_x_pos = psf[2].data.shape[0] / 2
    central_y_pos = psf[2].data.shape[1] / 2
    max_rad = np.min(psf[2].data.shape) / 2

    print(pixel_scale_super_sampled, fwhm)

    # get radial profile stats:
    rad_profile_stat_dict = phot_tools.ProfileTools.get_rad_profile(
        data=psf[2].data, x_pos=central_x_pos, y_pos=central_y_pos, max_rad=max_rad, err=None,
        pix_scale=pixel_scale_super_sampled, method='exact')

    plt.plot(rad_profile_stat_dict['rad'], rad_profile_stat_dict['profile'] / max(rad_profile_stat_dict['profile']),
             color='blue')
    # plt.plot([fwhm/2, fwhm/2], [0, 1], linestyle='--', color='k')
    # plt.show()

    ee_values_arcsec = phot_tools.ProfileTools.get_src_ee(data=psf[2].data, x_pos=central_x_pos, y_pos=central_y_pos,
                                                max_rad=max_rad, ee_values=ee_values,
                                                pix_scale=pixel_scale_super_sampled,
                                                err=None)
    ee_values_pix= phot_tools.ProfileTools.get_src_ee(data=psf[2].data, x_pos=central_x_pos, y_pos=central_y_pos,
                                                max_rad=max_rad, ee_values=ee_values,
                                                pix_scale=1/super_sample_factor_miri,
                                                err=None)

    print(band, ee_values_arcsec[5], ee_values_arcsec[11], rad_profile_stat_dict['gaussian_fwhm'])
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
        }
    })
    # encircled energy values
    for idx_ee, ee in enumerate(ee_str):
        psf_dict_jwst_miri[band].update({'ee_%s_arcsec' % ee: ee_values_arcsec[idx_ee]})
        psf_dict_jwst_miri[band].update({'ee_%s_pix' % ee: ee_values_pix[idx_ee]})



with open('data_output/miri_psf_dict.npy', 'wb') as file_name:
    pickle.dump(psf_dict_jwst_miri, file_name)

plt.show()