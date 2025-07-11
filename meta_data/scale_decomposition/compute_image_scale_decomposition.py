"""
create all needed data for the HST PSFs
"""
import numpy as np
import os
from astropy.io import fits
from phangs_data_access import phot_tools, phot_access, helper_func, sample_access, phangs_info
import matplotlib.pyplot as plt
import pickle
from astropy.visualization import SqrtStretch, SinhStretch, LogStretch
from astropy.visualization.mpl_normalize import ImageNormalize
from astropy.stats import sigma_clipped_stats


phangs_sample = sample_access.SampleAccess()

# parameters needed for procedure
min_power_pc = 2
max_power_pc = 8

compute_pix_scales_pc = (2.0 ** np.array(range(min_power_pc, max_power_pc + 1)))
compute_pix_scales_lo_pc = (2.0 ** (np.array(range(min_power_pc, max_power_pc + 1)) - 0.5))
compute_pix_scales_hi_pc = (2.0 ** (np.array(range(min_power_pc, max_power_pc + 1)) + 0.5))


min_power_sig = 1
max_power_sig = 5

compute_pix_scales_sig = (2.0 ** np.array(range(min_power_sig, max_power_sig + 1)))
compute_pix_scales_lo_sig = (2.0 ** (np.array(range(min_power_sig, max_power_sig + 1)) - 0.5))
compute_pix_scales_hi_sig = (2.0 ** (np.array(range(min_power_sig, max_power_sig + 1)) + 0.5))

print('pixscales_pc ', compute_pix_scales_pc)
print('pixscales_sig ', compute_pix_scales_sig)

flux_unit = 'mJy'

# hst_target_list = list(phangs_info.hst_obs_band_dict.keys())
# jwst_target_list = list(phangs_info.jwst_obs_band_dict.keys())
#
# target_list = np.unique(hst_target_list + jwst_target_list)

# target_list = list(phangs_info.jwst_obs_band_dict.keys())
target_list = phangs_info.phangs_treasury_jwst_galaxy_list

recompute_pc_flag = False
recompute_sig_flag = True

for target in target_list:
    print(target)

    # getting the distance
    distance_mpc = phangs_sample.get_target_dist(target=helper_func.FileTools.target_name_no_directions(target))

    # initialize data access
    phangs_phot = phot_access.PhotAccess(phot_target_name=target)

    # hst_band_list = helper_func.ObsTools.get_hst_obs_band_list(target=phangs_phot.phot_hst_target_name)
    # nircam_band_list = helper_func.ObsTools.get_nircam_obs_band_list(target=phangs_phot.phot_nircam_target_name)
    # miri_band_list = helper_func.ObsTools.get_miri_obs_band_list(target=phangs_phot.phot_miri_target_name)
    miri_band_list = ['F770W']

    # for hst_band in hst_band_list:
    #     print(hst_band)
    #     # load data
    #     phangs_phot.load_phangs_bands(band_list=[hst_band], flux_unit=flux_unit)
    #     img = phangs_phot.hst_bands_data['%s_data_img' % hst_band]
    #     header = phangs_phot.hst_bands_data['%s_header_img' % hst_band]
    #     wcs = phangs_phot.hst_bands_data['%s_wcs_img' % hst_band]
    #     # get the smallest dimension of the data
    #     min_dim_img = np.min([header['NAXIS1'], header['NAXIS2']])
    #     # get pixel scale of data
    #     pixel_scale_arcsec = helper_func.CoordTools.transform_pix2world_scale(length_in_pix=1.0, wcs=wcs)
    #     # convert pixel scale to pc
    #     pixel_per_pc = np.pi * 1e6 / (3600 * 180) * pixel_scale_arcsec * distance_mpc  # convert to parcecs per pixel
    #     # get resolution of observations
    #     psf_dict = phot_tools.PSFTools.load_obs_psf_dict(
    #         band=hst_band,
    #         instrument=helper_func.ObsTools.get_hst_instrument(target=phangs_phot.phot_hst_target_name, band=hst_band))
    #     gaussian_std_arcsec = psf_dict['gaussian_std']
    #     gaussian_std_pix = helper_func.CoordTools.transform_world2pix_scale(length_in_arcsec=gaussian_std_arcsec, wcs=wcs).value
    #     gaussian_std_per_pixel = 1 / gaussian_std_pix
    #     gaussian_std_in_pc = np.pi * 1e6 / (3600 * 180) * gaussian_std_arcsec * distance_mpc
    #
    #     print('pixel_per_pc ', pixel_per_pc)
    #     print('gaussian_std_arcsec ', gaussian_std_arcsec)
    #     print('gaussian_std_pix ', gaussian_std_pix)
    #     print('gaussian_std_per_pixel ', gaussian_std_per_pixel)
    #     print('gaussian_std_in_pc ', gaussian_std_in_pc)
    #
    #     pix_scales_lo_pc = compute_pix_scales_lo_pc / pixel_per_pc
    #     pix_scales_hi_pc = compute_pix_scales_hi_pc / pixel_per_pc
    #
    #     pix_scales_lo_sig = compute_pix_scales_lo_sig / gaussian_std_per_pixel
    #     pix_scales_hi_sig = compute_pix_scales_hi_sig / gaussian_std_per_pixel
    #
    #     # make sure the scales are working
    #     # the pc scale has to be at least 1.33 larger than the resolution
    #     # furthermore all the scales have to be smaller than the half of the smaller axis pixel size (You can not filter out
    #     # a scale that is larger than the image).
    #     mask_computable_pc = (pix_scales_lo_pc * pixel_per_pc >= 1.33 * gaussian_std_in_pc) & (
    #             pix_scales_hi_pc * pixel_per_pc / gaussian_std_in_pc <= 0.5 * min_dim_img)  # check that scales work
    #
    #     mask_computable_sig = (pix_scales_hi_sig * gaussian_std_per_pixel <= 0.5 * min_dim_img)  # check that scales work
    #
    #     pix_scales_lo_pc = pix_scales_lo_pc[mask_computable_pc]
    #     pix_scales_hi_pc = pix_scales_hi_pc[mask_computable_pc]
    #     pix_scales_pc = compute_pix_scales_pc[mask_computable_pc]
    #
    #     pix_scales_lo_sig = pix_scales_lo_sig[mask_computable_sig]
    #     pix_scales_hi_sig = pix_scales_hi_sig[mask_computable_sig]
    #     pix_scales_sig = compute_pix_scales_sig[mask_computable_sig]
    #     print('pix_scales_pc ', pix_scales_pc)
    #     print('pix_scales_sig ', pix_scales_sig)
    #
    #
    #     # compute pc scale maps
    #     scale_map_list_pc, residual_img_pc, kernel_sizes_list_pc = (
    #         phot_tools.ScaleTools.constrained_diffusion_decomposition_specific_scales(
    #             data=img, scales_pix_lo=pix_scales_lo_pc, scales_pix_hi=pix_scales_hi_pc, e_rel=3e-2, max_n=None,
    #             sm_mode='reflect', verbosity=True))
    #
    #     # save them the scale maps
    #     for pc_scale, scale in zip(pix_scales_pc, scale_map_list_pc):
    #         # create the hdu
    #         hdu = fits.PrimaryHDU(data=scale, header=wcs.to_header())
    #         hdul = fits.HDUList([hdu])
    #
    #         file_path = phangs_phot.get_phangs_band_pc_scale_map_filepath(pc_scale=pc_scale, band=hst_band, obs='hst', flux_unit=flux_unit)
    #         if not os.path.isdir(file_path.parent): os.makedirs(file_path.parent)
    #         print(file_path)
    #         hdul.writeto(file_path, overwrite=True)
    #         hdul.close()
    #
    #     # compute sigma scale maps
    #     scale_map_list_sig, residual_img_sig, kernel_sizes_list_sig = (
    #         phot_tools.ScaleTools.constrained_diffusion_decomposition_specific_scales(
    #             data=img, scales_pix_lo=pix_scales_lo_sig, scales_pix_hi=pix_scales_hi_sig, e_rel=3e-2, max_n=None,
    #             sm_mode='reflect', verbosity=True))
    #
    #     # save them the scale maps
    #     for sig_scale, scale in zip(pix_scales_sig, scale_map_list_sig):
    #         # create the hdu
    #         hdu = fits.PrimaryHDU(data=scale, header=wcs.to_header())
    #         hdul = fits.HDUList([hdu])
    #
    #         file_path = phangs_phot.get_phangs_band_sig_scale_map_filepath(sig_scale=sig_scale, band=hst_band, obs='hst', flux_unit=flux_unit)
    #         if not os.path.isdir(file_path.parent): os.makedirs(file_path.parent)
    #         print(file_path)
    #         hdul.writeto(file_path, overwrite=True)
    #         hdul.close()
    #
    #
    #     # plot results
    #     fontsize = 23
    #     fig_size_individual = (5, 5)
    #
    #     n_cols_pc = 4
    #     n_rows_pc = int(np.ceil(len(pix_scales_pc) / n_cols_pc) + 1)
    #
    #     n_cols_sig = 4
    #     n_rows_sig = int(np.ceil(len(pix_scales_sig) / n_cols_sig) + 1)
    #
    #     print('n_cols_pc ', n_cols_pc)
    #     print('n_rows_pc ', n_rows_pc)
    #
    #     print('n_cols_sig ', n_cols_sig)
    #     print('n_rows_sig ', n_rows_sig)
    #
    #     fig_pc, axs_pc = plt.subplots(ncols=n_cols_pc, nrows=n_rows_pc,
    #                                   figsize=(fig_size_individual[0] * n_cols_pc, fig_size_individual[1] * n_rows_pc))
    #     fig_sig, axs_sig = plt.subplots(ncols=n_cols_sig, nrows=n_rows_sig,
    #                                     figsize=(fig_size_individual[0] * n_cols_sig, fig_size_individual[1] * n_rows_sig))
    #     # get scale
    #     mean, median, std = sigma_clipped_stats(img, sigma=3.0)
    #     vmin = median - 1 * std
    #     vmax = median + 30 * std
    #     print(vmin, vmax)
    #     norm_img = ImageNormalize(stretch=LogStretch(), vmin=vmin, vmax=vmax)
    #
    #     # plot pc scale map
    #     axs_pc[0, 0].imshow(img, origin='lower', norm=norm_img, cmap='Greys')
    #     axs_pc[0, 0].set_title('original image', fontsize=fontsize)
    #     axs_pc[0, 1].imshow(residual_img_pc, origin='lower', norm=norm_img, cmap='Greys')
    #     axs_pc[0, 1].set_title('Residuals', fontsize=fontsize)
    #
    #     for idx_col in range(n_cols_pc - 2):
    #         axs_pc[0, idx_col + 2].axis('off')
    #
    #     running_scale_idx = 0
    #     for idx_row in range(n_rows_pc - 1):
    #         for idx_col in range(n_cols_pc):
    #             print(idx_row + 1, idx_col)
    #             print('running_scale_idx ', running_scale_idx)
    #             if running_scale_idx >= len(pix_scales_pc):
    #                 axs_pc[idx_row + 1, idx_col].axis('off')
    #             else:
    #                 axs_pc[idx_row + 1, idx_col].imshow(scale_map_list_pc[running_scale_idx], origin='lower', norm=norm_img,
    #                                                  cmap='Greys')
    #                 axs_pc[idx_row + 1, idx_col].set_title('Scale %i pc' % pix_scales_pc[running_scale_idx], fontsize=fontsize)
    #             running_scale_idx += 1
    #
    #     fig_pc.tight_layout()
    #
    #     fig_pc.savefig('plot_output/scale_decomposition_pc_%s_%s_n_scale_%i.png' % (target, hst_band, len(pix_scales_pc)))
    #     plt.close(fig_pc)
    #     plt.cla()
    #
    #
    #     # plot sig scale map
    #     axs_sig[0, 0].imshow(img, origin='lower', norm=norm_img, cmap='Greys')
    #     axs_sig[0, 0].set_title('original image', fontsize=fontsize)
    #     axs_sig[0, 1].imshow(residual_img_sig, origin='lower', norm=norm_img, cmap='Greys')
    #     axs_sig[0, 1].set_title('Residuals', fontsize=fontsize)
    #
    #     for idx_col in range(n_cols_sig - 2):
    #         axs_sig[0, idx_col + 2].axis('off')
    #
    #     running_scale_idx = 0
    #     for idx_row in range(n_rows_sig - 1):
    #         for idx_col in range(n_cols_sig):
    #             print(idx_row + 1, idx_col)
    #             print('running_scale_idx ', running_scale_idx)
    #             if running_scale_idx >= len(pix_scales_sig):
    #                 axs_sig[idx_row + 1, idx_col].axis('off')
    #             else:
    #                 axs_sig[idx_row + 1, idx_col].imshow(scale_map_list_sig[running_scale_idx], origin='lower', norm=norm_img,
    #                                                  cmap='Greys')
    #                 axs_sig[idx_row + 1, idx_col].set_title('Scale %i Sigma' % pix_scales_sig[running_scale_idx], fontsize=fontsize)
    #             running_scale_idx += 1
    #
    #     fig_sig.tight_layout()
    #
    #     fig_sig.savefig('plot_output/scale_decomposition_sig_%s_%s_n_scale_%i.png' % (target, hst_band, len(pix_scales_sig)))
    #     plt.close(fig_pc)
    #     plt.cla()
    #
    # for nircam_band in nircam_band_list:
    #     print(nircam_band)
    #     # load data
    #     phangs_phot.load_phangs_bands(band_list=[nircam_band], flux_unit=flux_unit)
    #     img = phangs_phot.nircam_bands_data['%s_data_img' % nircam_band]
    #     header = phangs_phot.nircam_bands_data['%s_header_img' % nircam_band]
    #     wcs = phangs_phot.nircam_bands_data['%s_wcs_img' % nircam_band]
    #     # get the smallest dimension of the data
    #     min_dim_img = np.min([header['NAXIS1'], header['NAXIS2']])
    #     # get pixel scale of data
    #     pixel_scale_arcsec = helper_func.CoordTools.transform_pix2world_scale(length_in_pix=1.0, wcs=wcs)
    #     # convert pixel scale to pc
    #     pixel_per_pc = np.pi * 1e6 / (3600 * 180) * pixel_scale_arcsec * distance_mpc  # convert to parcecs per pixel
    #     # get resolution of observations
    #     psf_dict = phot_tools.PSFTools.load_obs_psf_dict(band=nircam_band, instrument='nircam')
    #     gaussian_std_arcsec = psf_dict['gaussian_std']
    #     gaussian_std_pix = helper_func.CoordTools.transform_world2pix_scale(length_in_arcsec=gaussian_std_arcsec,
    #                                                                         wcs=wcs).value
    #     gaussian_std_per_pixel = 1 / gaussian_std_pix
    #     gaussian_std_in_pc = np.pi * 1e6 / (3600 * 180) * gaussian_std_arcsec * distance_mpc
    #
    #     print('pixel_per_pc ', pixel_per_pc)
    #     print('gaussian_std_arcsec ', gaussian_std_arcsec)
    #     print('gaussian_std_pix ', gaussian_std_pix)
    #     print('gaussian_std_per_pixel ', gaussian_std_per_pixel)
    #     print('gaussian_std_in_pc ', gaussian_std_in_pc)
    #
    #     pix_scales_lo_pc = compute_pix_scales_lo_pc / pixel_per_pc
    #     pix_scales_hi_pc = compute_pix_scales_hi_pc / pixel_per_pc
    #
    #     pix_scales_lo_sig = compute_pix_scales_lo_sig / gaussian_std_per_pixel
    #     pix_scales_hi_sig = compute_pix_scales_hi_sig / gaussian_std_per_pixel
    #
    #     # make sure the scales are working
    #     # the pc scale has to be at least 1.33 larger than the resolution
    #     # furthermore all the scales have to be smaller than the half of the smaller axis pixel size (You can not filter out
    #     # a scale that is larger than the image).
    #     mask_computable_pc = (pix_scales_lo_pc * pixel_per_pc >= 1.33 * gaussian_std_in_pc) & (
    #             pix_scales_hi_pc * pixel_per_pc / gaussian_std_in_pc <= 0.5 * min_dim_img)  # check that scales work
    #
    #     mask_computable_sig = (pix_scales_hi_sig * gaussian_std_per_pixel <= 0.5 * min_dim_img)  # check that scales work
    #
    #     pix_scales_lo_pc = pix_scales_lo_pc[mask_computable_pc]
    #     pix_scales_hi_pc = pix_scales_hi_pc[mask_computable_pc]
    #     pix_scales_pc = compute_pix_scales_pc[mask_computable_pc]
    #
    #     pix_scales_lo_sig = pix_scales_lo_sig[mask_computable_sig]
    #     pix_scales_hi_sig = pix_scales_hi_sig[mask_computable_sig]
    #     pix_scales_sig = compute_pix_scales_sig[mask_computable_sig]
    #     print('pix_scales_pc ', pix_scales_pc)
    #     print('pix_scales_sig ', pix_scales_sig)
    #
    #     # compute pc scale maps
    #     scale_map_list_pc, residual_img_pc, kernel_sizes_list_pc = (
    #         phot_tools.ScaleTools.constrained_diffusion_decomposition_specific_scales(
    #             data=img, scales_pix_lo=pix_scales_lo_pc, scales_pix_hi=pix_scales_hi_pc, e_rel=3e-2, max_n=None,
    #             sm_mode='reflect', verbosity=True))
    #
    #     # save them the scale maps
    #     for pc_scale, scale in zip(pix_scales_pc, scale_map_list_pc):
    #         # create the hdu
    #         hdu = fits.PrimaryHDU(data=scale, header=wcs.to_header())
    #         hdul = fits.HDUList([hdu])
    #
    #         file_path = phangs_phot.get_phangs_band_pc_scale_map_filepath(pc_scale=pc_scale, band=nircam_band, obs='nircam',
    #                                                                       flux_unit=flux_unit)
    #         if not os.path.isdir(file_path.parent): os.makedirs(file_path.parent)
    #         print(file_path)
    #         hdul.writeto(file_path, overwrite=True)
    #         hdul.close()
    #
    #     # compute sigma scale maps
    #     scale_map_list_sig, residual_img_sig, kernel_sizes_list_sig = (
    #         phot_tools.ScaleTools.constrained_diffusion_decomposition_specific_scales(
    #             data=img, scales_pix_lo=pix_scales_lo_sig, scales_pix_hi=pix_scales_hi_sig, e_rel=3e-2, max_n=None,
    #             sm_mode='reflect', verbosity=True))
    #
    #     # save them the scale maps
    #     for sig_scale, scale in zip(pix_scales_sig, scale_map_list_sig):
    #         # create the hdu
    #         hdu = fits.PrimaryHDU(data=scale, header=wcs.to_header())
    #         hdul = fits.HDUList([hdu])
    #
    #         file_path = phangs_phot.get_phangs_band_sig_scale_map_filepath(sig_scale=sig_scale, band=nircam_band, obs='nircam',
    #                                                                        flux_unit=flux_unit)
    #         if not os.path.isdir(file_path.parent): os.makedirs(file_path.parent)
    #         print(file_path)
    #         hdul.writeto(file_path, overwrite=True)
    #         hdul.close()
    #
    #     # plot results
    #     fontsize = 23
    #     fig_size_individual = (5, 5)
    #
    #     n_cols_pc = 4
    #     n_rows_pc = int(np.ceil(len(pix_scales_pc) / n_cols_pc) + 1)
    #
    #     n_cols_sig = 4
    #     n_rows_sig = int(np.ceil(len(pix_scales_sig) / n_cols_sig) + 1)
    #
    #     print('n_cols_pc ', n_cols_pc)
    #     print('n_rows_pc ', n_rows_pc)
    #
    #     print('n_cols_sig ', n_cols_sig)
    #     print('n_rows_sig ', n_rows_sig)
    #
    #     fig_pc, axs_pc = plt.subplots(ncols=n_cols_pc, nrows=n_rows_pc,
    #                                   figsize=(fig_size_individual[0] * n_cols_pc, fig_size_individual[1] * n_rows_pc))
    #     fig_sig, axs_sig = plt.subplots(ncols=n_cols_sig, nrows=n_rows_sig,
    #                                     figsize=(fig_size_individual[0] * n_cols_sig, fig_size_individual[1] * n_rows_sig))
    #     # get scale
    #     mean, median, std = sigma_clipped_stats(img, sigma=3.0)
    #     vmin = median - 1 * std
    #     vmax = median + 30 * std
    #     print(vmin, vmax)
    #     norm_img = ImageNormalize(stretch=LogStretch(), vmin=vmin, vmax=vmax)
    #
    #     # plot pc scale map
    #     axs_pc[0, 0].imshow(img, origin='lower', norm=norm_img, cmap='Greys')
    #     axs_pc[0, 0].set_title('original image', fontsize=fontsize)
    #     axs_pc[0, 1].imshow(residual_img_pc, origin='lower', norm=norm_img, cmap='Greys')
    #     axs_pc[0, 1].set_title('Residuals', fontsize=fontsize)
    #
    #     for idx_col in range(n_cols_pc - 2):
    #         axs_pc[0, idx_col + 2].axis('off')
    #
    #     running_scale_idx = 0
    #     for idx_row in range(n_rows_pc - 1):
    #         for idx_col in range(n_cols_pc):
    #             print(idx_row + 1, idx_col)
    #             print('running_scale_idx ', running_scale_idx)
    #             if running_scale_idx >= len(pix_scales_pc):
    #                 axs_pc[idx_row + 1, idx_col].axis('off')
    #             else:
    #                 axs_pc[idx_row + 1, idx_col].imshow(scale_map_list_pc[running_scale_idx], origin='lower', norm=norm_img,
    #                                                     cmap='Greys')
    #                 axs_pc[idx_row + 1, idx_col].set_title('Scale %i pc' % pix_scales_pc[running_scale_idx],
    #                                                        fontsize=fontsize)
    #             running_scale_idx += 1
    #
    #     fig_pc.tight_layout()
    #
    #     fig_pc.savefig('plot_output/scale_decomposition_pc_%s_%s_n_scale_%i.png' % (target, nircam_band, len(pix_scales_pc)))
    #     plt.close(fig_pc)
    #     plt.cla()
    #
    #     # plot sig scale map
    #     axs_sig[0, 0].imshow(img, origin='lower', norm=norm_img, cmap='Greys')
    #     axs_sig[0, 0].set_title('original image', fontsize=fontsize)
    #     axs_sig[0, 1].imshow(residual_img_sig, origin='lower', norm=norm_img, cmap='Greys')
    #     axs_sig[0, 1].set_title('Residuals', fontsize=fontsize)
    #
    #     for idx_col in range(n_cols_sig - 2):
    #         axs_sig[0, idx_col + 2].axis('off')
    #
    #     running_scale_idx = 0
    #     for idx_row in range(n_rows_sig - 1):
    #         for idx_col in range(n_cols_sig):
    #             print(idx_row + 1, idx_col)
    #             print('running_scale_idx ', running_scale_idx)
    #             if running_scale_idx >= len(pix_scales_sig):
    #                 axs_sig[idx_row + 1, idx_col].axis('off')
    #             else:
    #                 axs_sig[idx_row + 1, idx_col].imshow(scale_map_list_sig[running_scale_idx], origin='lower',
    #                                                      norm=norm_img,
    #                                                      cmap='Greys')
    #                 axs_sig[idx_row + 1, idx_col].set_title('Scale %i Sigma' % pix_scales_sig[running_scale_idx],
    #                                                         fontsize=fontsize)
    #             running_scale_idx += 1
    #
    #     fig_sig.tight_layout()
    #
    #     fig_sig.savefig(
    #         'plot_output/scale_decomposition_sig_%s_%s_n_scale_%i.png' % (target, nircam_band, len(pix_scales_sig)))
    #     plt.close(fig_pc)
    #     plt.cla()
    #
    #

    for miri_band in miri_band_list:

        print(miri_band)
        # load data
        phangs_phot.load_phangs_bands(band_list=[miri_band], flux_unit=flux_unit)
        img = phangs_phot.miri_bands_data['%s_data_img' % miri_band]
        header = phangs_phot.miri_bands_data['%s_header_img' % miri_band]
        wcs = phangs_phot.miri_bands_data['%s_wcs_img' % miri_band]
        # get the smallest dimension of the data
        min_dim_img = np.min([header['NAXIS1'], header['NAXIS2']])
        # get pixel scale of data
        pixel_scale_arcsec = helper_func.CoordTools.transform_pix2world_scale(length_in_pix=1.0, wcs=wcs)
        # convert pixel scale to pc
        pixel_per_pc = np.pi * 1e6 / (3600 * 180) * pixel_scale_arcsec * distance_mpc  # convert to parcecs per pixel
        # get resolution of observations
        psf_dict = phot_tools.PSFTools.load_obs_psf_dict(band=miri_band, instrument='miri')
        gaussian_std_arcsec = psf_dict['gaussian_std']
        gaussian_std_pix = helper_func.CoordTools.transform_world2pix_scale(length_in_arcsec=gaussian_std_arcsec,
                                                                            wcs=wcs).value
        gaussian_std_per_pixel = 1 / gaussian_std_pix
        gaussian_std_in_pc = np.pi * 1e6 / (3600 * 180) * gaussian_std_arcsec * distance_mpc

        print('pixel_per_pc ', pixel_per_pc)
        print('gaussian_std_arcsec ', gaussian_std_arcsec)
        print('gaussian_std_pix ', gaussian_std_pix)
        print('gaussian_std_per_pixel ', gaussian_std_per_pixel)
        print('gaussian_std_in_pc ', gaussian_std_in_pc)

        pix_scales_lo_pc = compute_pix_scales_lo_pc / pixel_per_pc
        pix_scales_hi_pc = compute_pix_scales_hi_pc / pixel_per_pc

        pix_scales_lo_sig = compute_pix_scales_lo_sig / gaussian_std_per_pixel
        pix_scales_hi_sig = compute_pix_scales_hi_sig / gaussian_std_per_pixel

        # make sure the scales are working
        # the pc scale has to be at least 1.33 larger than the resolution
        # furthermore all the scales have to be smaller than the half of the smaller axis pixel size (You can not filter out
        # a scale that is larger than the image).
        mask_computable_pc = (pix_scales_lo_pc * pixel_per_pc >= 1.33 * gaussian_std_in_pc) & (
                pix_scales_hi_pc * pixel_per_pc / gaussian_std_in_pc <= 0.5 * min_dim_img)  # check that scales work

        mask_computable_sig = (pix_scales_hi_sig * gaussian_std_per_pixel <= 0.5 * min_dim_img)  # check that scales work

        pix_scales_lo_pc = pix_scales_lo_pc[mask_computable_pc]
        pix_scales_hi_pc = pix_scales_hi_pc[mask_computable_pc]
        pix_scales_pc = compute_pix_scales_pc[mask_computable_pc]

        pix_scales_lo_sig = pix_scales_lo_sig[mask_computable_sig]
        pix_scales_hi_sig = pix_scales_hi_sig[mask_computable_sig]
        pix_scales_sig = compute_pix_scales_sig[mask_computable_sig]
        print('pix_scales_pc ', pix_scales_pc)
        print('pix_scales_sig ', pix_scales_sig)

        # check if scale is already computed
        skip_compute_pc_flag = True
        if recompute_pc_flag:
            skip_compute_pc_flag = False
        for pc_scale in pix_scales_pc:
            file_path = phangs_phot.get_phangs_band_pc_scale_map_filepath(pc_scale=pc_scale, band=miri_band, obs='miri',
                                                                          flux_unit=flux_unit)
            if not os.path.isfile(file_path):
                skip_compute_pc_flag *= False

        if not skip_compute_pc_flag:
            # compute pc scale maps
            scale_map_list_pc, residual_img_pc, kernel_sizes_list_pc = (
                phot_tools.ScaleTools.constrained_diffusion_decomposition_specific_scales(
                    data=img, scales_pix_lo=pix_scales_lo_pc, scales_pix_hi=pix_scales_hi_pc, e_rel=3e-2, max_n=None,
                    sm_mode='reflect', verbosity=True))

            # save them the scale maps
            for pc_scale, scale in zip(pix_scales_pc, scale_map_list_pc):
                # create the hdu
                hdu = fits.PrimaryHDU(data=scale, header=wcs.to_header())
                hdul = fits.HDUList([hdu])

                file_path = phangs_phot.get_phangs_band_pc_scale_map_filepath(pc_scale=pc_scale, band=miri_band, obs='miri',
                                                                              flux_unit=flux_unit)
                if not os.path.isdir(file_path.parent): os.makedirs(file_path.parent)
                print(file_path)
                hdul.writeto(file_path, overwrite=True)
                hdul.close()

        # check if scale is already computed
        skip_compute_sig_flag = True
        if recompute_sig_flag:
            skip_compute_sig_flag = False
        for sig_scale in pix_scales_sig:
            file_path = phangs_phot.get_phangs_band_sig_scale_map_filepath(sig_scale=sig_scale, band=miri_band, obs='miri',
                                                                          flux_unit=flux_unit)
            if not os.path.isfile(file_path):
                skip_compute_sig_flag *= False
        if not skip_compute_sig_flag:
            # compute sigma scale maps
            scale_map_list_sig, residual_img_sig, kernel_sizes_list_sig = (
                phot_tools.ScaleTools.constrained_diffusion_decomposition_specific_scales(
                    data=img, scales_pix_lo=pix_scales_lo_sig, scales_pix_hi=pix_scales_hi_sig, e_rel=3e-2, max_n=None,
                    sm_mode='reflect', verbosity=True))

            # save them the scale maps
            for sig_scale, scale in zip(pix_scales_sig, scale_map_list_sig):
                # create the hdu
                hdu = fits.PrimaryHDU(data=scale, header=wcs.to_header())
                hdul = fits.HDUList([hdu])

                file_path = phangs_phot.get_phangs_band_sig_scale_map_filepath(sig_scale=sig_scale, band=miri_band, obs='miri',
                                                                               flux_unit=flux_unit)
                if not os.path.isdir(file_path.parent): os.makedirs(file_path.parent)
                print(file_path)
                hdul.writeto(file_path, overwrite=True)
                hdul.close()

        # plot results
        fontsize = 23
        fig_size_individual = (5, 5)

        n_cols_pc = 4
        n_rows_pc = int(np.ceil(len(pix_scales_pc) / n_cols_pc) + 1)

        n_cols_sig = 4
        n_rows_sig = int(np.ceil(len(pix_scales_sig) / n_cols_sig) + 1)

        print('n_cols_pc ', n_cols_pc)
        print('n_rows_pc ', n_rows_pc)

        print('n_cols_sig ', n_cols_sig)
        print('n_rows_sig ', n_rows_sig)

        if not skip_compute_pc_flag:


            fig_pc, axs_pc = plt.subplots(ncols=n_cols_pc, nrows=n_rows_pc,
                                          figsize=(fig_size_individual[0] * n_cols_pc, fig_size_individual[1] * n_rows_pc))

            # get scale
            mean, median, std = sigma_clipped_stats(img, sigma=3.0)
            vmin = median - 1 * std
            vmax = median + 30 * std
            print(vmin, vmax)
            norm_img = ImageNormalize(stretch=LogStretch(), vmin=vmin, vmax=vmax)

            # plot pc scale map
            axs_pc[0, 0].imshow(img, origin='lower', norm=norm_img, cmap='Greys')
            axs_pc[0, 0].set_title('original image', fontsize=fontsize)
            axs_pc[0, 1].imshow(residual_img_pc, origin='lower', norm=norm_img, cmap='Greys')
            axs_pc[0, 1].set_title('Residuals', fontsize=fontsize)

            for idx_col in range(n_cols_pc - 2):
                axs_pc[0, idx_col + 2].axis('off')

            running_scale_idx = 0
            for idx_row in range(n_rows_pc - 1):
                for idx_col in range(n_cols_pc):
                    print(idx_row + 1, idx_col)
                    print('running_scale_idx ', running_scale_idx)
                    if running_scale_idx >= len(pix_scales_pc):
                        axs_pc[idx_row + 1, idx_col].axis('off')
                    else:

                        mean, median, std = sigma_clipped_stats(scale_map_list_pc[running_scale_idx], sigma=3.0)
                        vmin = median - 1 * std
                        vmax = median + 30 * std
                        print(vmin, vmax)
                        norm_img = ImageNormalize(stretch=LogStretch(), vmin=vmin, vmax=vmax)

                        axs_pc[idx_row + 1, idx_col].imshow(scale_map_list_pc[running_scale_idx], origin='lower', norm=norm_img,
                                                            cmap='Greys')
                        axs_pc[idx_row + 1, idx_col].set_title('Scale %i pc' % pix_scales_pc[running_scale_idx],
                                                               fontsize=fontsize)
                    running_scale_idx += 1

            fig_pc.tight_layout()

            fig_pc.savefig('plot_output/scale_decomposition_pc_%s_%s_n_scale_%i.png' % (target, miri_band, len(pix_scales_pc)))
            plt.close(fig_pc)
            plt.cla()


        if not recompute_sig_flag:
            fig_sig, axs_sig = plt.subplots(ncols=n_cols_sig, nrows=n_rows_sig,
                                            figsize=(fig_size_individual[0] * n_cols_sig, fig_size_individual[1] * n_rows_sig))


            mean, median, std = sigma_clipped_stats(img, sigma=3.0)
            vmin = median - 1 * std
            vmax = median + 30 * std
            print(vmin, vmax)
            norm_img = ImageNormalize(stretch=LogStretch(), vmin=vmin, vmax=vmax)

            # plot sig scale map
            axs_sig[0, 0].imshow(img, origin='lower', norm=norm_img, cmap='Greys')
            axs_sig[0, 0].set_title('original image', fontsize=fontsize)
            axs_sig[0, 1].imshow(residual_img_sig, origin='lower', norm=norm_img, cmap='Greys')
            axs_sig[0, 1].set_title('Residuals', fontsize=fontsize)

            for idx_col in range(n_cols_sig - 2):
                axs_sig[0, idx_col + 2].axis('off')

            running_scale_idx = 0
            for idx_row in range(n_rows_sig - 1):
                for idx_col in range(n_cols_sig):
                    print(idx_row + 1, idx_col)
                    print('running_scale_idx ', running_scale_idx)
                    if running_scale_idx >= len(pix_scales_sig):
                        axs_sig[idx_row + 1, idx_col].axis('off')
                    else:
                        mean, median, std = sigma_clipped_stats(scale_map_list_sig[running_scale_idx], sigma=3.0)
                        vmin = median - 1 * std
                        vmax = median + 30 * std
                        print(vmin, vmax)
                        norm_img = ImageNormalize(stretch=LogStretch(), vmin=vmin, vmax=vmax)

                        axs_sig[idx_row + 1, idx_col].imshow(scale_map_list_sig[running_scale_idx], origin='lower',
                                                             norm=norm_img,
                                                             cmap='Greys')
                        axs_sig[idx_row + 1, idx_col].set_title('Scale %i Sigma' % pix_scales_sig[running_scale_idx],
                                                                fontsize=fontsize)
                    running_scale_idx += 1

            fig_sig.tight_layout()

            fig_sig.savefig(
                'plot_output/scale_decomposition_sig_%s_%s_n_scale_%i.png' % (target, miri_band, len(pix_scales_sig)))
            plt.close(fig_pc)
            plt.cla()




