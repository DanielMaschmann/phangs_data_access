"""
create all needed data for the HST PSFs
"""
import numpy as np
import os
from astropy.io import fits
from phangs_data_access import phot_tools, phot_access, helper_func, sample_access
import matplotlib.pyplot as plt
import pickle
from astropy.visualization import SqrtStretch, SinhStretch, LogStretch
from astropy.visualization.mpl_normalize import ImageNormalize
from astropy.stats import sigma_clipped_stats



# parameters needed for procedure
flux_unit = 'mJy'

target = 'ngc0628c'

# initialize data access
phangs_phot = phot_access.PhotAccess(phot_target_name=target)

phangs_sample = sample_access.SampleAccess()

# getting the distance
distance_mpc = phangs_sample.get_target_dist(target=helper_func.FileTools.target_name_no_directions(target))

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

miri_band = 'F770W'

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

print('pix_scales_pc ', pix_scales_pc / pixel_per_pc)
print('pix_scales_sig ', pix_scales_sig / gaussian_std_per_pixel)

# compute sigma scale maps
scale_map_list_sig, residual_img_sig, kernel_sizes_list_sig = (
    phot_tools.ScaleTools.constrained_diffusion_decomposition_specific_scales(
        data=img, scales_pix_lo=pix_scales_lo_sig, scales_pix_hi=pix_scales_hi_sig, e_rel=3e-2, max_n=None,
        sm_mode='reflect', verbosity=True))

# exit()
#
# scale_data_sig1, scale_header_sig1, scale_wcs_sig1 = phangs_phot.get_phangs_band_sig_scale_map(sig_scale=1, band=miri_band, obs='miri', flux_unit=flux_unit)
# scale_data_sig2, scale_header_sig2, scale_wcs_sig2 = phangs_phot.get_phangs_band_sig_scale_map(sig_scale=2, band=miri_band, obs='miri', flux_unit=flux_unit)
# scale_data_sig4, scale_header_sig4, scale_wcs_sig4 = phangs_phot.get_phangs_band_sig_scale_map(sig_scale=4, band=miri_band, obs='miri', flux_unit=flux_unit)




plt.imshow(np.log10(scale_map_list_sig[0]))
plt.show()

plt.imshow(np.log10(scale_map_list_sig[1]))
plt.show()

plt.imshow(np.log10(scale_map_list_sig[0] + scale_map_list_sig[1]))
plt.show()


plt.imshow(np.log10(scale_map_list_sig[2]))
plt.show()

exit()






hst_band_list = helper_func.ObsTools.get_hst_obs_band_list(target=phangs_phot.phot_hst_target_name)
nircam_band_list = helper_func.ObsTools.get_nircam_obs_band_list(target=phangs_phot.phot_nircam_target_name)
miri_band_list = helper_func.ObsTools.get_miri_obs_band_list(target=phangs_phot.phot_miri_target_name)

phangs_phot.load_phangs_bands(band_list=['F770W'], flux_unit=flux_unit)
img = phangs_phot.miri_bands_data['%s_data_img' % 'F770W']
header = phangs_phot.miri_bands_data['%s_header_img' % 'F770W']
wcs = phangs_phot.miri_bands_data['%s_wcs_img' % 'F770W']

pixel_scale = helper_func.CoordTools.transform_pix2world_scale(length_in_pix=1.0, wcs=wcs)


print('pixel_scale ', pixel_scale)

distance_mpc = 9.84
res = 0.27
pixscale = 0.11
min_power = 3
max_power = 8

min_dim_img = np.min([header['NAXIS1'], header['NAXIS2']])

print(np.pi * 1e6 / (3600 * 180))


pix_pc = np.pi * 1e6 / (3600 * 180) * pixscale * distance_mpc  # convert to parcecs per pixel
print('pix_pc ', pix_pc)

print(range(min_power, max_power + 1))


# generate scales to decompose image into
pixscales = (2.0 ** np.array(range(min_power, max_power + 1))) / pix_pc
pcscales = pixscales * pix_pc
pixscales_lo = (2.0 ** (np.array(range(min_power, max_power + 1)) - 0.5)) / pix_pc
pixscales_hi = (2.0 ** (np.array(range(min_power, max_power + 1)) + 0.5)) / pix_pc
res_pc = np.pi * 1e6 / (3600 * 180) * res * distance_mpc

print('res_pc ', res_pc)


idx = (pixscales_lo * pix_pc >= 1.33 * res_pc / 2.35) & (
            pixscales_hi * pix_pc / res_pc <= 0.5 * min_dim_img / 2.35)  # check that scales work
pixscales_hi = pixscales_hi[idx]
pixscales_lo = pixscales_lo[idx]
pixscales = pixscales[idx]
pcscales = pcscales[idx]
print(f"pc ranges: {pixscales * pix_pc}")

exit()

print('pixscales_hi ', pixscales_hi)
print('pixscales_lo ', pixscales_lo)
print('pixscales ', pixscales)

scale_map_list, residual_img, kernel_sizes_list = (
    phot_tools.ScaleTools.constrained_diffusion_decomposition_specific_scales(
        data=img, scales_pix=pixscales, scales_pix_lo=pixscales_lo, scales_pix_hi=pixscales_hi,
        e_rel=3e-2,
        max_n=5, sm_mode='reflect', verbosity=False))


exit()




for hst_band in hst_band_list:
    print(hst_band)
    # load data
    phangs_phot.load_phangs_bands(band_list=[hst_band], flux_unit=flux_unit)
    img = phangs_phot.hst_bands_data['%s_data_img' % hst_band]
    wcs = phangs_phot.hst_bands_data['%s_wcs_img' % hst_band]

    psf_dict = phot_tools.PSFTools.load_obs_psf_dict(
        band=hst_band,
        instrument=helper_func.ObsTools.get_hst_instrument(target=phangs_phot.phot_hst_target_name, band=hst_band))
    gaussian_std_arcsec = psf_dict['gaussian_std']
    gaussian_std_pix = helper_func.CoordTools.transform_world2pix_scale(length_in_arcsec=gaussian_std_arcsec, wcs=wcs).value
    print('gaussian_std_arcsec ', gaussian_std_arcsec)
    print('gaussian_std_pix ', gaussian_std_pix)

    min_power = -1
    max_power = 4

    # pix_pc = 4.848 * pixscale * distance_mpc  # convert to parcecs per pixel

    # generate scales to decompose image into
    pixscales = (2.0 ** np.array(range(min_power, max_power + 1))) * gaussian_std_pix
    # pcscales = pixscales * pix_pc
    pixscales_lo = (2.0 ** (np.array(range(min_power, max_power + 1)) - 0.5)) * gaussian_std_pix
    pixscales_hi = (2.0 ** (np.array(range(min_power, max_power + 1)) + 0.5)) * gaussian_std_pix

    # res_pc = 4.848 * res * distance_mpc
    # idx = (pixscales_lo * pix_pc >= 1.33 * res_pc / 2.35) & (pixscales_hi * pix_pc / res_pc <= 0.5 * min_dim_img / 2.35)  # check that scales work
    # pixscales_hi = pixscales_hi[idx]
    # pixscales_lo = pixscales_lo[idx]
    # pixscales = pixscales[idx]
    # pcscales = pcscales[idx]
    print('pixscales_lo ', pixscales_lo)
    print('pixscales_hi ', pixscales_hi)

    print(f"pc ranges: {pixscales}")



    print(scale_map_list, residual_img, kernel_sizes_list)

    exit()

    # compute scale maps
    scale_map_list, residual_img, kernel_sizes_list = phot_tools.ScaleTools.constrained_diffusion_decomposition(
        img, max_n=n_scale_maps, verbosity=True)

    # save image
    for n_scale, scale in enumerate(scale_map_list):
        # create the hdu
        hdu = fits.PrimaryHDU(data=scale, header=wcs.to_header())
        hdul = fits.HDUList([hdu])

        file_path = phangs_phot.get_phangs_band_scale_map_filepath(n_scale=n_scale, band=hst_band, instrument='hst', flux_unit=flux_unit)
        if not os.path.isdir(file_path.parent): os.makedirs(file_path.parent)
        print(file_path)
        hdul.writeto(file_path, overwrite=True)
        hdul.close()

    # plot scale maps
    fontsize = 23
    fig_size_individual = (5, 5)
    n_cols = 4
    n_rows = int(np.rint((n_scale_maps) / n_cols) + 1)

    print(n_cols)
    print(n_rows)

    fig, axs = plt.subplots(ncols=n_cols, nrows=n_rows,
                            figsize=(fig_size_individual[0] * n_cols, fig_size_individual[1] * n_rows))

    mean, median, std = sigma_clipped_stats(img, sigma=3.0)
    vmin = median - 1 * std
    vmax = median + 30 * std
    print(vmin, vmax)

    norm_img = ImageNormalize(stretch=LogStretch(), vmin=vmin, vmax=vmax)

    axs[0, 0].imshow(img, origin='lower', norm=norm_img, cmap='Greys')
    axs[0, 0].set_title('original image', fontsize=fontsize)
    axs[0, 1].imshow(residual_img, origin='lower', norm=norm_img, cmap='Greys')
    axs[0, 1].set_title('Residuals', fontsize=fontsize)

    for idx_col in range(n_cols - 2):
        axs[0, idx_col + 2].axis('off')

    running_scale_idx = 0
    for idx_row in range(n_rows - 1):
        for idx_col in range(n_cols):
            print(idx_row + 1, idx_col)
            print('running_scale_idx ', running_scale_idx)
            if running_scale_idx >= n_scale_maps:
                axs[idx_row + 1, idx_col].axis('off')
            else:
                axs[idx_row + 1, idx_col].imshow(scale_map_list[running_scale_idx], origin='lower', norm=norm_img,
                                                 cmap='Greys')
                axs[idx_row + 1, idx_col].set_title('Scale %i' % (running_scale_idx + 1), fontsize=fontsize)
            running_scale_idx += 1

    fig.tight_layout()

    fig.savefig('plot_output/scale_decomposition_%s_%s_n_scale_%i.png' % (target, hst_band, n_scale_maps))
    plt.close(fig)
    plt.cla()











for nircam_band in nircam_band_list:
    print(nircam_band)
    # load data
    phangs_phot.load_phangs_bands(band_list=[nircam_band], flux_unit=flux_unit)
    img = phangs_phot.nircam_bands_data['%s_data_img' % nircam_band]
    wcs = phangs_phot.nircam_bands_data['%s_wcs_img' % nircam_band]

    # compute scale maps
    scale_map_list, residual_img, kernel_sizes_list = phot_tools.ScaleTools.constrained_diffusion_decomposition(
        img, max_n=n_scale_maps, verbosity=True)

    # save image
    for n_scale, scale in enumerate(scale_map_list):
        # create the hdu
        hdu = fits.PrimaryHDU(data=scale, header=wcs.to_header())
        hdul = fits.HDUList([hdu])

        file_path = phangs_phot.get_phangs_band_scale_map_filepath(n_scale=n_scale, band=nircam_band, instrument='nircam', flux_unit=flux_unit)
        if not os.path.isdir(file_path.parent): os.makedirs(file_path.parent)
        print(file_path)
        hdul.writeto(file_path, overwrite=True)
        hdul.close()

    # plot scale maps
    fontsize = 23
    fig_size_individual = (5, 5)
    n_cols = 4
    n_rows = int(np.rint((n_scale_maps) / n_cols) + 1)

    fig, axs = plt.subplots(ncols=n_cols, nrows=n_rows,
                            figsize=(fig_size_individual[0] * n_cols, fig_size_individual[1] * n_rows))

    mean, median, std = sigma_clipped_stats(img, sigma=3.0)
    vmin = median - 1 * std
    vmax = median + 30 * std
    print(vmin, vmax)

    norm_img = ImageNormalize(stretch=LogStretch(), vmin=vmin, vmax=vmax)

    axs[0, 0].imshow(img, origin='lower', norm=norm_img, cmap='Greys')
    axs[0, 0].set_title('original image', fontsize=fontsize)
    axs[0, 1].imshow(residual_img, origin='lower', norm=norm_img, cmap='Greys')
    axs[0, 1].set_title('Residuals', fontsize=fontsize)

    for idx_col in range(n_cols - 2):
        axs[0, idx_col + 2].axis('off')

    running_scale_idx = 0
    for idx_row in range(n_rows - 1):
        for idx_col in range(n_cols):
            print(idx_row + 1, idx_col)
            print('running_scale_idx ', running_scale_idx)
            if running_scale_idx >= n_scale_maps:
                axs[idx_row + 1, idx_col].axis('off')
            else:
                axs[idx_row + 1, idx_col].imshow(scale_map_list[running_scale_idx], origin='lower', norm=norm_img,
                                                 cmap='Greys')
                axs[idx_row + 1, idx_col].set_title('Scale %i' % (running_scale_idx + 1), fontsize=fontsize)
            running_scale_idx += 1
    fig.tight_layout()

    fig.savefig('plot_output/scale_decomposition_%s_%s_n_scale_%i.png' % (target, nircam_band, n_scale_maps))
    plt.close(fig)
    plt.cla()



for miri_band in miri_band_list:
    print(miri_band)
    # load data
    phangs_phot.load_phangs_bands(band_list=[miri_band], flux_unit=flux_unit)
    img = phangs_phot.miri_bands_data['%s_data_img' % miri_band]
    wcs = phangs_phot.miri_bands_data['%s_wcs_img' % miri_band]

    # compute scale maps
    scale_map_list, residual_img, kernel_sizes_list = phot_tools.ScaleTools.constrained_diffusion_decomposition(
        img, max_n=n_scale_maps, verbosity=True)

    # save image
    for n_scale, scale in enumerate(scale_map_list):
        # create the hdu
        hdu = fits.PrimaryHDU(data=scale, header=wcs.to_header())
        hdul = fits.HDUList([hdu])

        file_path = phangs_phot.get_phangs_band_scale_map_filepath(n_scale=n_scale, band=miri_band, instrument='miri', flux_unit=flux_unit)
        if not os.path.isdir(file_path.parent): os.makedirs(file_path.parent)
        print(file_path)
        hdul.writeto(file_path, overwrite=True)
        hdul.close()

    # plot scale maps
    fontsize = 23
    fig_size_individual = (5, 5)
    n_cols = 4
    n_rows = int(np.rint((n_scale_maps) / n_cols) + 1)

    fig, axs = plt.subplots(ncols=n_cols, nrows=n_rows,
                            figsize=(fig_size_individual[0] * n_cols, fig_size_individual[1] * n_rows))

    mean, median, std = sigma_clipped_stats(img, sigma=3.0)
    vmin = median - 1 * std
    vmax = median + 30 * std
    print(vmin, vmax)

    norm_img = ImageNormalize(stretch=LogStretch(), vmin=vmin, vmax=vmax)

    axs[0, 0].imshow(img, origin='lower', norm=norm_img, cmap='Greys')
    axs[0, 0].set_title('original image', fontsize=fontsize)
    axs[0, 1].imshow(residual_img, origin='lower', norm=norm_img, cmap='Greys')
    axs[0, 1].set_title('Residuals', fontsize=fontsize)

    for idx_col in range(n_cols - 2):
        axs[0, idx_col + 2].axis('off')

    running_scale_idx = 0
    for idx_row in range(n_rows - 1):
        for idx_col in range(n_cols):
            print(idx_row + 1, idx_col)
            print('running_scale_idx ', running_scale_idx)
            if running_scale_idx >= n_scale_maps:
                axs[idx_row + 1, idx_col].axis('off')
            else:
                axs[idx_row + 1, idx_col].imshow(scale_map_list[running_scale_idx], origin='lower', norm=norm_img,
                                                 cmap='Greys')
                axs[idx_row + 1, idx_col].set_title('Scale %i' % (running_scale_idx + 1), fontsize=fontsize)
            running_scale_idx += 1

    fig.tight_layout()

    fig.savefig('plot_output/scale_decomposition_%s_%s_n_scale_%i.png' % (target, miri_band, n_scale_maps))
    plt.close(fig)
    plt.cla()


