"""
Script to develop how to check the observational coverage of a PHANGS target
"""
import matplotlib.pyplot as plt
import numpy as np
import os
from phangs_data_access import phot_access
from phangs_data_access import helper_func
from phangs_data_access import phangs_info


target_list = phangs_info.astrosat_obs_band_dict.keys()

# print(target_list)
# print(phangs_info.hst_ha_obs_band_dict.keys())
# exit()


for target in target_list:

    # if os.path.isfile('data_output/%s_astrosat_obs_hull_dict.npy' % target):
    #     continue


    # now get astrosat bands
    phangs_phot = phot_access.PhotAccess(target_name=target)

    # get band list
    band_list = helper_func.BandTools.get_astrosat_obs_band_list(target=target)

    print(target, ' bands, available: ', band_list)
    phangs_phot.load_phangs_bands(band_list=band_list)

    obs_hull_dict = {}
    for band in band_list:

        img_data = phangs_phot.astrosat_bands_data['%s_data_img' % band]
        img_wcs = phangs_phot.astrosat_bands_data['%s_wcs_img' % band]

        # the field of view is about 28 arcmin
        fov_rad_arcsec = 14 * 60
        # get central coordinate
        central_x_pixel = img_data.shape[0]/2
        central_y_pixel = img_data.shape[1]/2

        fov_rad_pix = helper_func.CoordTools.transform_world2pix_scale(length_in_arcsec=fov_rad_arcsec, wcs=img_wcs,
                                                                       dim=0)

        angle_points = np.linspace(0, 2 * np.pi, 1000)
        x_convex_hull = central_x_pixel + fov_rad_pix * np.sin(angle_points)
        y_convex_hull = central_y_pixel + fov_rad_pix * np.cos(angle_points)

        # transform into coordinates
        coordinates = img_wcs.pixel_to_world(x_convex_hull, y_convex_hull)
        ra = coordinates.ra.deg
        dec = coordinates.dec.deg
        obs_hull_dict.update({band: {0: {'ra': ra, 'dec': dec}}})

    # save dictionary
    np.save('data_output/%s_astrosat_obs_hull_dict.npy' % target, obs_hull_dict)


