"""
Script to develop how to check the observational coverage of a PHANGS target
"""
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
from phangs_data_access import phot_access
from phangs_data_access import helper_func
from phangs_data_access import phangs_info


plot_flag = True

astrosat_version = 'v1p0'

target_list = list(getattr(phangs_info, 'astrosat_obs_band_dict_%s' % astrosat_version).keys())

print(target_list)

for target in target_list:

    # if os.path.isfile('data_output/%s_astrosat_obs_hull_dict.npy' % target):
    #     continue


    # now get astrosat bands
    phangs_phot = phot_access.PhotAccess(phot_target_name=target, phot_nircam_target_name=target, astrosat_data_ver=astrosat_version)

    # get band list
    band_list = helper_func.ObsTools.get_astrosat_obs_band_list(target=target, version=astrosat_version)

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
    if not os.path.isdir('data_output'):
        os.makedirs('data_output')

    with open('data_output/%s_astrosat_obs_hull_dict_%s.npy' % (target, astrosat_version), 'wb') as file_name:
        pickle.dump(obs_hull_dict, file_name)



