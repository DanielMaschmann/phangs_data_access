"""
Script to create script to rclone PHANGS HST observational data from google drive
"""
import os.path
from pathlib import Path
from phangs_data_access import phangs_info, phangs_access_config, helper_func

nircam_version = 'v2p0'
miri_version = 'v2p0'

target_list = list(getattr(phangs_info, 'jwst_obs_band_dict_%s' % nircam_version).keys())

# target_list = phangs_info.jwst_obs_band_dict.keys()

rclone_name = phangs_access_config.phangs_config_dict['rclone_name']

drive_path = 'rclone copy drive:'

jwst_rclone_file = open('download_scripts/rclone_jwst_data.sh', "w")

for target in target_list:

    path_str_nircam = ('rclone copy ' + phangs_access_config.phangs_config_dict['rclone_name'] + ':' +
                phangs_access_config.phangs_config_dict['jwst_obs_data_drive_path_%s' % nircam_version] +
                target + '/')
    path_str_miri = ('rclone copy ' + phangs_access_config.phangs_config_dict['rclone_name'] + ':' +
                phangs_access_config.phangs_config_dict['jwst_obs_data_drive_path_%s' % miri_version] +
                target + '/')

    destination_str_nircam = (phangs_access_config.phangs_config_dict['jwst_obs_data_local_path_%s' % nircam_version] + '/' +
                       target + '/')
    destination_str_miri = (phangs_access_config.phangs_config_dict['jwst_obs_data_local_path_%s' % miri_version] + '/' +
                       target + '/')

    if target == 'ngc1068':
        extension = ''
    else:
        extension = '_anchor'

    # loop over nircam bands
    print('jwst_obs_band_dict_%s' % nircam_version)
    for band in getattr(phangs_info, 'jwst_obs_band_dict_%s' % nircam_version)[target]['nircam_observed_bands']:
        print('NIRCam ', band)
        data_path = ('%s_nircam_lv3_%s_i2d%s.fits' %
                     (target, band.lower(), extension))

        if not os.path.isfile(destination_str_nircam + data_path):
            jwst_rclone_file.writelines(path_str_nircam + data_path + ' ' + destination_str_nircam + ' \n')

    # loop over miri bands
    for band in getattr(phangs_info, 'jwst_obs_band_dict_%s' % miri_version)[target]['miri_observed_bands']:
        print('MIRI ', band)
        data_path = ('%s_miri_lv3_%s_i2d%s.fits' %
                     (target, band.lower(), extension))

        if not os.path.isfile(destination_str_miri + data_path):
            jwst_rclone_file.writelines(path_str_miri + data_path + ' ' + destination_str_miri + ' \n')

jwst_rclone_file.close()

