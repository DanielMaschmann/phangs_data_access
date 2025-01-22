"""
Script to create script to rclone PHANGS HST observational data from google drive
"""
import os.path
from pathlib import Path
from phangs_data_access import phangs_info, phangs_access_config, helper_func


if phangs_access_config.phangs_config_dict['hst_obs_target_list'] == 'all':
    target_list = phangs_info.hst_obs_band_dict.keys()
else:
    target_list = phangs_access_config.phangs_config_dict['hst_obs_target_list']

rclone_name = phangs_access_config.phangs_config_dict['rclone_name']

drive_path = 'rclone copy drive:'

hst_rclone_file = open('download_scripts/rclone_hst_data.sh', "w")

for target in target_list:

    path_str = ('rclone copy ' + phangs_access_config.phangs_config_dict['rclone_name'] + ':' +
                phangs_access_config.phangs_config_dict['hst_obs_data_drive_path'] +
                helper_func.FileTools.target_names_no_zeros(target=target) + '/')

    destination_str = (phangs_access_config.phangs_config_dict['hst_obs_data_local_path'] + '/' +
                       helper_func.FileTools.target_names_no_zeros(target=target) + '/')
    check_destination_str = (phangs_access_config.phangs_config_dict['hst_obs_data_local_path'].replace('\ ', ' ') + '/' +
                       helper_func.FileTools.target_names_no_zeros(target=target) + '/')
    # loop over bands
    for band in phangs_info.hst_obs_band_dict[target]['acs']:
        print('ACS ', band)
        data_path = ('acs' + band.lower() + '/' + '%s_acs_%s_exp_drc_sci.fits' %
                     (helper_func.FileTools.target_names_no_zeros(target=target), band.lower()))
        err_path = ('acs' + band.lower() + '/' + '%s_acs_%s_err_drc_wht.fits' %
                    (helper_func.FileTools.target_names_no_zeros(target=target), band.lower()))
        wht_path = ('acs' + band.lower() + '/' + '%s_acs_%s_exp_drc_wht.fits' %
                    (helper_func.FileTools.target_names_no_zeros(target=target), band.lower()))
        if not os.path.isfile(check_destination_str + data_path):
            hst_rclone_file.writelines(path_str + data_path + ' ' + destination_str +'acs' + band.lower() + ' \n')
        if not os.path.isfile(check_destination_str + err_path):
            hst_rclone_file.writelines(path_str + err_path + ' ' + destination_str +'acs' + band.lower() + ' \n')
        if not os.path.isfile(check_destination_str + wht_path):
            hst_rclone_file.writelines(path_str + wht_path + ' ' + destination_str +'acs' + band.lower() + ' \n')

    for band in phangs_info.hst_obs_band_dict[target]['uvis']:
        print('UVIS ', band)
        data_path = ('uvis' + band.lower() + '/' + '%s_uvis_%s_exp_drc_sci.fits' %
                     (helper_func.FileTools.target_names_no_zeros(target=target), band.lower()))
        err_path = ('uvis' + band.lower() + '/' + '%s_uvis_%s_err_drc_wht.fits' %
                    (helper_func.FileTools.target_names_no_zeros(target=target), band.lower()))
        wht_path = ('uvis' + band.lower() + '/' + '%s_uvis_%s_exp_drc_wht.fits' %
                    (helper_func.FileTools.target_names_no_zeros(target=target), band.lower()))
        if not os.path.isfile(check_destination_str + data_path):
            hst_rclone_file.writelines(path_str + data_path + ' ' + destination_str + 'uvis' + band.lower() + ' \n')
        if not os.path.isfile(check_destination_str + err_path):
            hst_rclone_file.writelines(path_str + err_path + ' ' + destination_str + 'uvis' + band.lower() + ' \n')
        if not os.path.isfile(check_destination_str + wht_path):
            hst_rclone_file.writelines(path_str + wht_path + ' ' + destination_str + 'uvis' + band.lower() + ' \n')

    for band in phangs_info.hst_obs_band_dict[target]['acs_uvis']:
        print('ACS and UVIS ', band)
        data_path = (band.lower() + '/' + '%s_acsuvis_%s_exp_drc_sci.fits' %
                     (helper_func.FileTools.target_names_no_zeros(target=target), band.lower()))
        err_path = (band.lower() + '/' + '%s_acsuvis_%s_err_drc_wht.fits' %
                    (helper_func.FileTools.target_names_no_zeros(target=target), band.lower()))
        wht_path = (band.lower() + '/' + '%s_acsuvis_%s_exp_drc_wht.fits' %
                    (helper_func.FileTools.target_names_no_zeros(target=target), band.lower()))
        if not os.path.isfile(check_destination_str + data_path):
            hst_rclone_file.writelines(path_str + data_path + ' ' + destination_str + 'acs_uvis' + band.lower() + ' \n')
        if not os.path.isfile(check_destination_str + err_path):
            hst_rclone_file.writelines(path_str + err_path + ' ' + destination_str + 'acs_uvis' + band.lower() + ' \n')
        if not os.path.isfile(check_destination_str + wht_path):
            hst_rclone_file.writelines(path_str + wht_path + ' ' + destination_str + 'acs_uvis' + band.lower() + ' \n')

hst_rclone_file.close()

