"""
Script to create script to rclone PHANGS HST observational data from google drive
"""
import os.path
from pathlib import Path
from phangs_data_access import phangs_info, phangs_access_config, helper_func


target_list = phangs_info.hst_cluster_cat_target_list

rclone_name = phangs_access_config.phangs_config_dict['rclone_name']

drive_path = 'rclone copy drive:'

hst_rclone_file = open('download_scripts/rclone_hst_all_src_cat.sh', "w")




for target in target_list:

    path_str = ('rclone copy ' + phangs_access_config.phangs_config_dict['rclone_name'] + ':' +
                phangs_access_config.phangs_config_dict['hst_all_src_cat_drive_path'])

    destination_str = (phangs_access_config.phangs_config_dict['hst_all_src_cat_local_path'])
    check_destination_str = (phangs_access_config.phangs_config_dict['hst_all_src_cat_local_path'].replace('\ ', ' ') + '/' +
                       helper_func.FileTools.target_names_no_zeros(target=target) + '/')
    file_name = '%s_augmented_dolphot_candidates_photprocessed_v1p2.fits' % (helper_func.FileTools.target_names_no_zeros(target=target))

    hst_rclone_file.writelines(path_str + file_name + ' ' + destination_str + ' \n')

hst_rclone_file.close()

