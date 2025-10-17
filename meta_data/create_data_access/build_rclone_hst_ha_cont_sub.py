

from phangs_data_access.visualization_tool import PhotVisualize

photo_access_empty = PhotVisualize()


jwst_obs_targets = photo_access_empty.phangs_nircam_obs_target_list
hst_obs_targets = photo_access_empty.phangs_hst_obs_target_list


hst_ha_rclone_file = open('rclone_hst_h_alpha_data.sh', "w")


jwst_rclone_file.writelines('mkdir phangs_jwst' + ' \n')
jwst_rclone_file.writelines('mkdir phangs_jwst/v1p1p1' + ' \n')
# create jwst download script
for target in jwst_obs_targets:
    jwst_rclone_file.writelines('mkdir phangs_jwst/v1p1p1/%s' % target + ' \n')
    jwst_rclone_file.writelines('rclone copy drive:scratch/JWST_TECHNICAL_WORK/JWST_Cycle1_Treasury/v1p1/%s/'
                                ' /media/benutzer/Extreme\ Pro/data/phangs_jwst/v1p1p1/%s/ --include "*_anchor.fits"' %
                                (target, target) + ' \n')

jwst_rclone_file.close()

hst_rclone_file = open('rclone_hst_data.sh', "w")
hst_rclone_file.writelines('mkdir phangs_hst' + ' \n')

# create HST data structure
hst_rclone_file.writelines('mkdir phangs_hst/HST_reduced_images' + ' \n')

for target in hst_obs_targets:
    hst_rclone_file.writelines('mkdir phangs_hst/HST_reduced_images/%s' % target + ' \n')
    hst_rclone_file.writelines('mkdir phangs_hst/HST_reduced_images/%s' % target + ' \n')
    for acs_bands in photo_access_empty.phangs_hst_obs_band_dict[target]['acs_wfc1_observed_bands']:
        band_name = 'acs%s' % acs_bands.lower()
        hst_rclone_file.writelines('mkdir phangs_hst/HST_reduced_images/%s/%s' %
                                   (photo_access_empty.phangs_hst_obs_band_dict[target]['folder_name'], band_name)
                                   + ' \n')
        hst_rclone_file.writelines('rclone copy drive:'
                                   'scratch/HUBBLE_AND_CLUSTER_TECHNICAL_WORK/HST_image_products/HST_reduced_images/%s/%s '
                                   '/media/benutzer/Extreme\ Pro/data/phangs_hst/HST_reduced_images/%s/%s '
                                   '--include "*err_drc_sci.fits"' %
                                   (photo_access_empty.phangs_hst_obs_band_dict[target]['folder_name'], band_name,
                                    photo_access_empty.phangs_hst_obs_band_dict[target]['folder_name'], band_name) +
                                   ' \n')
        hst_rclone_file.writelines('rclone copy drive:scratch/HUBBLE_AND_CLUSTER_TECHNICAL_WORK/HST_image_products/HST_reduced_images/%s/%s '
              '/media/benutzer/Extreme\ Pro/data/phangs_hst/HST_reduced_images/%s/%s '
              '--include "*exp_drc_sci.fits"' %
              (photo_access_empty.phangs_hst_obs_band_dict[target]['folder_name'], band_name,
               photo_access_empty.phangs_hst_obs_band_dict[target]['folder_name'], band_name) + ' \n')
        hst_rclone_file.writelines('rclone copy drive:scratch/HUBBLE_AND_CLUSTER_TECHNICAL_WORK/HST_image_products/HST_reduced_images/%s/%s '
              '/media/benutzer/Extreme\ Pro/data/phangs_hst/HST_reduced_images/%s/%s '
              '--include "*exp_drc_wht.fits"' %
              (photo_access_empty.phangs_hst_obs_band_dict[target]['folder_name'], band_name,
               photo_access_empty.phangs_hst_obs_band_dict[target]['folder_name'], band_name) + ' \n')

    for uvis_bands in photo_access_empty.phangs_hst_obs_band_dict[target]['wfc3_uvis_observed_bands']:
        band_name = 'uvis%s' % uvis_bands.lower()
        hst_rclone_file.writelines('mkdir phangs_hst/HST_reduced_images/%s/%s' %
                                   (photo_access_empty.phangs_hst_obs_band_dict[target]['folder_name'], band_name) +
                                   ' \n')
        hst_rclone_file.writelines('rclone copy drive:scratch/HUBBLE_AND_CLUSTER_TECHNICAL_WORK/HST_image_products/HST_reduced_images/%s/%s '
              '/media/benutzer/Extreme\ Pro/data/phangs_hst/HST_reduced_images/%s/%s '
              '--include "*err_drc_sci.fits"' %
              (photo_access_empty.phangs_hst_obs_band_dict[target]['folder_name'], band_name,
               photo_access_empty.phangs_hst_obs_band_dict[target]['folder_name'], band_name) + ' \n')
        hst_rclone_file.writelines('rclone copy drive:scratch/HUBBLE_AND_CLUSTER_TECHNICAL_WORK/HST_image_products/HST_reduced_images/%s/%s '
              '/media/benutzer/Extreme\ Pro/data/phangs_hst/HST_reduced_images/%s/%s '
              '--include "*exp_drc_sci.fits"' %
              (photo_access_empty.phangs_hst_obs_band_dict[target]['folder_name'], band_name,
               photo_access_empty.phangs_hst_obs_band_dict[target]['folder_name'], band_name) + ' \n')
        hst_rclone_file.writelines('rclone copy drive:scratch/HUBBLE_AND_CLUSTER_TECHNICAL_WORK/HST_image_products/HST_reduced_images/%s/%s '
              '/media/benutzer/Extreme\ Pro/data/phangs_hst/HST_reduced_images/%s/%s '
              '--include "*exp_drc_wht.fits"' %
              (photo_access_empty.phangs_hst_obs_band_dict[target]['folder_name'], band_name,
               photo_access_empty.phangs_hst_obs_band_dict[target]['folder_name'], band_name) + ' \n')

hst_rclone_file.close()

