phangs_galaxy_list = ['ic1954', 'ic5332', 'ngc0628', 'ngc0685', 'ngc1087', 'ngc1097', 'ngc1300', 'ngc1317',
                      'ngc1365', 'ngc1385', 'ngc1433', 'ngc1512', 'ngc1559', 'ngc1566', 'ngc1672',
                      'ngc1792', 'ngc2775', 'ngc2835', 'ngc2903', 'ngc3351', 'ngc3621', 'ngc3627',
                      'ngc4254', 'ngc4298', 'ngc4303', 'ngc4321', 'ngc4535', 'ngc4536', 'ngc4548',
                      'ngc4569', 'ngc4571', 'ngc4654', 'ngc4689', 'ngc4826', 'ngc5068', 'ngc5248',
                      'ngc6744', 'ngc7496']


down_load_script = open("download_script_150pc.sh","w")

command_str = 'rsync -aP phangs@142.244.87.228:/home/phangs/PHANGS/Archive/ALMA/delivery_v4p0/'

for target in phangs_galaxy_list:
    file_name_tpeak = '%s_12m+7m+tp_co21_150pc_broad_tpeak.fits' % target
    file_name_tpeak_err = '%s_12m+7m+tp_co21_150pc_broad_etpeak.fits' % target
    source_str_tpeak = target + '/' + file_name_tpeak
    source_str_tpeak_err = target + '/' + file_name_tpeak_err

    destination_folder = '/home/benutzer/data/PHANGS-ALMA/delivery_v4p0/%s/' % target
    down_load_script.writelines(command_str + source_str_tpeak + ' ' + destination_folder + file_name_tpeak + '\n')
    down_load_script.writelines(command_str + source_str_tpeak_err + ' ' + destination_folder + file_name_tpeak_err + '\n')
down_load_script.close()



down_load_script = open("download_script_native.sh","w")

command_str = 'rsync -aP phangs@142.244.87.228:/home/phangs/PHANGS/Archive/ALMA/delivery_v4p0/'


for target in phangs_galaxy_list:
    file_name_tpeak = '%s_12m+7m+tp_co21_broad_tpeak.fits' % target
    file_name_tpeak_err = '%s_12m+7m+tp_co21_broad_etpeak.fits' % target
    source_str_tpeak = target + '/' + file_name_tpeak
    source_str_tpeak_err = target + '/' + file_name_tpeak_err

    destination_folder = '/home/benutzer/data/PHANGS-ALMA/delivery_v4p0/%s/' % target
    down_load_script.writelines(command_str + source_str_tpeak + ' ' + destination_folder + file_name_tpeak + '\n')
    down_load_script.writelines(command_str + source_str_tpeak_err + ' ' + destination_folder + file_name_tpeak_err + '\n')
down_load_script.close()



down_load_script = open("download_script_150pc_mom0.sh","w")

command_str = 'rsync -aP phangs@142.244.87.228:/home/phangs/PHANGS/Archive/ALMA/delivery_v4p0/'

for target in phangs_galaxy_list:
    file_name_mom0 = '%s_12m+7m+tp_co21_150pc_broad_mom0.fits' % target
    file_name_mom0_err = '%s_12m+7m+tp_co21_150pc_broad_emom0.fits' % target
    source_str_mom0 = target + '/' + file_name_mom0
    source_str_mom0_err = target + '/' + file_name_mom0_err

    destination_folder = '/home/benutzer/data/PHANGS-ALMA/delivery_v4p0/%s/' % target
    down_load_script.writelines(command_str + source_str_mom0 + ' ' + destination_folder + file_name_mom0 + '\n')
    down_load_script.writelines(command_str + source_str_mom0_err + ' ' + destination_folder + file_name_mom0_err + '\n')
down_load_script.close()

down_load_script = open("download_script_native_mom0.sh","w")

command_str = 'rsync -aP phangs@142.244.87.228:/home/phangs/PHANGS/Archive/ALMA/delivery_v4p0/'

for target in phangs_galaxy_list:
    file_name_mom0 = '%s_12m+7m+tp_co21_broad_mom0.fits' % target
    file_name_mom0_err = '%s_12m+7m+tp_co21_broad_emom0.fits' % target
    source_str_mom0 = target + '/' + file_name_mom0
    source_str_mom0_err = target + '/' + file_name_mom0_err

    destination_folder = '/home/benutzer/data/PHANGS-ALMA/delivery_v4p0/%s/' % target
    down_load_script.writelines(command_str + source_str_mom0 + ' ' + destination_folder + file_name_mom0 + '\n')
    down_load_script.writelines(command_str + source_str_mom0_err + ' ' + destination_folder + file_name_mom0_err + '\n')
down_load_script.close()

