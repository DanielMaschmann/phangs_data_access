"""
class to gather all information need to access PHANGS observational data products
"""

# TO DO add missing galaxies !!!!



phangs_alma_galaxy_list = [
    'ic1954', 'ic5332', 'ngc0628', 'ngc0685', 'ngc1087', 'ngc1097', 'ngc1300', 'ngc1317', 'ngc1365', 'ngc1385',
    'ngc1433', 'ngc1512', 'ngc1559', 'ngc1566', 'ngc1672', 'ngc1792', 'ngc2775', 'ngc2835', 'ngc2903',
    'ngc3351', 'ngc3621', 'ngc3627', 'ngc4254', 'ngc4298', 'ngc4303', 'ngc4321', 'ngc4535', 'ngc4536', 'ngc4548',
    'ngc4569', 'ngc4571', 'ngc4654', 'ngc4689', 'ngc4826', 'ngc5068', 'ngc5248', 'ngc6744', 'ngc7496'
]
phangs_treasury_hst_galaxy_list = [
    'ic1954', 'ic5332', 'ngc0628', 'ngc0628c', 'ngc0628e', 'ngc0685', 'ngc1087', 'ngc1097', 'ngc1300', 'ngc1317', 'ngc1365', 'ngc1385',
    'ngc1433', 'ngc1510', 'ngc1512', 'ngc1559', 'ngc1566', 'ngc1672', 'ngc1792', 'ngc2775', 'ngc2835', 'ngc2903',
    'ngc3351', 'ngc3621', 'ngc3627', 'ngc4254', 'ngc4298', 'ngc4303', 'ngc4321', 'ngc4535', 'ngc4536', 'ngc4548',
    'ngc4569', 'ngc4571', 'ngc4654', 'ngc4689', 'ngc4826', 'ngc5068', 'ngc5248', 'ngc6744', 'ngc7496'
]
phangs_add_hst_galaxy_list = ['ngc5194']
phangs_treasury_jwst_galaxy_list = [
    'ic5332', 'ngc0628', 'ngc1087', 'ngc1300', 'ngc1365', 'ngc1385', 'ngc1433', 'ngc1512', 'ngc1566', 'ngc1672',
    'ngc2835', 'ngc3351', 'ngc3627', 'ngc4254', 'ngc4303', 'ngc4321', 'ngc4535', 'ngc5068', 'ngc7496'
]
phangs_add_jwst_galaxy_list = ['ngc5194']
phangs_muse_galaxy_list = [
    'ic5332', 'ngc0628', 'ngc1087', 'ngc1300', 'ngc1365', 'ngc1385', 'ngc1433', 'ngc1512', 'ngc1566', 'ngc1672',
    'ngc2835', 'ngc3351', 'ngc3627', 'ngc4254', 'ngc4303', 'ngc4321', 'ngc4535', 'ngc5068', 'ngc7496'
]

# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# !!!!!!! TO DO: make cluster catalog list version depending !!!!!!!
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
hst_cluster_cat_target_list = ['ic1954', 'ic5332', 'ngc0628e', 'ngc0628c', 'ngc0685', 'ngc1087', 'ngc1097',
                               'ngc1300', 'ngc1317', 'ngc1365', 'ngc1385', 'ngc1433', 'ngc1510', 'ngc1512',
                               'ngc1559', 'ngc1566', 'ngc1672', 'ngc1792', 'ngc2775', 'ngc2835', 'ngc2903',
                               'ngc3351', 'ngc3621', 'ngc3627', 'ngc4254', 'ngc4298', 'ngc4303', 'ngc4321',
                               'ngc4535', 'ngc4536', 'ngc4548', 'ngc4569', 'ngc4571', 'ngc4654', 'ngc4689',
                               'ngc4826', 'ngc5068', 'ngc5248', 'ngc6744', 'ngc7496']

hst_cluster_cat_hst_ha_target_list = ['ic5332', 'ngc0628e', 'ngc0628c', 'ngc1087', 'ngc1300', 'ngc1365', 'ngc1385',
                                      'ngc1433', 'ngc1512', 'ngc1566', 'ngc1672', 'ngc3351', 'ngc3627', 'ngc4254',
                                      'ngc4303', 'ngc4321', 'ngc5068', 'ngc7496']

# specification of observed bands for each HST target here we prefer to have a mosaic as a final product
hst_obs_band_dict = {
    'ic1954':
        {'acs': [],
         'uvis': ['F275W', 'F336W', 'F438W', 'F555W', 'F814W'],
         'acs_uvis': [],
         'ir': []},
    'ic5332':
        {'acs': [],
         'uvis': ['F275W', 'F336W', 'F438W', 'F555W', 'F658N', 'F814W'],
         'acs_uvis': [],
         'ir': []},
    'ngc0628':
        {'acs': ['F435W', 'F814W'],
         'uvis': ['F275W', 'F336W'],
         'acs_uvis': ['F555W', 'F658N'],
         'ir': []},
    'ngc0628c':
        {'acs': ['F435W', 'F555W', 'F658N', 'F814W'],
         'uvis': ['F275W', 'F336W'],
         'acs_uvis': [],
         'ir': []},
    'ngc0628e':
        {'acs': ['F435W', 'F814W'],
         'uvis': ['F275W', 'F336W', 'F555W', 'F658N'],
         'acs_uvis': [],
         'ir': []},
    'ngc0685':
        {'acs': [],
         'uvis': ['F275W', 'F336W', 'F438W', 'F555W', 'F814W'],
         'acs_uvis': [],
         'ir': []},
    'ngc1087':
        {'acs': [],
         'uvis': ['F275W', 'F336W', 'F438W', 'F555W', 'F657N', 'F814W'],
         'acs_uvis': [],
         'ir': []},
    'ngc1097':
        {'acs': [],
         'uvis': ['F275W', 'F336W', 'F438W', 'F555W', 'F657N', 'F814W'],
         'acs_uvis': [],
         'ir': []},
    'ngc1300':
        {'acs': ['F435W', 'F555W', 'F658N', 'F814W'],
         'uvis': ['F275W', 'F336W'],
         'acs_uvis': [],
         'ir': []},
    'ngc1317':
        {'acs': [],
         'uvis': ['F275W', 'F336W', 'F438W', 'F555W', 'F814W'],
         'acs_uvis': [],
         'ir': []},
    'ngc1365':
        {'acs': [],
         'uvis': ['F275W', 'F336W', 'F438W', 'F555W', 'F657N', 'F814W'],
         'acs_uvis': [],
         'ir': []},
    'ngc1385':
        {'acs': [],
         'uvis': ['F275W', 'F336W', 'F438W', 'F555W', 'F657N', 'F814W'],
         'acs_uvis': [],
         'ir': []},
    'ngc1433':
        {'acs': [],
         'uvis': ['F275W', 'F336W', 'F438W', 'F555W', 'F657N', 'F814W'],
         'acs_uvis': [],
         'ir': []},
    'ngc1512':
        {'acs': [],
         'uvis': ['F275W', 'F336W', 'F438W', 'F555W', 'F658N', 'F814W'],
         'acs_uvis': [],
         'ir': []},
    'ngc1510':
        {'acs': [],
         'uvis': ['F275W', 'F336W', 'F438W', 'F555W', 'F658N', 'F814W'],
         'acs_uvis': [],
         'ir': []},
    'ngc1559':
        {'acs': [],
         'uvis': ['F275W', 'F336W', 'F438W', 'F555W', 'F814W'],
         'acs_uvis': [],
         'ir': []},
    'ngc1566':
        {'acs': [],
         'uvis': ['F275W', 'F336W', 'F438W', 'F555W', 'F658N', 'F814W'],
         'acs_uvis': [],
         'ir': []},
    'ngc1672':
        {'acs': ['F435W', 'F658N', 'F814W'],
         'uvis': ['F275W', 'F336W', 'F555W'],
         'acs_uvis': [],
         'ir': []},
    'ngc1792':
        {'acs': [],
         'uvis': ['F275W', 'F336W', 'F438W', 'F555W', 'F814W'],
         'acs_uvis': [],
         'ir': []},
    'ngc2775':
        {'acs': [],
         'uvis': ['F275W', 'F336W', 'F438W', 'F555W', 'F657N', 'F814W'],
         'acs_uvis': [],
         'ir': []},
    'ngc2835':
        {'acs': [],
         'uvis': ['F275W', 'F336W', 'F438W', 'F555W', 'F658N', 'F814W'],
         'acs_uvis': [],
         'ir': []},
    'ngc2903':
        {'acs': [],
         'uvis': ['F275W', 'F336W', 'F438W', 'F555W', 'F814W'],
         'acs_uvis': [],
         'ir': []},
    'ngc2997e':
        {'acs': [],
         'uvis': ['F275W', 'F336W', 'F438W', 'F555W', 'F657N', 'F814W'],
         'acs_uvis': [],
         'ir': []},
    'ngc3059':
        {'acs': [],
         'uvis': ['F275W', 'F336W', 'F438W', 'F555W', 'F657N', 'F814W'],
         'acs_uvis': [],
         'ir': []},
    'ngc3351':
        {'acs': [],
         'uvis': ['F275W', 'F336W', 'F438W', 'F555W', 'F657N', 'F814W'],
         'acs_uvis': [],
         'ir': []},
    'ngc3368':
        {'acs': [],
         'uvis': ['F275W', 'F336W', 'F438W', 'F555W', 'F657N', 'F814W'],
         'acs_uvis': [],
         'ir': []},
    'ngc3621':
        {'acs': ['F435W', 'F555W', 'F814W'],
         'uvis': ['F275W', 'F336W'],
         'acs_uvis': [],
         'ir': []},
    'ngc3627':
        {'acs': [],
         'uvis': ['F275W', 'F336W', 'F438W', 'F555W', 'F658N', 'F814W'],
         'acs_uvis': [],
         'ir': []},
    'ngc4254':
        {'acs': [],
         'uvis': ['F275W', 'F336W', 'F438W', 'F555W', 'F657N', 'F814W'],
         'acs_uvis': [],
         'ir': []},
    'ngc4298':
        {'acs': [],
         'uvis': ['F275W', 'F336W', 'F438W', 'F555W', 'F657N', 'F814W'],
         'acs_uvis': [],
         'ir': []},
    'ngc4303':
        {'acs': [],
         'uvis': ['F275W', 'F336W', 'F438W', 'F555W', 'F658N', 'F814W'],
         'acs_uvis': [],
         'ir': []},
    'ngc4321':
        {'acs': [],
         'uvis': ['F275W', 'F336W', 'F438W', 'F555W', 'F657N', 'F814W'],
         'acs_uvis': [],
         'ir': []},
    'ngc4424':
        {'acs': [],
         'uvis': ['F275W', 'F336W', 'F438W', 'F555W', 'F657N', 'F814W'],
         'acs_uvis': [],
         'ir': []},
    'ngc4535':
        {'acs': [],
         'uvis': ['F275W', 'F336W', 'F438W', 'F555W',
                  'F657N',
                  'F814W'],
         'acs_uvis': [],
         'ir': []},
    'ngc4536':
        {'acs': [],
         'uvis': ['F275W', 'F336W', 'F438W', 'F555W', 'F657N', 'F814W'],
         'acs_uvis': [],
         'ir': []},
    'ngc4548':
        {'acs': [],
         'uvis': ['F275W', 'F336W', 'F438W', 'F555W', 'F814W'],
         'acs_uvis': [],
         'ir': []},
    'ngc4569':
        {'acs': [],
         'uvis': ['F275W', 'F336W', 'F438W', 'F555W', 'F657N', 'F814W'],
         'acs_uvis': [],
         'ir': []},
    'ngc4571':
        {'acs': [],
         'uvis': ['F275W', 'F336W', 'F438W', 'F555W', 'F657N', 'F814W'],
         'acs_uvis': [],
         'ir': []},
    'ngc4654':
        {'acs': [],
         'uvis': ['F275W', 'F336W', 'F438W', 'F555W', 'F814W'],
         'acs_uvis': [],
         'ir': []},
    'ngc4689':
        {'acs': [],
         'uvis': ['F275W', 'F336W', 'F438W', 'F555W', 'F657N', 'F814W'],
         'acs_uvis': [],
         'ir': []},
    'ngc4694':
        {'acs': [],
         'uvis': ['F275W', 'F336W', 'F438W', 'F555W', 'F657N', 'F814W'],
         'acs_uvis': [],
         'ir': []},
    'ngc4731':
        {'acs': [],
         'uvis': ['F275W', 'F336W', 'F438W', 'F555W', 'F657N', 'F814W'],
         'acs_uvis': [],
         'ir': []},
    'ngc4826':
        {'acs': [],
         'uvis': ['F275W', 'F336W', 'F438W', 'F555W', 'F657N', 'F814W'],
         'acs_uvis': [],
         'ir': []},
    'ngc4951':
        {'acs': [],
         'uvis': ['F275W', 'F336W', 'F438W', 'F555W', 'F657N', 'F814W'],
         'acs_uvis': [],
         'ir': []},
    'ngc5068':
        {'acs': [],
         'uvis': ['F275W', 'F336W', 'F438W', 'F555W', 'F658N', 'F814W'],
         'acs_uvis': [],
         'ir': []},
    # 'ngc5194':
    #     {'acs': ['F435W', 'F555W', 'F658N', 'F814W'],
    #      'uvis': ['F275W', 'F336W'],
    #      'acs_uvis': [],
    #      'ir}: [,
    'ngc5248':
        {'acs': [],
         'uvis': ['F275W', 'F336W', 'F438W', 'F555W', 'F814W'],
         'acs_uvis': [],
         'ir': []},
    'ngc6744':
        {'acs': [],
         'uvis': ['F275W', 'F336W', 'F438W', 'F555W', 'F657N', 'F814W'],
         'acs_uvis': [],
         'ir': []},
    'ngc7496':
        {'acs': [],
         'uvis': ['F275W', 'F336W', 'F438W', 'F555W', 'F657N', 'F814W'],
         'acs_uvis': [],
         'ir': []},
    'ngc7793':
        {'acs': [],
         'uvis': ['F275W', 'F336W', 'F438W'],
         'acs_uvis': ['F555W', 'F814W'],
         'ir': []},
        }

hst_ha_cont_sub_dict = {
    'ic5332': {'uvis': 'F658N', 'acs': None, 'cont_sub': True},
    'ngc0628c': {'uvis': None, 'acs': 'F658N', 'cont_sub': True},
    'ngc0628e': {'uvis': 'F658N', 'acs': None, 'cont_sub': True},
    'ngc1087': {'uvis': 'F657N', 'acs': None, 'cont_sub': True},
    #'ngc1097': {'uvis': None, 'acs': None, 'cont_sub': True},
    'ngc1300': {'uvis': None, 'acs': 'F658N', 'cont_sub': True},
    'ngc1365': {'uvis': 'F657N', 'acs': None, 'cont_sub': True},
    'ngc1385': {'uvis': 'F657N', 'acs': None, 'cont_sub': True},
    'ngc1433': {'uvis': 'F657N', 'acs': None, 'cont_sub': True},
    'ngc1512': {'uvis': 'F658N', 'acs': None, 'cont_sub': True},
    'ngc1566': {'uvis': 'F658N', 'acs': None, 'cont_sub': True},
    'ngc1672': {'uvis': None, 'acs': 'F658N', 'cont_sub': True},
    #'ngc2775': {'uvis': None, 'acs': None, 'cont_sub': True},
    'ngc2835': {'uvis': 'F658N', 'acs': None, 'cont_sub': True},
    'ngc3351': {'uvis': 'F658N', 'acs': None, 'cont_sub': True},
    'ngc3627': {'uvis': 'F658N', 'acs': None, 'cont_sub': True},
    'ngc4254e': {'uvis': 'F657N', 'acs': None, 'cont_sub': True},
    'ngc4254w': {'uvis': 'F657N', 'acs': None, 'cont_sub': True},
    'ngc4254': {'uvis': 'F657N', 'acs': None, 'cont_sub': True},
    'ngc4303': {'uvis': 'F658N', 'acs': None, 'cont_sub': True},
    'ngc4321': {'uvis': 'F657N', 'acs': None, 'cont_sub': True},
    'ngc5068n': {'uvis': 'F658N', 'acs': None, 'cont_sub': True},
    'ngc5068s': {'uvis': 'F658N', 'acs': None, 'cont_sub': True},
    'ngc5068': {'uvis': 'F658N', 'acs': None, 'cont_sub': True},
    'ngc7496': {'uvis': 'F657N', 'acs': None, 'cont_sub': True},
}

jwst_obs_band_dict = {
    'ic5332': {'nircam_observed_bands': ['F200W', 'F300M', 'F335M', 'F360M'],
               'miri_observed_bands': ['F770W', 'F1000W', 'F1130W', 'F2100W']},
    'ngc0628': {'nircam_observed_bands': ['F200W', 'F300M', 'F335M', 'F360M'],
                'miri_observed_bands': ['F770W', 'F1000W', 'F1130W', 'F2100W']},
    'ngc1087': {'nircam_observed_bands': ['F200W', 'F300M', 'F335M', 'F360M'],
                'miri_observed_bands': ['F770W', 'F1000W', 'F1130W', 'F2100W']},
    'ngc1300': {'nircam_observed_bands': ['F200W', 'F300M', 'F335M', 'F360M'],
                'miri_observed_bands': ['F770W', 'F1000W', 'F1130W', 'F2100W']},
    'ngc1365': {'nircam_observed_bands': ['F200W', 'F300M', 'F335M', 'F360M'],
                'miri_observed_bands': ['F770W', 'F1000W', 'F1130W', 'F2100W']},
    'ngc1385': {'nircam_observed_bands': ['F200W', 'F300M', 'F335M', 'F360M'],
                'miri_observed_bands': ['F770W', 'F1000W', 'F1130W', 'F2100W']},
    'ngc1433': {'nircam_observed_bands': ['F200W', 'F300M', 'F335M', 'F360M'],
                'miri_observed_bands': ['F770W', 'F1000W', 'F1130W', 'F2100W']},
    'ngc1512': {'nircam_observed_bands': ['F200W', 'F300M', 'F335M', 'F360M'],
                'miri_observed_bands': ['F770W', 'F1000W', 'F1130W', 'F2100W']},
    'ngc1566': {'nircam_observed_bands': ['F200W', 'F300M', 'F335M', 'F360M'],
                'miri_observed_bands': ['F770W', 'F1000W', 'F1130W', 'F2100W']},
    'ngc1672': {'nircam_observed_bands': ['F200W', 'F300M', 'F335M', 'F360M'],
                'miri_observed_bands': ['F770W', 'F1000W', 'F1130W', 'F2100W']},
    'ngc2835': {'nircam_observed_bands': ['F200W', 'F300M', 'F335M', 'F360M'],
                'miri_observed_bands': ['F770W', 'F1000W', 'F1130W', 'F2100W']},
    'ngc3351': {'nircam_observed_bands': ['F200W', 'F300M', 'F335M', 'F360M'],
                'miri_observed_bands': ['F770W', 'F1000W', 'F1130W', 'F2100W']},
    'ngc3627': {'nircam_observed_bands': ['F200W', 'F300M', 'F335M', 'F360M'],
                'miri_observed_bands': ['F770W', 'F1000W', 'F1130W', 'F2100W']},
    'ngc4254': {'nircam_observed_bands': ['F200W', 'F300M', 'F335M', 'F360M'],
                'miri_observed_bands': ['F770W', 'F1000W', 'F1130W', 'F2100W']},
    'ngc4303': {'nircam_observed_bands': ['F200W', 'F300M', 'F335M', 'F360M'],
                'miri_observed_bands': ['F770W', 'F1000W', 'F1130W', 'F2100W']},
    'ngc4321': {'nircam_observed_bands': ['F200W', 'F300M', 'F335M', 'F360M'],
                'miri_observed_bands': ['F770W', 'F1000W', 'F1130W', 'F2100W']},
    'ngc4535': {'nircam_observed_bands': ['F200W', 'F300M', 'F335M', 'F360M'],
                'miri_observed_bands': ['F770W', 'F1000W', 'F1130W', 'F2100W']},
    'ngc5068': {'nircam_observed_bands': ['F200W', 'F300M', 'F335M', 'F360M'],
                'miri_observed_bands': ['F770W', 'F1000W', 'F1130W', 'F2100W']},
    'ngc7496': {'nircam_observed_bands': ['F200W', 'F300M', 'F335M', 'F360M'],
                'miri_observed_bands': ['F770W', 'F1000W', 'F1130W', 'F2100W']},
}

astrosat_obs_band_dict = {
    'ic5332': {'observed_bands': ['F148W']},
    'ngc0253': {'observed_bands': ['F169M', 'N219M', 'N263M']},
    'ngc0300': {'observed_bands': ['F148W', 'F154W', 'F169M', 'F172M', 'N219M', 'N245M', 'N263M']},
    'ngc0628': {'observed_bands': ['F148W', 'F154W', 'F169M', 'F172M', 'N242W', 'N219M', 'N245M', 'N263M', 'N279N']},
    'ngc1097': {'observed_bands': ['F148W']},
    'ngc1300': {'observed_bands': ['F148W']},
    'ngc1317': {'observed_bands': ['F148W', 'F154W', 'F169M', 'F172M', 'N219M']},
    'ngc1365': {'observed_bands': ['F148W', 'F169M', 'F172M', 'N219M', 'N263M', 'N279N']},
    'ngc1385': {'observed_bands': ['F148W']},
    'ngc1433': {'observed_bands': ['F154W', 'F169M', 'N219M', 'N245M', 'N263M', 'N279N']},
    'ngc1512': {'observed_bands': ['F154W', 'N242W', 'N245M', 'N263M']},
    'ngc1546': {'observed_bands': ['F148W']},
    'ngc1566': {'observed_bands': ['F148W', 'F154W', 'F172M', 'N219M', 'N263M']},
    'ngc2090': {'observed_bands': ['F148W']},
    'ngc2835': {'observed_bands': ['F148W']},
    'ngc2903': {'observed_bands': ['F148W', 'F169M', 'N219M', 'N263M']},
    'ngc3351': {'observed_bands': ['F148W']},
    'ngc3621': {'observed_bands': ['F148W', 'F172M']},
    'ngc3627': {'observed_bands': ['F148W']},
    'ngc4254': {'observed_bands': ['F154W']},
    'ngc4298': {'observed_bands': ['F148W']},
    'ngc4321': {'observed_bands': ['F154W']},
    'ngc4476': {'observed_bands': ['F154W', 'N242W']},
    'ngc4535': {'observed_bands': ['F148W']},
    'ngc4571': {'observed_bands': ['F154W', 'N263M']},
    'ngc4579': {'observed_bands': ['F154W']},
    'ngc4654': {'observed_bands': ['F148W']},
    'ngc5128': {'observed_bands': ['F148W', 'N219M', 'N245M', 'N279N']},
    'ngc6744': {'observed_bands': ['F148W']},
    'ngc7496': {'observed_bands': ['F148W']},
    'ngc7793': {'observed_bands': ['F148W', 'N242W']}
}

muse_obs_res_dict = {
    'ic5332': {'copt_res': 0.87},
    'ngc0628': {'copt_res': 0.92},
    'ngc1087': {'copt_res': 0.92},
    'ngc1300': {'copt_res': 0.89},
    'ngc1365': {'copt_res': 1.15},
    'ngc1385': {'copt_res': 0.77},
    'ngc1433': {'copt_res': 0.91},
    'ngc1512': {'copt_res': 1.25},
    'ngc1566': {'copt_res': 0.80},
    'ngc1672': {'copt_res': 0.96},
    'ngc2835': {'copt_res': 1.15},
    'ngc3351': {'copt_res': 1.05},
    'ngc3627': {'copt_res': 1.05},
    'ngc4254': {'copt_res': 0.89},
    'ngc4303': {'copt_res': 0.78},
    'ngc4321': {'copt_res': 1.16},
    'ngc4535': {'copt_res': 0.56},
    'ngc5068': {'copt_res': 1.04},
    'ngc7496': {'copt_res': 0.89}
}

# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# !!!!!!! TO DO: make cluster catalog list version depending !!!!!!!
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
hst_cluster_cat_obs_band_dict = {
    'ic1954':
        {'acs': [],
         'uvis': ['F275W', 'F336W', 'F438W', 'F555W', 'F814W']},
    'ic5332':
        {'acs': [],
         'uvis': ['F275W', 'F336W', 'F438W', 'F555W', 'F814W']},
    'ngc0628e':
        {'acs': ['F435W', 'F814W'],
         'uvis': ['F275W', 'F336W', 'F555W']},
    'ngc0628c':
        {'acs': ['F435W', 'F555W', 'F658N', 'F814W'],
         'uvis': ['F275W', 'F336W']},
    'ngc0685':
        {'acs': [],
         'uvis': ['F275W', 'F336W', 'F438W', 'F555W', 'F814W']},
    'ngc1087':
        {'acs': [],
         'uvis': ['F275W', 'F336W', 'F438W', 'F555W', 'F814W']},
    'ngc1097':
        {'acs': [],
         'uvis': ['F275W', 'F336W', 'F438W', 'F555W', 'F814W']},
    'ngc1300':
        {'acs': ['F435W', 'F555W', 'F814W'],
         'uvis': ['F275W', 'F336W']},
    'ngc1317':
        {'acs': [],
         'uvis': ['F275W', 'F336W', 'F438W', 'F555W', 'F814W']},
    'ngc1365':
        {'acs': [],
         'uvis': ['F275W', 'F336W', 'F438W', 'F555W', 'F814W']},
    'ngc1385':
        {'acs': [],
         'uvis': ['F275W', 'F336W', 'F438W', 'F555W', 'F814W']},
    'ngc1433':
        {'acs': [],
         'uvis': ['F275W', 'F336W', 'F438W', 'F555W', 'F814W']},
    'ngc1510':
        {'acs': [],
         'uvis': ['F275W', 'F336W', 'F438W', 'F555W', 'F814W']},
    'ngc1512':
        {'acs': [],
         'uvis': ['F275W', 'F336W', 'F438W', 'F555W', 'F814W']},
    'ngc1559':
        {'acs': [],
         'uvis': ['F275W', 'F336W', 'F438W', 'F555W', 'F814W']},
    'ngc1566':
        {'acs': [],
         'uvis': ['F275W', 'F336W', 'F438W', 'F555W', 'F814W']},
    'ngc1672':
        {'acs': ['F435W', 'F814W'],
         'uvis': ['F275W', 'F336W', 'F555W']},
    'ngc1792':
        {'acs': [],
         'uvis': ['F275W', 'F336W', 'F438W', 'F555W', 'F814W']},
    'ngc2775':
        {'acs': [],
         'uvis': ['F275W', 'F336W', 'F438W', 'F555W', 'F814W']},
    'ngc2835':
        {'acs': [],
         'uvis': ['F275W', 'F336W', 'F438W', 'F555W', 'F814W']},
    'ngc2903':
        {'acs': [],
         'uvis': ['F275W', 'F336W', 'F438W', 'F555W', 'F814W']},
    'ngc3351':
        {'acs': [],
         'uvis': ['F275W', 'F336W', 'F438W', 'F555W', 'F814W']},
    'ngc3621':
        {'acs': ['F435W', 'F555W', 'F814W'],
         'uvis': ['F275W', 'F336W']},
    'ngc3627':
        {'acs': [],
         'uvis': ['F275W', 'F336W', 'F438W', 'F555W', 'F814W']},
    'ngc4254':
        {'acs': [],
         'uvis': ['F275W', 'F336W', 'F438W', 'F555W', 'F814W']},
    'ngc4298':
        {'acs': [],
         'uvis': ['F275W', 'F336W', 'F438W', 'F555W', 'F814W']},
    'ngc4303':
        {'acs': [],
         'uvis': ['F275W', 'F336W', 'F438W', 'F555W', 'F814W']},
    'ngc4321':
        {'acs': [],
         'uvis': ['F275W', 'F336W', 'F438W', 'F555W', 'F814W']},
    'ngc4535':
        {'acs': [],
         'uvis': ['F275W', 'F336W', 'F438W', 'F555W', 'F814W']},
    'ngc4536':
        {'acs': [],
         'uvis': ['F275W', 'F336W', 'F438W', 'F555W', 'F814W']},
    'ngc4548':
        {'acs': [],
         'uvis': ['F275W', 'F336W', 'F438W', 'F555W', 'F814W']},
    'ngc4569':
        {'acs': [],
         'uvis': ['F275W', 'F336W', 'F438W', 'F555W', 'F814W']},
    'ngc4571':
        {'acs': [],
         'uvis': ['F275W', 'F336W', 'F438W', 'F555W', 'F814W']},
    'ngc4654':
        {'acs': [],
         'uvis': ['F275W', 'F336W', 'F438W', 'F555W', 'F814W']},
    'ngc4689':
        {'acs': [],
         'uvis': ['F275W', 'F336W', 'F438W', 'F555W', 'F814W']},
    'ngc4826':
        {'acs': [],
         'uvis': ['F275W', 'F336W', 'F438W', 'F555W', 'F814W']},
    'ngc5068':
        {'acs': [],
         'uvis': ['F275W', 'F336W', 'F438W', 'F555W', 'F814W']},
    'ngc5248':
        {'acs': [],
         'uvis': ['F275W', 'F336W', 'F438W', 'F555W', 'F814W']},
    'ngc6744':
        {'acs': [],
         'uvis': ['F275W', 'F336W', 'F438W', 'F555W', 'F814W']},
    'ngc7496':
        {'acs': [],
         'uvis': ['F275W', 'F336W', 'F438W', 'F555W', 'F814W']},
        }

############################
##### PSF availability #####
############################
acs_wfc_psf_band_list = ['F435W', 'F475W', 'F606W', 'F625W', 'F658N', 'F775W', 'F814W', 'F850L']
wfc3_uv_psf_band_list = ['F225W', 'F275W', 'F336W', 'F390W', 'F438W', 'F467M', 'F555W', 'F606W', 'F621M', 'F775W', 'F814W',
                    'F850L']
wfc3_ir_psf_band_list = ['F098M', 'F105W', 'F110W', 'F125W', 'F127M', 'F140W', 'F153M', 'F160W']








class OLDPhangsObsInfo:
    """
    all info needed for PHANGS observations are gathered in dictionaries as attributes
    """

    def __init__(self):
        super().__init__()

        # List of all Phangs galaxies.
        # This list is biased towards photometry observations and some ALMA observed galaxies are not incuded
        self.phangs_galaxy_list = ['ic1954', 'ic5332', 'ngc0628', 'ngc0685', 'ngc1087', 'ngc1097', 'ngc1300', 'ngc1317',
                                   'ngc1365', 'ngc1385', 'ngc1433', 'ngc1512', 'ngc1559', 'ngc1566', 'ngc1672',
                                   'ngc1792', 'ngc2775', 'ngc2835', 'ngc2903', 'ngc3351', 'ngc3621', 'ngc3627',
                                   'ngc4254', 'ngc4298', 'ngc4303', 'ngc4321', 'ngc4535', 'ngc4536', 'ngc4548',
                                   'ngc4569', 'ngc4571', 'ngc4654', 'ngc4689', 'ngc4826', 'ngc5068', 'ngc5248',
                                   'ngc6744', 'ngc7496']

        # Phangs target with existing HST observations
        self.phangs_hst_obs_target_list = ['ic1954', 'ic5332', 'ngc0628', 'ngc0685', 'ngc1087',
                                           'ngc1097', 'ngc1300', 'ngc1317', 'ngc1365', 'ngc1385', 'ngc1433', 'ngc1512',
                                           'ngc1559', 'ngc1566', 'ngc1672', 'ngc1792', 'ngc2775', 'ngc2835', 'ngc2903',
                                           'ngc3351', 'ngc3621', 'ngc3627', 'ngc4254', 'ngc4298', 'ngc4303', 'ngc4321',
                                           'ngc4535', 'ngc4536', 'ngc4548', 'ngc4569', 'ngc4571', 'ngc4654', 'ngc4689',
                                           'ngc4826', 'ngc5068', 'ngc5248', 'ngc6744', 'ngc7496']

        # Phangs target with existing HST H-alpha observations
        self.phangs_hst_ha_obs_target_list = ['ic5332', 'ngc0628', 'ngc0628e', 'ngc628c', 'ngc1087', 'ngc1300',
                                              'ngc1365n', 'ngc1365', 'ngc1385', 'ngc1433', 'ngc1512', 'ngc1566',
                                              'ngc1672', 'ngc2835', 'ngc3351', 'ngc3627', 'ngc4254', 'ngc4303',
                                              'ngc4321', 'ngc5068n', 'ngc5068s', 'ngc5068', 'ngc7496']

        # Phangs target with existing NIRCAM observations
        self.phangs_nircam_obs_target_list = ['ic5332', 'ngc0628', 'ngc1087', 'ngc1300', 'ngc1365', 'ngc1385', 'ngc1433',
                                              'ngc1512', 'ngc1566', 'ngc1672', 'ngc2835', 'ngc3351', 'ngc3627', 'ngc4254',
                                              'ngc4303', 'ngc4321', 'ngc4535', 'ngc5068', 'ngc7496']
        # Phangs target with existing MIRI observations
        self.phangs_miri_obs_target_list = ['ic5332', 'ngc0628', 'ngc1087', 'ngc1300', 'ngc1365', 'ngc1385', 'ngc1433',
                                            'ngc1512', 'ngc1566', 'ngc1672', 'ngc2835', 'ngc3351', 'ngc3627', 'ngc4254',
                                            'ngc4303', 'ngc4321', 'ngc4535', 'ngc5068', 'ngc7496']
        # Phangs target with existing AstroSat observations
        self.phangs_astrosat_obs_target_list = ['ic5332', 'ngc0253', 'ngc0300', 'ngc0628', 'ngc1097', 'ngc1300',
                                                'ngc1317', 'ngc1365', 'ngc1385', 'ngc1433', 'ngc1512', 'ngc1546',
                                                'ngc1566', 'ngc2090', 'ngc2835', 'ngc2903', 'ngc3351', 'ngc3621',
                                                'ngc3627', 'ngc4254', 'ngc4298', 'ngc4321', 'ngc4476', 'ngc4535',
                                                'ngc4571', 'ngc4579', 'ngc4654', 'ngc5128', 'ngc6744', 'ngc7496',
                                                'ngc7793']

        # targets for which a cluster catalog was created
        self.phangs_hst_cluster_cat_target_list = ['ic1954', 'ic5332', 'ngc0628e', 'ngc0628c', 'ngc0685', 'ngc1087',
                                                   'ngc1097', 'ngc1300', 'ngc1317', 'ngc1365', 'ngc1385', 'ngc1433',
                                                   'ngc1512', 'ngc1559', 'ngc1566', 'ngc1672', 'ngc1792', 'ngc2775',
                                                   'ngc2835', 'ngc2903', 'ngc3351', 'ngc3621', 'ngc3627', 'ngc4254',
                                                   'ngc4298', 'ngc4303', 'ngc4321', 'ngc4535', 'ngc4536', 'ngc4548',
                                                   'ngc4569', 'ngc4571', 'ngc4654', 'ngc4689', 'ngc4826', 'ngc5068',
                                                   'ngc5248', 'ngc6744', 'ngc7496']

        self.hst_ver_folder_names = {'v1.0': 'v1.0'}
        self.hst_ha_ver_folder_names = {'v1p0': 'phangs-hst-ha_v1p0'}
        self.nircam_ver_folder_names = {'v0p9': 'v0p9', 'v0p9p1': 'v0p9p1', 'v0p9p2': 'v0p9p2', 'v1p1p1': 'v1p1p1'}
        self.miri_ver_folder_names = {'v0p9': 'v0p9', 'v0p9p1': 'v0p9p1', 'v0p9p2': 'v0p9p2', 'v1p1p1': 'v1p1p1'}
        self.astrosat_ver_folder_names = {'v1p0': 'v1p0'}



        # specification of HST observed galaxies
        self.phangs_hst_ha_obs_band_dict = {
            'ic5332': {'ha_observed': ['ha', 'ha_s', 'ha_si', 'ha_sic', 'F658N']},
            'ngc0628': {'ha_observed': ['ha', 'ha_s', 'ha_si', 'ha_sic', 'F658N']},
            'ngc0628e': {'ha_observed': ['ha', 'ha_s', 'ha_si', 'ha_sic', 'F658N']},
            'ngc628c': {'ha_observed': ['ha', 'ha_s', 'ha_si', 'ha_sic', 'F658N']},
            'ngc1087': {'ha_observed': ['ha', 'ha_s', 'ha_si', 'ha_sic', 'F657N']},
            'ngc1300': {'ha_observed': ['ha', 'ha_s', 'ha_si', 'ha_sic', 'F658N']},
            'ngc1365': {'ha_observed': ['ha', 'ha_s', 'ha_si', 'ha_sic', 'F657N']},
            'ngc1365n': {'ha_observed': ['ha', 'ha_s', 'ha_si', 'ha_sic', 'F657N']},
            'ngc1385': {'ha_observed': ['ha', 'ha_s', 'ha_si', 'ha_sic', 'F657N']},
            'ngc1433': {'ha_observed': ['ha', 'ha_s', 'ha_si', 'ha_sic', 'F657N']},
            'ngc1512': {'ha_observed': ['ha', 'ha_s', 'ha_si', 'ha_sic', 'F658N']},
            'ngc1566': {'ha_observed': ['ha', 'ha_s', 'ha_si', 'ha_sic', 'F658N']},
            'ngc1672': {'ha_observed': ['ha', 'ha_s', 'ha_si', 'ha_sic', 'F658N']},
            'ngc2835': {'ha_observed': ['ha', 'ha_s', 'ha_si', 'ha_sic', 'F658N']},
            'ngc3351': {'ha_observed': ['ha', 'ha_s', 'ha_si', 'ha_sic', 'F658N']},
            'ngc3627': {'ha_observed': ['ha', 'ha_s', 'ha_si', 'ha_sic', 'F658N']},
            'ngc4254': {'ha_observed': ['ha', 'ha_s', 'ha_si', 'ha_sic', 'F657N']},
            'ngc4303': {'ha_observed': ['ha', 'ha_s', 'ha_si', 'ha_sic', 'F658N']},
            'ngc4321': {'ha_observed': ['ha', 'ha_s', 'ha_si', 'ha_sic', 'F657N']},
            'ngc5068n': {'ha_observed': ['ha', 'ha_s', 'ha_si', 'ha_sic', 'F658N']},
            'ngc5068s': {'ha_observed': ['ha', 'ha_s', 'ha_si', 'ha_sic', 'F658N']},
            'ngc5068': {'ha_observed': ['ha', 'ha_s', 'ha_si', 'ha_sic', 'F658N']},
            'ngc7496': {'ha_observed': ['ha', 'ha_s', 'ha_si', 'ha_sic', 'F657N']},
        }
        # specification of observed bands for each NIRCAM target
        self.nircam_targets = {
            'ic5332': {'folder_name': 'ic5332', 'observed_bands': ['F200W', 'F300M', 'F335M', 'F360M']},
            'ngc0628': {'folder_name': 'ngc0628', 'observed_bands': ['F200W', 'F300M', 'F335M', 'F360M']},
            'ngc1087': {'folder_name': 'ngc1087', 'observed_bands': ['F200W', 'F300M', 'F335M', 'F360M']},
            'ngc1300': {'folder_name': 'ngc1300', 'observed_bands': ['F200W', 'F300M', 'F335M', 'F360M']},
            'ngc1365': {'folder_name': 'ngc1365', 'observed_bands': ['F200W', 'F300M', 'F335M', 'F360M']},
            'ngc1385': {'folder_name': 'ngc1385', 'observed_bands': ['F200W', 'F300M', 'F335M', 'F360M']},
            'ngc1433': {'folder_name': 'ngc1433', 'observed_bands': ['F200W', 'F300M', 'F335M', 'F360M']},
            'ngc1512': {'folder_name': 'ngc1512', 'observed_bands': ['F200W', 'F300M', 'F335M', 'F360M']},
            'ngc1566': {'folder_name': 'ngc1566', 'observed_bands': ['F200W', 'F300M', 'F335M', 'F360M']},
            'ngc1672': {'folder_name': 'ngc1672', 'observed_bands': ['F200W', 'F300M', 'F335M', 'F360M']},
            'ngc2835': {'folder_name': 'ngc2835', 'observed_bands': ['F200W', 'F300M', 'F335M', 'F360M']},
            'ngc3351': {'folder_name': 'ngc3351', 'observed_bands': ['F200W', 'F300M', 'F335M', 'F360M']},
            'ngc3627': {'folder_name': 'ngc3627', 'observed_bands': ['F200W', 'F300M', 'F335M', 'F360M']},
            'ngc4254': {'folder_name': 'ngc4254', 'observed_bands': ['F200W', 'F300M', 'F335M', 'F360M']},
            'ngc4303': {'folder_name': 'ngc4303', 'observed_bands': ['F200W', 'F300M', 'F335M', 'F360M']},
            'ngc4321': {'folder_name': 'ngc4321', 'observed_bands': ['F200W', 'F300M', 'F335M', 'F360M']},
            'ngc4535': {'folder_name': 'ngc4535', 'observed_bands': ['F200W', 'F300M', 'F335M', 'F360M']},
            'ngc5068': {'folder_name': 'ngc5068', 'observed_bands': ['F200W', 'F300M', 'F335M', 'F360M']},
            'ngc7496': {'folder_name': 'ngc7496', 'observed_bands': ['F200W', 'F300M', 'F335M', 'F360M']},
        }
        # specification of observed bands for each MIRI target
        self.miri_targets = {
            'ic5332': {'folder_name': 'ic5332', 'observed_bands': ['F770W', 'F1000W', 'F1130W', 'F2100W']},
            'ngc0628': {'folder_name': 'ngc0628', 'observed_bands': ['F770W', 'F1000W', 'F1130W', 'F2100W']},
            'ngc1087': {'folder_name': 'ngc1087', 'observed_bands': ['F770W', 'F1000W', 'F1130W', 'F2100W']},
            'ngc1300': {'folder_name': 'ngc1300', 'observed_bands': ['F770W', 'F1000W', 'F1130W', 'F2100W']},
            'ngc1365': {'folder_name': 'ngc1365', 'observed_bands': ['F770W', 'F1000W', 'F1130W', 'F2100W']},
            'ngc1385': {'folder_name': 'ngc1385', 'observed_bands': ['F770W', 'F1000W', 'F1130W', 'F2100W']},
            'ngc1433': {'folder_name': 'ngc1433', 'observed_bands': ['F770W', 'F1000W', 'F1130W', 'F2100W']},
            'ngc1512': {'folder_name': 'ngc1512', 'observed_bands': ['F770W', 'F1000W', 'F1130W', 'F2100W']},
            'ngc1566': {'folder_name': 'ngc1566', 'observed_bands': ['F770W', 'F1000W', 'F1130W', 'F2100W']},
            'ngc1672': {'folder_name': 'ngc1672', 'observed_bands': ['F770W', 'F1000W', 'F1130W', 'F2100W']},
            'ngc2835': {'folder_name': 'ngc2835', 'observed_bands': ['F770W', 'F1000W', 'F1130W', 'F2100W']},
            'ngc3351': {'folder_name': 'ngc3351', 'observed_bands': ['F770W', 'F1000W', 'F1130W', 'F2100W']},
            'ngc3627': {'folder_name': 'ngc3627', 'observed_bands': ['F770W', 'F1000W', 'F1130W', 'F2100W']},
            'ngc4254': {'folder_name': 'ngc4254', 'observed_bands': ['F770W', 'F1000W', 'F1130W', 'F2100W']},
            'ngc4303': {'folder_name': 'ngc4303', 'observed_bands': ['F770W', 'F1000W', 'F1130W', 'F2100W']},
            'ngc4321': {'folder_name': 'ngc4321', 'observed_bands': ['F770W', 'F1000W', 'F1130W', 'F2100W']},
            'ngc4535': {'folder_name': 'ngc4535', 'observed_bands': ['F770W', 'F1000W', 'F1130W', 'F2100W']},
            'ngc5068': {'folder_name': 'ngc5068', 'observed_bands': ['F770W', 'F1000W', 'F1130W', 'F2100W']},
            'ngc7496': {'folder_name': 'ngc7496', 'observed_bands': ['F770W', 'F1000W', 'F1130W', 'F2100W']},
        }

        self.astrosat_targets = {
            'ic5332': {'observed_bands': ['F148W']},
            'ngc0253': {'observed_bands': ['F169M', 'N219M', 'N263M']},
            'ngc0300': {'observed_bands': ['F148W', 'F154W', 'F169M', 'F172M', 'N219M', 'N245M', 'N263M']},
            'ngc0628': {'observed_bands': ['F148W', 'F154W', 'F169M', 'F172M', 'N242W', 'N219M', 'N245M', 'N263M',
                                           'N279N']},
            'ngc1097': {'observed_bands': ['F148W']},
            'ngc1300': {'observed_bands': ['F148W']},
            'ngc1317': {'observed_bands': ['F148W', 'F154W', 'F169M', 'F172M', 'N219M']},
            'ngc1365': {'observed_bands': ['F148W', 'F169M', 'F172M', 'N219M', 'N263M', 'N279N']},
            'ngc1385': {'observed_bands': ['F148W']},
            'ngc1433': {'observed_bands': ['F154W', 'F169M', 'N219M', 'N245M', 'N263M', 'N279N']},
            'ngc1512': {'observed_bands': ['F154W', 'N242W', 'N245M', 'N263M']},
            'ngc1546': {'observed_bands': ['F148W']},
            'ngc1566': {'observed_bands': ['F148W', 'F154W', 'F172M', 'N219M', 'N263M']},
            'ngc2090': {'observed_bands': ['F148W']},
            'ngc2835': {'observed_bands': ['F148W']},
            'ngc2903': {'observed_bands': ['F148W', 'F169M', 'N219M', 'N263M']},
            'ngc3351': {'observed_bands': ['F148W']},
            'ngc3621': {'observed_bands': ['F148W', 'F172M']},
            'ngc3627': {'observed_bands': ['F148W']},
            'ngc4254': {'observed_bands': ['F148W']},
            'ngc4298': {'observed_bands': ['F148W']},
            'ngc4321': {'observed_bands': ['F154W']},
            'ngc4476': {'observed_bands': ['F154W', 'N242W']},
            'ngc4535': {'observed_bands': ['F148W']},
            'ngc4571': {'observed_bands': ['F154W', 'N263M']},
            'ngc4579': {'observed_bands': ['F154W']},
            'ngc4654': {'observed_bands': ['F148W', 'F154W']},
            'ngc5128': {'observed_bands': ['F148W', 'N219M', 'N245M', 'N279N']},
            'ngc6744': {'observed_bands': ['F148W']},
            'ngc7496': {'observed_bands': ['F148W']},
            'ngc7793': {'observed_bands': ['F148W', 'N242W']}
        }


class PhysParams:
    """
    Class to gather all physical params
    """

    def __init__(self):
        super().__init__()

        """
        distances need to be done!!!!! See Lee et al 2022 Table 1
        """

        self.sr_per_square_deg = 0.00030461741978671  # steradians per square degree

        # zero point NIRCAM flux corrections for data from the pipeline version v0p4p2
        self.nircam_zero_point_flux_corr = {'F200W': 0.854, 'F300M': 0.997, 'F335M': 1.000, 'F360M': 1.009}

        # filter names from http://svo2.cab.inta-csic.es
        self.hst_acs_wfc1_bands = ['FR388N', 'FR423N', 'F435W', 'FR459M', 'FR462N', 'F475W', 'F502N', 'FR505N', 'F555W',
                                   'FR551N', 'F550M', 'FR601N', 'F606W', 'F625W', 'FR647M', 'FR656N', 'F658N', 'F660N',
                                   'FR716N', 'POL_UV', 'POL_V', 'G800L', 'F775W', 'FR782N', 'F814W', 'FR853N', 'F892N',
                                   'FR914M', 'F850LP', 'FR931N', 'FR1016N']
        self.hst_wfc3_uvis2_bands = ['F218W', 'FQ232N', 'F225W', 'FQ243N', 'F275W', 'F280N', 'F300X', 'F336W', 'F343N',
                                     'F373N', 'FQ378N', 'FQ387N', 'F390M', 'F390W', 'F395N', 'F410M', 'FQ422M', 'F438W',
                                     'FQ436N', 'FQ437N', 'G280', 'F467M', 'F469N', 'F475W', 'F487N', 'FQ492N', 'F502N',
                                     'F475X', 'FQ508N', 'F555W', 'F547M', 'FQ575N', 'F606W', 'F200LP', 'FQ619N',
                                     'F621M', 'F625W', 'F631N', 'FQ634N', 'F645N', 'F350LP', 'F656N', 'F657N', 'F658N',
                                     'F665N', 'FQ672N', 'FQ674N', 'F673N', 'F680N', 'F689M', 'FQ727N', 'FQ750N',
                                     'F763M', 'F600LP', 'F775W', 'F814W', 'F845M', 'FQ889N', 'FQ906N', 'F850LP',
                                     'FQ924N', 'FQ937N', 'F953N']
        self.nircam_bands = ['F070W', 'F090W', 'F115W', 'F140M', 'F150W', 'F162M', 'F164N', 'F150W2', 'F182M', 'F187N',
                             'F200W', 'F210M', 'F212N', 'F250M', 'F277W', 'F300M', 'F323N', 'F322W2', 'F335M', 'F356W',
                             'F360M', 'F405N', 'F410M', 'F430M', 'F444W', 'F460M', 'F466N', 'F470N', 'F480M']
        self.miri_bands = ['F560W', 'F770W', 'F1000W', 'F1065C', 'F1140C', 'F1130W', 'F1280W', 'F1500W', 'F1550C',
                           'F1800W', 'F2100W', 'F2300C', 'F2550W']

        self.astrosat_fuv_bands = ['F148W', 'F154W', 'F169M', 'F172M']
        self.astrosat_nuv_bands = ['N219M', 'N242W', 'N245M', 'N263M', 'N279N']

        # band wavelength taken from
        # http://svo2.cab.inta-csic.es/svo/theory/fps3/index.php?mode=browse&gname=HST&gname2=ACS_WFC&asttype=
        self.hst_acs_wfc1_bands_wave = {
            'FR388N': {'mean_wave': 3881.71, 'min_wave': 3811.90, 'max_wave': 3951.43},
            'FR423N': {'mean_wave': 4230.39, 'min_wave': 4159.35, 'max_wave': 4301.72},
            'F435W': {'mean_wave': 4360.06, 'min_wave': 3610.23, 'max_wave': 4883.77},
            'FR459M': {'mean_wave': 4592.76, 'min_wave': 4278.58, 'max_wave': 4907.27},
            'FR462N': {'mean_wave': 4620.13, 'min_wave': 4543.06, 'max_wave': 4697.34},
            'F475W': {'mean_wave': 4802.31, 'min_wave': 3863.58, 'max_wave': 5562.80},
            'F502N': {'mean_wave': 5023.13, 'min_wave': 4966.53, 'max_wave': 5079.84},
            'FR505N': {'mean_wave': 5050.39, 'min_wave': 4955.61, 'max_wave': 5145.27},
            'F555W': {'mean_wave': 5397.60, 'min_wave': 4584.27, 'max_wave': 6208.72},
            'FR551N': {'mean_wave': 5510.30, 'min_wave': 5425.09, 'max_wave': 5595.43},
            'F550M': {'mean_wave': 5588.24, 'min_wave': 5248.45, 'max_wave': 5931.37},
            'FR601N': {'mean_wave': 6010.50, 'min_wave': 5907.50, 'max_wave': 6113.41},
            'F606W': {'mean_wave': 6035.73, 'min_wave': 4634.30, 'max_wave': 7180.10},
            'F625W': {'mean_wave': 6352.46, 'min_wave': 5446.00, 'max_wave': 7099.62},
            'FR647M': {'mean_wave': 6476.15, 'min_wave': 6027.10, 'max_wave': 6925.31},
            'FR656N': {'mean_wave': 6560.36, 'min_wave': 6445.62, 'max_wave': 6675.13},
            'F658N': {'mean_wave': 6584.10, 'min_wave': 6509.98, 'max_wave': 6659.44},
            'F660N': {'mean_wave': 6599.50, 'min_wave': 6562.00, 'max_wave': 6642.91},
            'FR716N': {'mean_wave': 7160.04, 'min_wave': 7040.58, 'max_wave': 7279.37},
            'POL_UV': {'mean_wave': 7294.11, 'min_wave': 3306.71, 'max_wave': 10881.83},
            'POL_V': {'mean_wave': 7523.49, 'min_wave': 3623.57, 'max_wave': 10884.79},
            'G800L': {'mean_wave': 7704.08, 'min_wave': 5274.80, 'max_wave': 10827.01},
            'F775W': {'mean_wave': 7730.77, 'min_wave': 6803.72, 'max_wave': 8631.82},
            'FR782N': {'mean_wave': 7819.44, 'min_wave': 7687.33, 'max_wave': 7951.48},
            'F814W': {'mean_wave': 8129.21, 'min_wave': 6869.59, 'max_wave': 9632.01},
            'FR853N': {'mean_wave': 8528.80, 'min_wave': 8395.60, 'max_wave': 8661.95},
            'F892N': {'mean_wave': 8915.37, 'min_wave': 8788.92, 'max_wave': 9035.83},
            'FR914M': {'mean_wave': 9079.84, 'min_wave': 8350.59, 'max_wave': 9802.04},
            'F850LP': {'mean_wave': 9080.26, 'min_wave': 8007.01, 'max_wave': 10862.13},
            'FR931N': {'mean_wave': 9306.31, 'min_wave': 9133.28, 'max_wave': 9479.02},
            'FR1016N': {'mean_wave': 10150.22, 'min_wave': 9966.99, 'max_wave': 10335.19},
        }
        self.hst_wfc3_uvis1_bands_wave = {
            'F218W': {'mean_wave': 2231.14, 'min_wave': 1990.00, 'max_wave': 2626.06},
            'FQ232N': {'mean_wave': 2327.12, 'min_wave': 2294.18, 'max_wave': 2362.03},
            'F225W': {'mean_wave': 2377.24, 'min_wave': 1990.00, 'max_wave': 3005.66},
            'FQ243N': {'mean_wave': 2420.59, 'min_wave': 2388.36, 'max_wave': 2453.82},
            'F275W': {'mean_wave': 2718.36, 'min_wave': 2289.09, 'max_wave': 3124.83},
            'F280N': {'mean_wave': 2796.98, 'min_wave': 2761.85, 'max_wave': 2840.93},
            'F300X': {'mean_wave': 2867.82, 'min_wave': 2161.14, 'max_wave': 4196.56},
            'F336W': {'mean_wave': 3365.86, 'min_wave': 3015.90, 'max_wave': 3708.23},
            'F343N': {'mean_wave': 3438.50, 'min_wave': 3262.54, 'max_wave': 3644.62},
            'F373N': {'mean_wave': 3730.19, 'min_wave': 3693.44, 'max_wave': 3770.53},
            'FQ378N': {'mean_wave': 3792.78, 'min_wave': 3724.59, 'max_wave': 3871.99},
            'FQ387N': {'mean_wave': 3873.61, 'min_wave': 3849.58, 'max_wave': 3897.86},
            'F390M': {'mean_wave': 3898.62, 'min_wave': 3724.53, 'max_wave': 4052.27},
            'F390W': {'mean_wave': 3952.50, 'min_wave': 3895.07, 'max_wave': 4018.40},
            'F395N': {'mean_wave': 3955.38, 'min_wave': 3259.29, 'max_wave': 4470.97},
            'F410M': {'mean_wave': 4109.81, 'min_wave': 3984.47, 'max_wave': 4238.00},
            'FQ422M': {'mean_wave': 4219.70, 'min_wave': 4114.99, 'max_wave': 4327.89},
            'F438W': {'mean_wave': 4338.57, 'min_wave': 3898.67, 'max_wave': 4710.43},
            'FQ436N': {'mean_wave': 4367.41, 'min_wave': 4334.24, 'max_wave': 4401.70},
            'FQ437N': {'mean_wave': 4371.30, 'min_wave': 4348.50, 'max_wave': 4393.42},
            'G280': {'mean_wave': 4628.43, 'min_wave': 4546.74, 'max_wave': 4830.80},
            'F467M': {'mean_wave': 4683.55, 'min_wave': 4653.86, 'max_wave': 4723.12},
            'F469N': {'mean_wave': 4688.29, 'min_wave': 2000.00, 'max_wave': 9500.00},
            'F475W': {'mean_wave': 4827.71, 'min_wave': 3945.10, 'max_wave': 5584.63},
            'F487N': {'mean_wave': 4871.54, 'min_wave': 4827.63, 'max_wave': 4917.77},
            'FQ492N': {'mean_wave': 4933.83, 'min_wave': 4859.29, 'max_wave': 5012.92},
            'F502N': {'mean_wave': 5009.93, 'min_wave': 4963.34, 'max_wave': 5058.54},
            'F475X': {'mean_wave': 5076.23, 'min_wave': 3742.28, 'max_wave': 6964.25},
            'FQ508N': {'mean_wave': 5091.59, 'min_wave': 5003.28, 'max_wave': 5181.21},
            'F555W': {'mean_wave': 5388.55, 'min_wave': 4382.83, 'max_wave': 7098.13},
            'F547M': {'mean_wave': 5459.04, 'min_wave': 5040.52, 'max_wave': 5912.28},
            'FQ575N': {'mean_wave': 5756.92, 'min_wave': 5742.15, 'max_wave': 5771.15},
            'F606W': {'mean_wave': 5999.27, 'min_wave': 4712.79, 'max_wave': 7208.10},
            'F200LP': {'mean_wave': 6043.00, 'min_wave': 1990.00, 'max_wave': 10809.67},
            'FQ619N': {'mean_wave': 6198.49, 'min_wave': 6146.74, 'max_wave': 6253.36},
            'F621M': {'mean_wave': 6227.39, 'min_wave': 5820.99, 'max_wave': 6619.32},
            'F625W': {'mean_wave': 6291.29, 'min_wave': 5417.26, 'max_wave': 7140.56},
            'F631N': {'mean_wave': 6304.27, 'min_wave': 6263.86, 'max_wave': 6346.80},
            'FQ634N': {'mean_wave': 6349.37, 'min_wave': 6294.10, 'max_wave': 6407.48},
            'F645N': {'mean_wave': 6453.59, 'min_wave': 6383.19, 'max_wave': 6521.54},
            'F350LP': {'mean_wave': 6508.00, 'min_wave': 3210.40, 'max_wave': 10806.90},
            'F656N': {'mean_wave': 6561.54, 'min_wave': 6548.77, 'max_wave': 6574.27},
            'F657N': {'mean_wave': 6566.93, 'min_wave': 6476.03, 'max_wave': 6674.16},
            'F658N': {'mean_wave': 6585.64, 'min_wave': 6566.93, 'max_wave': 6604.64},
            'F665N': {'mean_wave': 6656.23, 'min_wave': 6552.78, 'max_wave': 6755.94},
            'FQ672N': {'mean_wave': 6717.13, 'min_wave': 6702.24, 'max_wave': 6731.72},
            'FQ674N': {'mean_wave': 6730.58, 'min_wave': 6716.75, 'max_wave': 6744.20},
            'F673N': {'mean_wave': 6766.27, 'min_wave': 6681.24, 'max_wave': 6860.32},
            'F680N': {'mean_wave': 6880.13, 'min_wave': 6631.44, 'max_wave': 7145.80},
            'F689M': {'mean_wave': 6885.92, 'min_wave': 6451.41, 'max_wave': 7326.57},
            'FQ727N': {'mean_wave': 7275.84, 'min_wave': 7216.95, 'max_wave': 7336.29},
            'FQ750N': {'mean_wave': 7502.54, 'min_wave': 7436.57, 'max_wave': 7570.57},
            'F763M': {'mean_wave': 7623.09, 'min_wave': 7164.92, 'max_wave': 8092.73},
            'F600LP': {'mean_wave': 7656.67, 'min_wave': 5928.33, 'max_wave': 10815.55},
            'F775W': {'mean_wave': 7683.41, 'min_wave': 6870.61, 'max_wave': 8576.34},
            'F814W': {'mean_wave': 8117.36, 'min_wave': 6978.64, 'max_wave': 9695.01},
            'F845M': {'mean_wave': 8449.34, 'min_wave': 7896.09, 'max_wave': 9019.64},
            'FQ889N': {'mean_wave': 8892.56, 'min_wave': 8706.82, 'max_wave': 9030.90},
            'FQ906N': {'mean_wave': 9058.19, 'min_wave': 8870.92, 'max_wave': 9159.61},
            'F850LP': {'mean_wave': 9207.49, 'min_wave': 8254.88, 'max_wave': 10980.16},
            'FQ924N': {'mean_wave': 9247.91, 'min_wave': 9146.28, 'max_wave': 9336.10},
            'FQ937N': {'mean_wave': 9372.90, 'min_wave': 9262.19, 'max_wave': 9486.78},
            'F953N': {'mean_wave': 9531.11, 'min_wave': 9320.27, 'max_wave': 9724.71},
        }
        self.nircam_bands_wave = {
            'F070W': {'mean_wave': 7088.30, 'min_wave': 6048.20, 'max_wave': 7927.07},
            'F090W': {'mean_wave': 9083.40, 'min_wave': 7881.88, 'max_wave': 10243.08},
            'F115W': {'mean_wave': 11623.89, 'min_wave': 9975.60, 'max_wave': 13058.40},
            'F140M': {'mean_wave': 14074.46, 'min_wave': 13042.25, 'max_wave': 15058.58},
            'F150W': {'mean_wave': 15104.23, 'min_wave': 13041.19, 'max_wave': 16948.89},
            'F162M': {'mean_wave': 16296.59, 'min_wave': 15126.16, 'max_wave': 17439.17},
            'F164N': {'mean_wave': 16445.95, 'min_wave': 16171.41, 'max_wave': 16717.72},
            'F150W2': {'mean_wave': 17865.58, 'min_wave': 9774.71, 'max_wave': 23946.87},
            'F182M': {'mean_wave': 18494.30, 'min_wave': 16959.53, 'max_wave': 20010.97},
            'F187N': {'mean_wave': 18739.65, 'min_wave': 18445.28, 'max_wave': 19029.98},
            'F200W': {'mean_wave': 20028.15, 'min_wave': 17249.08, 'max_wave': 22596.64},
            'F210M': {'mean_wave': 20982.22, 'min_wave': 19618.54, 'max_wave': 22337.29},
            'F212N': {'mean_wave': 21213.97, 'min_wave': 20900.93, 'max_wave': 21524.99},
            'F250M': {'mean_wave': 25049.39, 'min_wave': 23935.49, 'max_wave': 26177.91},
            'F277W': {'mean_wave': 27844.64, 'min_wave': 23673.12, 'max_wave': 32203.22},
            'F300M': {'mean_wave': 29940.44, 'min_wave': 27703.55, 'max_wave': 32505.92},
            'F323N': {'mean_wave': 32369.29, 'min_wave': 32046.29, 'max_wave': 32761.07},
            'F322W2': {'mean_wave': 33334.98, 'min_wave': 23851.45, 'max_wave': 41234.69},
            'F335M': {'mean_wave': 33675.24, 'min_wave': 31203.36, 'max_wave': 36442.23},
            'F356W': {'mean_wave': 35934.49, 'min_wave': 30732.91, 'max_wave': 40801.26},
            'F360M': {'mean_wave': 36298.10, 'min_wave': 33260.34, 'max_wave': 39037.39},
            'F405N': {'mean_wave': 40517.39, 'min_wave': 40097.87, 'max_wave': 40966.10},
            'F410M': {'mean_wave': 40886.55, 'min_wave': 37763.56, 'max_wave': 44048.41},
            'F430M': {'mean_wave': 42829.39, 'min_wave': 41227.68, 'max_wave': 44448.79},
            'F444W': {'mean_wave': 44393.50, 'min_wave': 38039.57, 'max_wave': 50995.50},
            'F460M': {'mean_wave': 46315.57, 'min_wave': 44652.64, 'max_wave': 48146.41},
            'F466N': {'mean_wave': 46545.31, 'min_wave': 46021.35, 'max_wave': 47042.62},
            'F470N': {'mean_wave': 47078.82, 'min_wave': 46553.98, 'max_wave': 47566.82},
            'F480M': {'mean_wave': 48213.27, 'min_wave': 45820.02, 'max_wave': 50919.02},
        }
        self.miri_bands_wave = {
            'F560W': {'mean_wave': 56651.28, 'min_wave': 48944.36, 'max_wave': 64279.58},
            'F770W': {'mean_wave': 77111.39, 'min_wave': 64802.79, 'max_wave': 88382.09},
            'F1000W': {'mean_wave': 99981.09, 'min_wave': 87645.92, 'max_wave': 111053.33},
            'F1065C': {'mean_wave': 105681.52, 'min_wave': 100226.73, 'max_wave': 111577.48},
            'F1140C': {'mean_wave': 113156.52, 'min_wave': 107357.90, 'max_wave': 119593.95},
            'F1130W': {'mean_wave': 113159.44, 'min_wave': 106439.78, 'max_wave': 119874.08},
            'F1280W': {'mean_wave': 128738.34, 'min_wave': 112674.80, 'max_wave': 143435.71},
            'F1500W': {'mean_wave': 151469.08, 'min_wave': 131345.04, 'max_wave': 171580.84},
            'F1550C': {'mean_wave': 155219.65, 'min_wave': 149413.67, 'max_wave': 161556.33},
            'F1800W': {'mean_wave': 180508.31, 'min_wave': 160441.28, 'max_wave': 203000.78},
            'F2100W': {'mean_wave': 209373.20, 'min_wave': 179077.84, 'max_wave': 244780.51},
            'F2300C': {'mean_wave': 227630.49, 'min_wave': 196484.64, 'max_wave': 262492.33},
            'F2550W': {'mean_wave': 254994.19, 'min_wave': 223494.34, 'max_wave': 299940.00},
        }
        self.astrosat_bands_wave = {
            'F148W': {'mean_wave': 1481.00, 'min_wave': 1250.27, 'max_wave': 1799.21},
            'F148W_old': {'mean_wave': 1481.00, 'min_wave': 1250.27, 'max_wave': 1750.00},
            'F148Wa': {'mean_wave': 1485.00, 'min_wave': 1250.29, 'max_wave': 1750.00},
            'F154W': {'mean_wave': 1541.00, 'min_wave': 1340.58, 'max_wave': 1799.24},
            'F154W_old': {'mean_wave': 1541.00, 'min_wave': 1340.58, 'max_wave': 1799.24},
            'F169M': {'mean_wave': 1608.00, 'min_wave': 1428.94, 'max_wave': 1799.26},
            'F169M_old': {'mean_wave': 1608.00, 'min_wave': 1428.98, 'max_wave': 1799.26},
            'F172M': {'mean_wave': 1717.00, 'min_wave': 1620.00, 'max_wave': 1828.51},
            'F172M_old': {'mean_wave': 1717.00, 'min_wave': 1620.00, 'max_wave': 1828.51},
            'N219M': {'mean_wave': 2196.00, 'min_wave': 1948.32, 'max_wave': 2410.00},
            'N219M_old': {'mean_wave': 2196.00, 'min_wave': 1948.08, 'max_wave': 2409.50},
            'N242W': {'mean_wave': 2418.00, 'min_wave': 1700.00, 'max_wave': 3050.00},
            'N242W_old': {'mean_wave': 2418.00, 'min_wave': 1700.00, 'max_wave': 3050.00},
            'N245M': {'mean_wave': 2447.00, 'min_wave': 2195.08, 'max_wave': 2634.78},
            'N245M_old': {'mean_wave': 2447.00, 'min_wave': 2194.69, 'max_wave': 2634.71},
            'N263M': {'mean_wave': 2632.00, 'min_wave': 2462.90, 'max_wave': 2842.96},
            'N263M_old': {'mean_wave': 2632.00, 'min_wave': 2462.42, 'max_wave': 2842.46},
            'N279N': {'mean_wave': 2792.00, 'min_wave': 2722.26, 'max_wave': 2877.19},
            'N279N_old': {'mean_wave': 2792.00, 'min_wave': 2722.06, 'max_wave': 2877.15},
            'VIS1': {'mean_wave': 3466.00, 'min_wave': 3186.97, 'max_wave': 3738.73},
            'VIS2': {'mean_wave': 3909.00, 'min_wave': 3621.46, 'max_wave': 4166.70},
            'BK7': {'mean_wave': 4200.00, 'min_wave': 3076.29, 'max_wave': 5495.00},
            'ND1': {'mean_wave': 4354.00, 'min_wave': 3584.82, 'max_wave': 5452.59},
            'VIS3': {'mean_wave': 4614.00, 'min_wave': 3878.26, 'max_wave': 5325.00},
            }
        # hst encircled energy for 50% and 80% of a point source
        # the computed values are interpolated for the aperture energyenclosure for the UVIS1 and UVIS2 table found at:
        # https://www.stsci.edu/hst/instrumentation/wfc3/data-analysis/photometric-calibration/uvis-encircled-energy
        # the interpolation procedure can be found at ``../hst_psf_ee/compute_hst_psf_ee.py``
        self.hst_encircle_apertures_wfc3_uvis1_arcsec = {
            'F275W': {'ee50': 0.0822707423580786, 'ee80': 0.21022900763358784},
            'F300X': {'ee50': 0.07183566878980892, 'ee80': 0.18605544880592925},
            'F280N': {'ee50': 0.0742747695104532, 'ee80': 0.19406145107152087},
            'F336W': {'ee50': 0.08497576396206534, 'ee80': 0.1871036106750393},
            'F343N': {'ee50': 0.06925450666336849, 'ee80': 0.16801159420289863},
            'F373N': {'ee50': 0.06637465395262997, 'ee80': 0.1573038842345773},
            'F390M': {'ee50': 0.06754620972933206, 'ee80': 0.1618176197836168},
            'F390W': {'ee50': 0.06913956513659172, 'ee80': 0.1608947211452431},
            'F395N': {'ee50': 0.06875161033065456, 'ee80': 0.16032039433148496},
            'F410M': {'ee50': 0.09371942446043166, 'ee80': 0.18274568084711132},
            'F438W': {'ee50': 0.06903736698836921, 'ee80': 0.1557715430861724},
            'F467M': {'ee50': 0.06795220286417951, 'ee80': 0.15191359135913596},
            'F469N': {'ee50': 0.06886956521739131, 'ee80': 0.15647894645642005},
            'F475W': {'ee50': 0.069850040445523, 'ee80': 0.15439822165766914},
            'F487N': {'ee50': 0.09325647899910634, 'ee80': 0.17768742058449816},
            'F475X': {'ee50': 0.0834711893424643, 'ee80': 0.1957687914096414},
            'F200LP': {'ee50': 0.07210149198176122, 'ee80': 0.1558672656136792},
            'F502N': {'ee50': 0.06777562136104677, 'ee80': 0.1479508771929825},
            'F555W': {'ee50': 0.07046691129950652, 'ee80': 0.15283876757403533},
            'F547M': {'ee50': 0.0712762460068852, 'ee80': 0.15255306603773588},
            'F350LP': {'ee50': 0.07336316039980961, 'ee80': 0.1607216494845361},
            'F606W': {'ee50': 0.07091174788741343, 'ee80': 0.15282094594594597},
            'F621M': {'ee50': 0.07030923161609948, 'ee80': 0.1496267517691134},
            'F625W': {'ee50': 0.07346899099215864, 'ee80': 0.1552398847551046},
            'F631N': {'ee50': 0.06967976144172176, 'ee80': 0.15119572661279282},
            'F645N': {'ee50': 0.06969593034760241, 'ee80': 0.14867894100255344},
            'F656N': {'ee50': 0.07031221060986903, 'ee80': 0.15098054374436287},
            'F657N': {'ee50': 0.07014802901499984, 'ee80': 0.15021556256572033},
            'F658N': {'ee50': 0.0708986229419885, 'ee80': 0.15386164171399075},
            'F665N': {'ee50': 0.0706210006299526, 'ee80': 0.1514525139664805},
            'F673N': {'ee50': 0.09633659008890062, 'ee80': 0.18216850586792785},
            'F689M': {'ee50': 0.0968180044230519, 'ee80': 0.18145735392881132},
            'F680N': {'ee50': 0.0721983626358878, 'ee80': 0.15341682419659736},
            'F600LP': {'ee50': 0.07462507022703989, 'ee80': 0.15720930232558142},
            'F763M': {'ee50': 0.07236761426978819, 'ee80': 0.15524155844155846},
            'F775W': {'ee50': 0.0733488841694809, 'ee80': 0.15742775742775744},
            'F814W': {'ee50': 0.07625649913344887, 'ee80': 0.1674208144796381},
            'F845M': {'ee50': 0.07625649913344887, 'ee80': 0.1674208144796381},
            'F850LP': {'ee50': 0.07625649913344887, 'ee80': 0.1674208144796381},
            'F953N': {'ee50': 0.07625649913344887, 'ee80': 0.1674208144796381}
        }
        self.hst_encircle_apertures_wfc3_uvis2_arcsec = {
            'F275W': {'ee50': 0.11002563163676309, 'ee80': 0.2126182965299685},
            'F300X': {'ee50': 0.07362485839132546, 'ee80': 0.18871158725683682},
            'F280N': {'ee50': 0.07019743109621891, 'ee80': 0.18288455772113948},
            'F336W': {'ee50': 0.06656083690660827, 'ee80': 0.15241806908768826},
            'F343N': {'ee50': 0.06917672954052154, 'ee80': 0.155386012715713},
            'F373N': {'ee50': 0.06940505900113997, 'ee80': 0.15713519952352592},
            'F390M': {'ee50': 0.06846401585532019, 'ee80': 0.15556587707075403},
            'F390W': {'ee50': 0.06709837054918527, 'ee80': 0.14826257459505543},
            'F395N': {'ee50': 0.06823408871745419, 'ee80': 0.15171940763834765},
            'F410M': {'ee50': 0.09201353485224453, 'ee80': 0.17397061426801208},
            'F438W': {'ee50': 0.06631333191837725, 'ee80': 0.14449639655475485},
            'F467M': {'ee50': 0.0663031226199543, 'ee80': 0.1464906333630687},
            'F469N': {'ee50': 0.06619528826366065, 'ee80': 0.1473578475336323},
            'F475W': {'ee50': 0.06864697401920186, 'ee80': 0.14801877934272303},
            'F487N': {'ee50': 0.06836516751083176, 'ee80': 0.15293060409385922},
            'F475X': {'ee50': 0.07797421609680502, 'ee80': 0.1923851203501094},
            'F200LP': {'ee50': 0.07087352362204724, 'ee80': 0.15511143911439118},
            'F502N': {'ee50': 0.06698717750656574, 'ee80': 0.1469007055584237},
            'F555W': {'ee50': 0.06755263238774319, 'ee80': 0.14312530552387162},
            'F547M': {'ee50': 0.0684225921892018, 'ee80': 0.14788227767114526},
            'F350LP': {'ee50': 0.07050133218999785, 'ee80': 0.15470160116448328},
            'F606W': {'ee50': 0.06889893283113621, 'ee80': 0.1469464285714286},
            'F621M': {'ee50': 0.06885909850802763, 'ee80': 0.1482506682506683},
            'F625W': {'ee50': 0.07011921613035137, 'ee80': 0.15006351446718422},
            'F631N': {'ee50': 0.07010144642974017, 'ee80': 0.15391515497786035},
            'F645N': {'ee50': 0.06977947973062194, 'ee80': 0.1532566396818634},
            'F656N': {'ee50': 0.07016378100140383, 'ee80': 0.15726596491228073},
            'F657N': {'ee50': 0.07006809917355372, 'ee80': 0.15509116409537171},
            'F658N': {'ee50': 0.07000720791560186, 'ee80': 0.15564334085778783},
            'F665N': {'ee50': 0.07103805297835149, 'ee80': 0.15450959488272922},
            'F673N': {'ee50': 0.0703399377112186, 'ee80': 0.15386321626617377},
            'F689M': {'ee50': 0.07224687239366137, 'ee80': 0.15868572861800856},
            'F680N': {'ee50': 0.07155458953352282, 'ee80': 0.15526838466373355},
            'F600LP': {'ee50': 0.07462507022703989, 'ee80': 0.15720930232558142},
            'F763M': {'ee50': 0.07236761426978819, 'ee80': 0.15524155844155846},
            'F775W': {'ee50': 0.0733488841694809, 'ee80': 0.15742775742775744},
            'F814W': {'ee50': 0.09630835117773019, 'ee80': 0.18683307332293292},
            'F845M': {'ee50': 0.09630835117773019, 'ee80': 0.18683307332293292},
            'F850LP': {'ee50': 0.09630835117773019, 'ee80': 0.18683307332293292},
            'F953N': {'ee50': 0.09630835117773019, 'ee80': 0.18683307332293292}
        }
        self.hst_encircle_apertures_acs_wfc1_arcsec = {
            'F435W': {'ee50': 0.07552552552552552, 'ee80': 0.15851063829787237},
            'F475W': {'ee50': 0.0750733137829912, 'ee80': 0.15625000000000003},
            'F502N': {'ee50': 0.07514619883040935, 'ee80': 0.15625000000000003},
            'F555W': {'ee50': 0.07529411764705882, 'ee80': 0.1563829787234043},
            'F550M': {'ee50': 0.07544378698224852, 'ee80': 0.15652173913043482},
            'F606W': {'ee50': 0.07582582582582582, 'ee80': 0.15568181818181823},
            'F625W': {'ee50': 0.07615384615384616, 'ee80': 0.15581395348837213},
            'F658N': {'ee50': 0.07640625000000001, 'ee80': 0.15681818181818186},
            'F660N': {'ee50': 0.07648902821316614, 'ee80': 0.15681818181818186},
            'F775W': {'ee50': 0.07888513513513513, 'ee80': 0.16603773584905665},
            'F814W': {'ee50': 0.08079584775086505, 'ee80': 0.17500000000000007},
            'F892N': {'ee50': 0.0914179104477612, 'ee80': 0.22096774193548396},
            'F850LP': {'ee50': 0.09393939393939393, 'ee80': 0.23529411764705882}
        }
        # nircam encircled energy for filters
        # taken from Table 2 in:
        # https://jwst-docs.stsci.edu/jwst-near-infrared-camera/nircam-performance/nircam-point-spread-functions
        # We use the empirical EE estimation
        # Latest updates of webpage 27 Nov 2022
        # version April/23/2023
        self.nircam_encircle_apertures_arcsec = {
            'F070W': {'ee50': 0.038, 'ee80': 0.173},
            'F090W': {'ee50': 0.034, 'ee80': 0.157},
            'F115W': {'ee50': 0.029, 'ee80': 0.148},
            'F140M': {'ee50': 0.031, 'ee80': 0.149},
            'F150W': {'ee50': None, 'ee80': None},
            'F150W2': {'ee50': 0.033, 'ee80': 0.146},
            'F162M': {'ee50': 0.033, 'ee80': 0.141},
            'F164N': {'ee50': 0.034, 'ee80': 0.140},
            'F182M': {'ee50': 0.037, 'ee80': 0.138},
            'F187N': {'ee50': 0.038, 'ee80': 0.138},
            'F200W': {'ee50': 0.040, 'ee80': 0.144},
            'F210M': {'ee50': 0.042, 'ee80': 0.146},
            'F212N': {'ee50': 0.043, 'ee80': 0.147},
            'F250M': {'ee50': 0.049, 'ee80': 0.176},
            'F277W': {'ee50': 0.054, 'ee80': 0.192},
            'F300M': {'ee50': 0.059, 'ee80': 0.200},
            'F322W2': {'ee50': None, 'ee80': None},
            'F323N': {'ee50': 0.064, 'ee80': 0.213},
            'F335M': {'ee50': 0.066, 'ee80': 0.221},
            'F356W': {'ee50': 0.069, 'ee80': 0.230},
            'F360M': {'ee50': 0.071, 'ee80': 0.234},
            'F405N': {'ee50': 0.078, 'ee80': 0.253},
            'F410M': {'ee50': 0.079, 'ee80': 0.258},
            'F430M': {'ee50': 0.083, 'ee80': 0.269},
            'F444W': {'ee50': 0.085, 'ee80': 0.276},
            'F460M': {'ee50': 0.090, 'ee80': 0.290},
            'F466N': {'ee50': 0.090, 'ee80': 0.292},
            'F470N': {'ee50': 0.091, 'ee80': 0.294},
            'F480M': {'ee50': 0.094, 'ee80': 0.301}
        }
        # MIRI encircled energy for filters
        # taken from Table 2 in:
        # https://jwst-docs.stsci.edu/jwst-mid-infrared-instrument/miri-performance/miri-point-spread-functions
        # version 11/11/2022
        self.miri_encircle_apertures_arcsec = {
            'F560W': {'ee50': 0.131, 'ee80': 0.422},
            'F770W': {'ee50': 0.168, 'ee80': 0.519},
            'F1000W': {'ee50': 0.209, 'ee80': 0.636},
            'F1130W': {'ee50': 0.236, 'ee80': 0.712},
            'F1280W': {'ee50': 0.266, 'ee80': 0.801},
            'F1500W': {'ee50': 0.307, 'ee80': 0.932},
            'F1800W': {'ee50': 0.367, 'ee80': 1.110},
            'F2100W': {'ee50': 0.420, 'ee80': 1.276},
            'F2550W': {'ee50': 0.510, 'ee80': 1.545}
        }

