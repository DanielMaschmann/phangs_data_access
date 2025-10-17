import os
import tarfile

import numpy as np
import matplotlib.pyplot as plt

from astropy.table import Table

from synphot import Observation

from astropy.io import fits

file_name = '/media/benutzer/derka_derka/data/hst/ngc5194/F128N_HST_WFC3_IR_IVM_drz.fits'
hdu = fits.open(file_name)
print(hdu.info())
header = hdu['SCI'].header
data = hdu['SCI'].data
mjd = header['ROUTTIME']

print(header)

print(mjd)



# tar_archive = 'hlsp_reference-atlases_hst_multi_everything_multi_v11_sed.tar'
# extract_to = 'hlsp_reference-atlases_hst_multi_everything_multi_v11_sed'
# with tarfile.open(tar_archive, 'r') as tar:
#     tar.extractall(path=extract_to, filter='data')

# os.environ['PYSYN_CDBS'] = 'hlsp_reference-atlases_hst_multi_everything_multi_v11_sed/grp/redcat/trds/'

os.environ['PYSYN_CDBS'] = ('/home/benutzer/software/python_packages/phangs_data_access/meta_data/'
                                    'wfc3_calibration_files/'
                                    'hlsp_reference-atlases_hst_multi_everything_multi_v11_sed/grp/redcat/trds/')


import stsynphot as stsyn

vega_url = 'https://ssb.stsci.edu/trds/calspec/alpha_lyr_stis_010.fits'
stsyn.Vega = stsyn.spectrum.SourceSpectrum.from_file(vega_url)

detectors = ['ir']

filtnames = ['f098m', 'f105w', 'f110w', 'f125w', 'f126n', 'f127m', 'f128n', 'f130n',
             'f132n', 'f139m', 'f140w', 'f153m', 'f160w', 'f164n', 'f167n']

# aper = '6.0'
aper = '0.385'

# 5.172772917018699e-19
# 12831.213891305677 Angstrom

# obsmode = f'wfc3, {detector}, {filt}, mjd#{mjd}, aper#{aper}'
obsmode = f'wfc3, ir, f128n, mjd#{mjd}, aper#{aper}'

bp = stsyn.band(obsmode)

photflam = bp.unit_response(stsyn.conf.area)  # inverse sensitivity in flam

photplam = bp.pivot() # pivot wavelength in angstroms

print(photflam.value)
print(photplam)
