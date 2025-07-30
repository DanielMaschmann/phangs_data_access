"""
Access structure for HST star cluster access
"""
import os.path
import numpy as np
from pathlib import Path
from astropy.table import Table, hstack
import astropy.units as u

from phangs_data_access import phangs_access_config, helper_func, phangs_info, dust_tools
from phangs_data_access.sample_access import SampleAccess
from phangs_data_access.phot_access import PhotAccess
from phangs_data_access.gas_access import GasAccess
from phangs_data_access.spec_access import SpecAccess

# from dust_tools.extinction_tools import ExtinctionTools


class ClusterCatAccess:
    """
    This class is the basis to access star cluster catalogs the catalog content is described in the papers:
    Maschmann et al. 2024 and Thilker et al. 2024

    To Do:
    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    change the extended catalog access to a regular catalog access
    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    """

    def __init__(self,
                 phangs_hst_cluster_cat_sed_version='ground_based_ha',
                 phangs_hst_cluster_cat_ver='v1',
                 phangs_hst_cluster_cat_extend_sed_ver='HST_Ha_nircam'):
        """

        """
        self.phangs_hst_cluster_cat_data_path = (
            phangs_access_config.phangs_config_dict)['phangs_hst_cluster_cat_data_path']
        self.phangs_hst_cluster_cat_release = (
            phangs_access_config.phangs_config_dict)['phangs_hst_cluster_cat_release']
        self.phangs_hst_cluster_cat_sed_version = phangs_hst_cluster_cat_sed_version
        self.phangs_hst_cluster_cat_ver = phangs_hst_cluster_cat_ver
        self.phangs_hst_cluster_cat_extend_sed_ver = phangs_hst_cluster_cat_extend_sed_ver

        self.hst_cc_data = {}
        self.extend_data = {}

        super().__init__()

    def open_hst_cluster_cat(self, target, classify='human', cluster_class='class12', cat_type='obs'):
        """

        Function to load cluster catalogs into the constructor

        Parameters
        ----------
        target : str
        classify : str
        cluster_class : str
        cat_type : str

        Returns
        -------
        ``astropy.table.Table``
        """

        # assemble file name and path
        # get all instruments involved
        instruments = ''
        if phangs_info.hst_cluster_cat_obs_band_dict[target]['acs']:
            instruments += 'acs'
            if phangs_info.hst_cluster_cat_obs_band_dict[target]['uvis']:
                instruments += '-uvis'
        else:
            instruments += 'uvis'

        if cat_type in ['sed', 'obs-sed']:
            if self.phangs_hst_cluster_cat_sed_version == 'ground_based_ha':
                cat_type += '-ground-halpha'
            elif self.phangs_hst_cluster_cat_sed_version == 'hst_ha':
                cat_type += '-hst-halpha'


        if cluster_class == 'candidates':
            file_string = Path('hlsp_phangs-cat_hst_%s_%s_multi_%s_%s-candidates.fits' %
                               (instruments, target, self.phangs_hst_cluster_cat_ver, cat_type))

        else:
            if classify == 'human':
                classify_str = 'human'
            elif classify == 'ml':
                classify_str = 'machine'
            else:
                raise KeyError('classify must be human or ml')

            if cluster_class == 'class12':
                cluster_str = 'cluster-class12'
            elif cluster_class == 'class3':
                cluster_str = 'compact-association-class3'
            else:
                raise KeyError('cluster_class must be class12 or class3')

            file_string = Path('hlsp_phangs-cat_hst_%s_%s_multi_%s_%s-%s-%s.fits'
                               % (instruments, target, self.phangs_hst_cluster_cat_ver, cat_type, classify_str,
                                  cluster_str))

        cluster_dict_path = (Path(self.phangs_hst_cluster_cat_data_path) /
                             Path(self.phangs_hst_cluster_cat_release + '/catalogs'))

        file_path = cluster_dict_path / file_string

        if not os.path.isfile(file_path):
            raise FileNotFoundError('there is no HST cluster catalog for the target ', target,
                                    ' make sure that the file ', file_path, ' exists')
        return Table.read(file_path)

    def load_hst_cluster_cat(self, target_list=None, classify_list=None, cluster_class_list=None):
        """

        Function to load Phangs sample table into the constructor.
        Required to access global sample data

        Parameters
        ----------
        target_list : list
        classify_list : list
        cluster_class_list : list

        Returns
        -------
        None
        """
        if target_list is None:
            target_list = phangs_info.hst_cluster_cat_target_list

        if classify_list is None:
            classify_list = ['human', 'ml']

        if cluster_class_list is None:
            cluster_class_list = ['class12', 'class3']

        for target in target_list:
            for classify in classify_list:
                for cluster_class in cluster_class_list:
                    # check if catalog is already loaded
                    if (str(target) + '_' + classify + '_' + cluster_class) in self.hst_cc_data:
                        continue
                    if cluster_class in ['class12', 'class3']:
                        cluster_catalog_obs = self.open_hst_cluster_cat(target=target, classify=classify,
                                                                        cluster_class=cluster_class)
                        cluster_catalog_sed = self.open_hst_cluster_cat(target=target, classify=classify,
                                                                        cluster_class=cluster_class, cat_type='sed')

                        names_obs = list(cluster_catalog_obs.colnames)
                        names_sed = list(cluster_catalog_sed.colnames)
                        # get list of columns which are double
                        all_names_list = names_obs + names_sed
                        identifier_names = []
                        [identifier_names.append(x) for x in all_names_list if all_names_list.count(x) == 2 and
                         x not in identifier_names]

                        sed_name_list_double_id = identifier_names + names_sed
                        unique_sed_names = []
                        [unique_sed_names.append(x) for x in sed_name_list_double_id
                         if sed_name_list_double_id.count(x) == 1 and x not in unique_sed_names]

                        cluster_catalog = hstack([cluster_catalog_obs, cluster_catalog_sed[unique_sed_names]])
                    elif cluster_class == 'candidates':
                        cluster_catalog = self.open_hst_cluster_cat(target=target, classify=classify,
                                                                    cluster_class=cluster_class, cat_type='obs-sed')
                    else:
                        raise KeyError(cluster_class, ' not understood')
                    self.hst_cc_data.update({str(target) + '_' + classify + '_' + cluster_class: cluster_catalog})

    def check_load_hst_cluster_cat(self, target, classify, cluster_class):
        """
        check if catalog was loaded
        Parameters
        ----------
        target : str
        classify : str
        cluster_class : str
        """
        if not (str(target) + '_' + classify + '_' + cluster_class) in self.hst_cc_data.keys():
            self.load_hst_cluster_cat(target_list=[target], classify_list=[classify],
                                      cluster_class_list=[cluster_class])

    def get_hst_cc_phangs_candidate_id(self, target, classify='human', cluster_class='class12'):
        """
        candidate ID, can be used to connect cross identify with the initial candidate sample
        """
        self.check_load_hst_cluster_cat(target=target, classify=classify, cluster_class=cluster_class)
        return np.array(self.hst_cc_data[str(target) + '_' + classify + '_' + cluster_class]['ID_PHANGS_CANDIDATE'])

    def get_hst_cc_phangs_cluster_id(self, target, classify='human', cluster_class='class12'):
        """
        Phangs ID
        """
        self.check_load_hst_cluster_cat(target=target, classify=classify, cluster_class=cluster_class)
        return np.array(self.hst_cc_data[str(target) + '_' + classify + '_' + cluster_class]['ID_PHANGS_CLUSTER'])

    def get_hst_cc_index(self, target, classify='human', cluster_class='class12'):
        """
        running index for each individual catalog
        """
        self.check_load_hst_cluster_cat(target=target, classify=classify, cluster_class=cluster_class)
        return np.array(self.hst_cc_data[str(target) + '_' + classify + '_' + cluster_class]['INDEX'])

    def get_hst_cc_coords_pix(self, target, classify='human', cluster_class='class12'):
        """
        cluster X and Y coordinates for the PHANGS HST image products.
        These images are re-drizzled and therefore valid for all bands
        """
        self.check_load_hst_cluster_cat(target=target, classify=classify, cluster_class=cluster_class)
        x = np.array(self.hst_cc_data[str(target) + '_' + classify + '_' + cluster_class]['PHANGS_X'])
        y = np.array(self.hst_cc_data[str(target) + '_' + classify + '_' + cluster_class]['PHANGS_Y'])
        return x, y

    def get_hst_cc_coords_world(self, target, classify='human', cluster_class='class12'):
        """
        cluster coordinates RA and dec [Unit is degree]
        """
        self.check_load_hst_cluster_cat(target=target, classify=classify, cluster_class=cluster_class)
        ra = np.array(self.hst_cc_data[str(target) + '_' + classify + '_' + cluster_class]['PHANGS_RA'])
        dec = np.array(self.hst_cc_data[str(target) + '_' + classify + '_' + cluster_class]['PHANGS_DEC'])
        return ra, dec

    def get_hst_cc_class_human(self, target, classify='human', cluster_class='class12'):
        """
        Human classification
        1 [class 1] single peak, circularly symmetric, with radial profile more extended relative to point source
        2 [class 2] tar cluster – similar to Class 1, but elongated or asymmetric
        3 [class 3] compact stellar association – asymmetric, multiple peaks
        4 and above [class 4] not a star cluster or compact stellar association
        (e.g. image artifacts, background galaxies, individual stars or pairs of stars)
        For a more detailed description of other class numbers see readme of the catalog data release
        https://archive.stsci.edu/hlsps/phangs-cat/dr4/hlsp_phangs-cat_hst_multi_all_multi_v1_readme.txt
        """
        self.check_load_hst_cluster_cat(target=target, classify=classify, cluster_class=cluster_class)
        return np.array(self.hst_cc_data[str(target) + '_' + classify + '_' + cluster_class]
                        ['PHANGS_CLUSTER_CLASS_HUMAN'])

    def get_hst_cc_class_ml_vgg(self, target, classify='human', cluster_class='class12'):
        """
        Machine learning classification
        """
        self.check_load_hst_cluster_cat(target=target, classify=classify, cluster_class=cluster_class)
        return np.array(self.hst_cc_data[str(target) + '_' + classify + '_' + cluster_class]
                        ['PHANGS_CLUSTER_CLASS_ML_VGG'])

    def get_hst_cc_class_ml_vgg_qual(self, target, classify='human', cluster_class='class12'):
        """
        Estimated accuracy of machine learning classification
        """
        self.check_load_hst_cluster_cat(target=target, classify=classify, cluster_class=cluster_class)
        return np.array(self.hst_cc_data[str(target) + '_' + classify + '_' + cluster_class]
                        ['PHANGS_CLUSTER_CLASS_ML_VGG_QUAL'])

    def get_hst_ccd_class(self, target, classify='human', cluster_class='class12'):
        """
        Classification based in U-B vs. V-I color-color diagram (See Maschmann et al. 2024) Section 4.4
        """
        self.check_load_hst_cluster_cat(target=target, classify=classify, cluster_class=cluster_class)
        return np.array(self.hst_cc_data[str(target) + '_' + classify + '_' + cluster_class]
                        ['CC_CLASS_%s' % classify.upper()])

    def get_hst_ogc_floyd24_mask(self, target, classify='human', cluster_class='class12'):
        """
        Classification based in U-B vs. V-I color-color diagram (See Maschmann et al. 2024) Section 4.4
        """
        self.check_load_hst_cluster_cat(target=target, classify=classify, cluster_class=cluster_class)
        return np.array(self.hst_cc_data[str(target) + '_' + classify + '_' + cluster_class]
                        ['PHANGS_GLOBULAR_FLOYD24'])

    def get_hst_cc_age(self, target, classify='human', cluster_class='class12'):
        """
        Age see Thilker et al. 2024 [unit is Myr]
        """
        self.check_load_hst_cluster_cat(target=target, classify=classify, cluster_class=cluster_class)
        return np.array(self.hst_cc_data[str(target) + '_' + classify + '_' + cluster_class]['PHANGS_SED_AGE'])

    def get_hst_cc_age_err(self, target, classify='human', cluster_class='class12'):
        """
        Upper and lower uncertainty of Ages [unit is Myr]
        """
        return (np.array(self.hst_cc_data[str(target) + '_' + classify + '_' + cluster_class]['PHANGS_SED_AGE_LIMLO']),
                np.array(self.hst_cc_data[str(target) + '_' + classify + '_' + cluster_class]['PHANGS_SED_AGE_LIMHI']))

    def get_hst_cc_mstar(self, target, classify='human', cluster_class='class12'):
        """
        stellar mass [unit M_sun]
        """
        self.check_load_hst_cluster_cat(target=target, classify=classify, cluster_class=cluster_class)
        return np.array(self.hst_cc_data[str(target) + '_' + classify + '_' + cluster_class]['PHANGS_SED_MASS'])

    def get_hst_cc_mstar_err(self, target, classify='human', cluster_class='class12'):
        """
        Upper and lower uncertainty of stellar mass [unit M_sun]

        """
        return (np.array(self.hst_cc_data[str(target) + '_' + classify + '_' + cluster_class]['PHANGS_SED_MASS_LIMLO']),
                np.array(self.hst_cc_data[str(target) + '_' + classify + '_' + cluster_class]['PHANGS_SED_MASS_LIMHI']))

    def get_hst_cc_ebv(self, target, classify='human', cluster_class='class12'):
        """
        Dust attenuation measured in E(B-V) [unit mag]
        """
        self.check_load_hst_cluster_cat(target=target, classify=classify, cluster_class=cluster_class)
        return np.array(self.hst_cc_data[str(target) + '_' + classify + '_' + cluster_class]['PHANGS_SED_EBV'])

    def get_hst_cc_ebv_err(self, target, classify='human', cluster_class='class12'):
        """
        Upper and lower uncertainty of dust attenuation measured in E(B-V) [unit mag]
        """
        return (np.array(self.hst_cc_data[str(target) + '_' + classify + '_' + cluster_class]['PHANGS_SED_EBV_LIMLO']),
                np.array(self.hst_cc_data[str(target) + '_' + classify + '_' + cluster_class]['PHANGS_SED_EBV_LIMHI']))

    def get_hst_cc_av(self, target, classify='human', cluster_class='class12'):
        """
        Dust attenuation measured in A_v [unit mag]
        """
        ebv_values = self.get_hst_cc_ebv(target=target, classify=classify, cluster_class=cluster_class)
        return dust_tools.DustTools.ebv2av(ebv=ebv_values)

    def get_hst_cc_ir4_age(self, target, classify='human', cluster_class='class12'):
        """
        Old age estimation without decision tree [unit Age]
        """
        self.check_load_hst_cluster_cat(target=target, classify=classify, cluster_class=cluster_class)
        return np.array(self.hst_cc_data[str(target) + '_' + classify + '_' + cluster_class]['PHANGS_AGE_MINCHISQ'])

    def get_hst_cc_ir4_age_err(self, target, classify='human', cluster_class='class12'):
        """
        Uncertainties of old age estimation without decision tree [unit Age]
        """
        self.check_load_hst_cluster_cat(target=target, classify=classify, cluster_class=cluster_class)
        return np.array(self.hst_cc_data[str(target) + '_' + classify + '_' + cluster_class]['PHANGS_AGE_MINCHISQ_ERR'])

    def get_hst_cc_ir4_mstar(self, target, classify='human', cluster_class='class12'):
        """
        Old stellar mass estimation without decision tree [unit M_sun]
        """
        self.check_load_hst_cluster_cat(target=target, classify=classify, cluster_class=cluster_class)
        return np.array(self.hst_cc_data[str(target) + '_' + classify + '_' + cluster_class]['PHANGS_MASS_MINCHISQ'])

    def get_hst_cc_ir4_mstar_err(self, target, classify='human', cluster_class='class12'):
        """
        Uncertainties of old stellar mass estimation without decision tree [unit M_sun]
        """
        self.check_load_hst_cluster_cat(target=target, classify=classify, cluster_class=cluster_class)
        return np.array(self.hst_cc_data[str(target) + '_' + classify + '_' + cluster_class]
                        ['PHANGS_MASS_MINCHISQ_ERR'])

    def get_hst_cc_ir4_ebv(self, target, classify='human', cluster_class='class12'):
        """
        Old dust attenuation E(B-V) estimation without decision tree [unit mag]
        """
        self.check_load_hst_cluster_cat(target=target, classify=classify, cluster_class=cluster_class)
        return np.array(self.hst_cc_data[str(target) + '_' + classify + '_' + cluster_class]['PHANGS_EBV_MINCHISQ'])

    def get_hst_cc_ir4_ebv_err(self, target, classify='human', cluster_class='class12'):
        """
        Uncertainties of old dust attenuation E(B-V) estimation without decision tree [unit mag]
        """
        self.check_load_hst_cluster_cat(target=target, classify=classify, cluster_class=cluster_class)
        return np.array(self.hst_cc_data[str(target) + '_' + classify + '_' + cluster_class]['PHANGS_EBV_MINCHISQ_ERR'])

    def get_hst_cc_ir4_av(self, target, classify='human', cluster_class='class12'):
        """
        Dust attenuation measured in A_v estimation without decision tree [unit mag]
        """
        ebv_values = self.get_hst_cc_ir4_ebv(target=target, classify=classify, cluster_class=cluster_class)
        return dust_tools.DustTools.ebv2av(ebv=ebv_values)

    def get_hst_cc_ci(self, target, classify='human', cluster_class='class12'):
        """
        V-band concentration index, difference in magnitudes measured in 1 pix and 3 pix radii apertures.
        """
        self.check_load_hst_cluster_cat(target=target, classify=classify, cluster_class=cluster_class)
        return np.array(self.hst_cc_data[str(target) + '_' + classify + '_' + cluster_class]['PHANGS_CI'])

    def get_hst_cc_ir4_min_chi2(self, target, classify='human', cluster_class='class12'):
        """
        Old minimal reduced chi-square
        """
        self.check_load_hst_cluster_cat(target=target, classify=classify, cluster_class=cluster_class)
        return np.array(self.hst_cc_data[str(target) + '_' + classify + '_' + cluster_class]['PHANGS_REDUCED_MINCHISQ'])

    def get_hst_cc_cov_flag(self, target, classify='human', cluster_class='class12'):
        """
        Integer denoting the number of bands with no coverage for object
        """
        self.check_load_hst_cluster_cat(target=target, classify=classify, cluster_class=cluster_class)
        return np.array(self.hst_cc_data[str(target) + '_' + classify + '_' + cluster_class]['PHANGS_NO_COVERAGE_FLAG'])

    def get_hst_cc_non_det_flag(self, target, classify='human', cluster_class='class12'):
        """
        Integer denoting the number of bands in which the photometry for the object was below the requested
        signal-to-noise ratio (S/N=1). 0 indicates all five bands had detections. A value of 1 and 2 means the object
         was detected in four and three bands, respectively. By design, this flag cannot be higher than 2.
        """
        self.check_load_hst_cluster_cat(target=target, classify=classify, cluster_class=cluster_class)
        return np.array(self.hst_cc_data[str(target) + '_' + classify + '_' + cluster_class]
                        ['PHANGS_NON_DETECTION_FLAG'])

    def get_hst_cc_band_flux(self, target, filter_name, classify='human', cluster_class='class12'):
        """
        Flux in a specific band [unit is mJy]
        """
        self.check_load_hst_cluster_cat(target=target, classify=classify, cluster_class=cluster_class)
        band = helper_func.ObsTools.filter_name2hst_band(target=target, filter_name=filter_name)
        return np.array(self.hst_cc_data[str(target) + '_' + classify + '_' + cluster_class]
                        ['PHANGS_%s_mJy' % band.upper()])

    def get_hst_cc_band_flux_err(self, target, filter_name, classify='human', cluster_class='class12'):
        """
        Uncertainty of flux in a specific band [unit is mJy]
        """
        self.check_load_hst_cluster_cat(target=target, classify=classify, cluster_class=cluster_class)
        band = helper_func.ObsTools.filter_name2hst_band(target=target, filter_name=filter_name)
        return np.array(self.hst_cc_data[str(target) + '_' + classify + '_' + cluster_class]
                        ['PHANGS_%s_mJy_ERR' % band.upper()])

    def get_hst_cc_band_sn(self, target, filter_name, classify='human', cluster_class='class12'):
        """
        Signa-to-noise in a specific band
        """
        self.check_load_hst_cluster_cat(target=target, classify=classify, cluster_class=cluster_class)
        return (self.get_hst_cc_band_flux(target=target, filter_name=filter_name, classify=classify,
                                          cluster_class=cluster_class) /
                self.get_hst_cc_band_flux_err(target=target, filter_name=filter_name, classify=classify,
                                              cluster_class=cluster_class))

    def get_hst_cc_band_detect_mask(self, target, filter_name, sn=3, classify='human', cluster_class='class12'):
        """
        get boolean mask for objects detected at a signal-to-noise ratio (sn)
        """
        self.check_load_hst_cluster_cat(target=target, classify=classify, cluster_class=cluster_class)
        return np.array(self.get_hst_cc_band_sn(target=target, filter_name=filter_name, classify=classify,
                                                cluster_class=cluster_class) > sn)

    def get_hst_cc_band_vega_mag(self, target, filter_name, classify='human', cluster_class='class12'):
        """
        magnitude [unit is Vega mag]
        """
        self.check_load_hst_cluster_cat(target=target, classify=classify, cluster_class=cluster_class)
        band = helper_func.ObsTools.filter_name2hst_band(target=target, filter_name=filter_name)
        return np.array(self.hst_cc_data[str(target) + '_' + classify + '_' + cluster_class]
                        ['PHANGS_%s_VEGA' % band.upper()])

    def get_hst_cc_band_vega_mag_err(self, target, filter_name, classify='human', cluster_class='class12'):
        """
        Uncertainty of magnitude. Since there is only a specific offset between AB and Vega magnitude systems,
        this is also valid for AB magnitudes [unit is mag]
        """
        self.check_load_hst_cluster_cat(target=target, classify=classify, cluster_class=cluster_class)
        band = helper_func.ObsTools.filter_name2hst_band(target=target, filter_name=filter_name)
        return np.array(self.hst_cc_data[str(target) + '_' + classify + '_' + cluster_class]
                        ['PHANGS_%s_VEGA_ERR' % band.upper()])

    def get_hst_cc_band_ab_mag(self, target, filter_name, classify='human', cluster_class='class12'):
        """
        magnitude [unit is AB mag]
        """
        self.check_load_hst_cluster_cat(target=target, classify=classify, cluster_class=cluster_class)
        flux = self.get_hst_cc_band_flux(target=target, filter_name=filter_name, classify=classify,
                                         cluster_class=cluster_class)
        return helper_func.UnitTools.conv_mjy2ab_mag(flux=flux)

    def get_hst_cc_color(self, target, filter_name_1, filter_name_2, mag_sys='vega', classify='human',
                         cluster_class='class12'):
        """
        get magnitude difference between two bands also called color.

        Parameters
        ----------
        target : str
        filter_name_1: str
        filter_name_2: str
        mag_sys: str
        classify: str
        cluster_class: str
        """
        self.check_load_hst_cluster_cat(target=target, classify=classify, cluster_class=cluster_class)
        assert (filter_name_1 in ['NUV', 'U', 'B', 'V', 'I']) & (filter_name_2 in ['NUV', 'U', 'B', 'V', 'I'])
        assert mag_sys in ['vega', 'ab']
        band_mag_1 = (getattr(self, 'get_hst_cc_band_%s_mag' % mag_sys)
                      (target=target, filter_name=filter_name_1, classify=classify, cluster_class=cluster_class))
        band_mag_2 = (getattr(self, 'get_hst_cc_band_%s_mag' % mag_sys)
                      (target=target, filter_name=filter_name_2, classify=classify, cluster_class=cluster_class))
        return band_mag_1 - band_mag_2

    def get_hst_cc_color_err(self, target, filter_name_1, filter_name_2, classify='human', cluster_class='class12'):
        """
        Uncertainty of color.

        Parameters
        ----------
        target : str
        filter_name_1: str
        filter_name_2: str
        classify: str
        cluster_class: str
        """
        self.check_load_hst_cluster_cat(target=target, classify=classify, cluster_class=cluster_class)
        assert (filter_name_1 in ['NUV', 'U', 'B', 'V', 'I']) & (filter_name_2 in ['NUV', 'U', 'B', 'V', 'I'])
        band_mag_err_1 = (self.get_hst_cc_band_vega_mag_err
                          (target=target, filter_name=filter_name_1, classify=classify, cluster_class=cluster_class))
        band_mag_err_2 = (self.get_hst_cc_band_vega_mag_err
                          (target=target, filter_name=filter_name_1, classify=classify, cluster_class=cluster_class))

        return np.sqrt(band_mag_err_1 ** 2 + band_mag_err_2 ** 2)

    def get_quick_access(self, target='all', classify='human', cluster_class='class123',
                         save_quick_access=True, reload=False, nircam_data_ver='v1p1p1',
                         miri_data_ver='v1p1p1', astrosat_data_ver='v1p0'):
        """
        Function to quickly access catalog data. The loaded data are stored locally in a dictionary and one can
        directly access them via key-words

        Parameters
        ----------
        target : str
        classify: str
        cluster_class: str
        save_quick_access: bool
            if you want to save the dictionary for easier access next time you are loading this.
            However, for this the keyword phangs_hst_cluster_cat_quick_access_path must be specified in the
            phangs_config_dict
        reload: bool
            In case you want to reload this dictionary for example due to an update
        """
        quick_access_dict_path = \
            (Path(phangs_access_config.phangs_config_dict['phangs_hst_cluster_cat_quick_access_path']) /
             ('quick_access_dict_%s_%s_%s_%s_%s.npy' %
              (self.phangs_hst_cluster_cat_sed_version, self.phangs_hst_cluster_cat_ver, target, classify, cluster_class)))

        if os.path.isfile(quick_access_dict_path) and not reload:
            return np.load(quick_access_dict_path, allow_pickle=True).item()
        else:
            phangs_cluster_id = np.array([])
            phangs_candidate_id = np.array([])
            index = np.array([])
            target_name = np.array([], dtype=str)
            ra = np.array([])
            dec = np.array([])
            x = np.array([])
            y = np.array([])
            cluster_class_hum = np.array([])
            cluster_class_ml = np.array([])
            cluster_class_ml_qual = np.array([])
            ci = np.array([])
            ogc_floyd24_mask = np.array([])


            mask_hst_broad_band_covered = np.array([], dtype=bool)
            mask_hst_ha_covered = np.array([], dtype=bool)
            mask_nircam_covered = np.array([], dtype=bool)
            mask_miri_covered = np.array([], dtype=bool)
            mask_astrosat_covered = np.array([], dtype=bool)
            mask_muse_covered = np.array([], dtype=bool)
            mask_alma_covered = np.array([], dtype=bool)

            color_vi_vega = np.array([])
            color_nuvu_vega = np.array([])
            color_ub_vega = np.array([])
            color_bv_vega = np.array([])

            color_vi_ab = np.array([])
            color_nuvu_ab = np.array([])
            color_ub_ab = np.array([])
            color_bv_ab = np.array([])

            color_vi_err = np.array([])
            color_nuvu_err = np.array([])
            color_ub_err = np.array([])
            color_bv_err = np.array([])

            detect_nuv = np.array([], dtype=bool)
            detect_u = np.array([], dtype=bool)
            detect_b = np.array([], dtype=bool)
            detect_v = np.array([], dtype=bool)
            detect_i = np.array([], dtype=bool)

            v_mag_vega = np.array([])
            abs_v_mag_vega = np.array([])
            v_mag_ab = np.array([])
            abs_v_mag_ab = np.array([])

            ccd_class = np.array([])
            age = np.array([])
            mstar = np.array([])
            ebv = np.array([])

            age_limlo = np.array([])
            mstar_limlo = np.array([])
            ebv_limlo = np.array([])

            age_limhi = np.array([])
            mstar_limhi = np.array([])
            ebv_limhi = np.array([])

            if target == 'all':
                if self.phangs_hst_cluster_cat_sed_version == 'ground_based_ha':
                    target_list = phangs_info.hst_cluster_cat_target_list
                elif self.phangs_hst_cluster_cat_sed_version == 'hst_ha':
                    target_list = phangs_info.hst_cluster_cat_hst_ha_target_list
                else:
                    raise KeyError('The catalog version is not understood. '
                                   'especially in order to select which target list?')
            else:
                target_list = [target]

            # add which cluster classes need to be added
            cluster_class_list = []
            if '12' in cluster_class:
                cluster_class_list.append('class12')
            if '3' in cluster_class:
                cluster_class_list.append('class3')

            for target in target_list:

                for cluster_class in cluster_class_list:
                    # # make sure that data is loaded
                    # self.load_hst_cluster_cat(target_list=target_list)
                    # make sure that data is loaded
                    self.load_hst_cluster_cat(target_list=target_list, classify_list=[classify],
                                              cluster_class_list=[cluster_class])

                    phangs_cluster_id = np.concatenate([phangs_cluster_id,
                                                        self.get_hst_cc_phangs_cluster_id(target=target,
                                                                                          cluster_class=cluster_class,
                                                                                          classify=classify)])
                    phangs_candidate_id = np.concatenate([
                        phangs_candidate_id, self.get_hst_cc_phangs_candidate_id(target=target,
                                                                                 cluster_class=cluster_class,
                                                                                 classify=classify)])
                    index = np.concatenate([index, self.get_hst_cc_index(target=target, cluster_class=cluster_class,
                                                                         classify=classify)])
                    ra_, dec_ = self.get_hst_cc_coords_world(target=target, cluster_class=cluster_class,
                                                             classify=classify)
                    ra = np.concatenate([ra, ra_])
                    dec = np.concatenate([dec, dec_])
                    x_, y_ = self.get_hst_cc_coords_pix(target=target, cluster_class=cluster_class, classify=classify)
                    x = np.concatenate([x, x_])
                    y = np.concatenate([y, y_])
                    target_name = np.concatenate([target_name, [target]*len(ra_)])
                    cluster_class_hum = np.concatenate([cluster_class_hum,
                                                        self.get_hst_cc_class_human(target=target,
                                                                                    cluster_class=cluster_class,
                                                                                    classify=classify)])
                    cluster_class_ml = np.concatenate([cluster_class_ml,
                                                       self.get_hst_cc_class_ml_vgg(target=target,
                                                                                    cluster_class=cluster_class,
                                                                                    classify=classify)])
                    cluster_class_ml_qual =\
                        np.concatenate([cluster_class_ml_qual,
                                        self.get_hst_cc_class_ml_vgg_qual(target=target, cluster_class=cluster_class,
                                                                          classify=classify)])
                    ci = np.concatenate([ci, self.get_hst_cc_ci(target=target, cluster_class=cluster_class,
                                                                classify=classify)])
                    ogc_floyd24_mask = np.concatenate([ogc_floyd24_mask, self.get_hst_ogc_floyd24_mask(target=target, cluster_class=cluster_class,
                                                                classify=classify)])

                    # coverage mask
                    # load hst broad band obs.
                    if target == 'ngc1510': galaxy_name = 'ngc1512'
                    else: galaxy_name = target

                    phangs_phot = PhotAccess(phot_hst_target_name=galaxy_name, phot_hst_ha_cont_sub_target_name=galaxy_name,
                        phot_nircam_target_name=helper_func.FileTools.target_name_no_directions(target=galaxy_name),
                        phot_miri_target_name=helper_func.FileTools.target_name_no_directions(target=galaxy_name),
                        phot_astrosat_target_name=helper_func.FileTools.target_name_no_directions(target=galaxy_name),
                        nircam_data_ver=nircam_data_ver, miri_data_ver=miri_data_ver, astrosat_data_ver=astrosat_data_ver)

                    phangs_gas = GasAccess(
                        gas_target_name=helper_func.FileTools.target_name_no_directions(target=galaxy_name))
                    phangs_spec = SpecAccess(
                        spec_target_name=helper_func.FileTools.target_name_no_directions(target=galaxy_name))
                    mask_hst_broad_band_covered = (
                        np.concatenate([mask_hst_broad_band_covered,
                                        phangs_phot.check_coords_covered_by_telescope(
                                            telescope='hst', ra=ra_, dec=dec_, band_list=
                                            helper_func.ObsTools.get_hst_obs_broad_band_list(
                                                target=
                                                helper_func.FileTools.target_name_no_directions(target=galaxy_name)))]))
                    if helper_func.ObsTools.check_hst_ha_obs(
                            target=galaxy_name):
                        mask_hst_ha_covered = (
                            np.concatenate([mask_hst_ha_covered,
                                            phangs_phot.check_coords_covered_by_telescope(
                                                telescope='hst', ra=ra_, dec=dec_, band_list=
                                                [helper_func.ObsTools.get_hst_ha_band(
                                                    target=galaxy_name)])]))
                    else:
                        mask_hst_ha_covered = np.concatenate([mask_hst_ha_covered, np.zeros(len(ra_), dtype=bool)])

                    if helper_func.ObsTools.check_nircam_obs(
                            target=helper_func.FileTools.target_name_no_directions(target=galaxy_name)):
                        mask_nircam_covered = (
                            np.concatenate([mask_nircam_covered,
                                            phangs_phot.check_coords_covered_by_telescope(
                                                telescope='nircam', ra=ra_, dec=dec_, band_list=
                                                helper_func.ObsTools.get_nircam_obs_band_list(
                                                    target=
                                                    helper_func.FileTools.target_name_no_directions(target=galaxy_name),
                                                version=nircam_data_ver))]))
                    else:
                        mask_nircam_covered = np.concatenate([mask_nircam_covered, np.zeros(len(ra_), dtype=bool)])

                    if helper_func.ObsTools.check_miri_obs(
                            target=helper_func.FileTools.target_name_no_directions(target=galaxy_name)):
                        mask_miri_covered = (
                            np.concatenate([mask_miri_covered,
                                            phangs_phot.check_coords_covered_by_telescope(
                                                telescope='miri', ra=ra_, dec=dec_, band_list=
                                                helper_func.ObsTools.get_miri_obs_band_list(
                                                    target=
                                                    helper_func.FileTools.target_name_no_directions(target=
                                                                                                    galaxy_name),
                                                version=miri_data_ver))]))
                    else:
                        mask_miri_covered = np.concatenate([mask_miri_covered, np.zeros(len(ra_), dtype=bool)])

                    if helper_func.ObsTools.check_astrosat_obs(
                            target=helper_func.FileTools.target_name_no_directions(target=galaxy_name)):
                        mask_astrosat_covered = (
                            np.concatenate([mask_astrosat_covered,
                                            phangs_phot.check_coords_covered_by_telescope(
                                                telescope='astrosat', ra=ra_, dec=dec_, band_list=
                                                helper_func.ObsTools.get_astrosat_obs_band_list(
                                                    target=
                                                    helper_func.FileTools.target_name_no_directions(target=
                                                                                                    galaxy_name),
                                                version=astrosat_data_ver))]))
                    else:
                        mask_astrosat_covered = np.concatenate([mask_astrosat_covered, np.zeros(len(ra_), dtype=bool)])

                    if helper_func.ObsTools.check_muse_obs(
                            target=helper_func.FileTools.target_name_no_directions(target=galaxy_name)):
                        mask_muse_covered = (
                            np.concatenate([mask_muse_covered,
                                            phangs_spec.check_coords_covered_by_muse(ra=ra_, dec=dec_)]))
                    else:
                        mask_muse_covered = np.concatenate([mask_muse_covered, np.zeros(len(ra_), dtype=bool)])

                    if helper_func.ObsTools.check_alma_obs(
                            target=helper_func.FileTools.target_name_no_directions(target=galaxy_name)):
                        mask_alma_covered = (
                            np.concatenate([mask_alma_covered,
                                            phangs_gas.check_coords_covered_by_alma(ra=ra_, dec=dec_)]))
                    else:
                        mask_alma_covered = np.concatenate([mask_alma_covered, np.zeros(len(ra_), dtype=bool)])


                    color_vi_vega = np.concatenate([color_vi_vega,
                                                    self.get_hst_cc_color(target=target, cluster_class=cluster_class,
                                                                          filter_name_1='V', filter_name_2='I',
                                                                          classify=classify)])
                    color_nuvu_vega = np.concatenate([color_nuvu_vega,
                                                      self.get_hst_cc_color(target=target, cluster_class=cluster_class,
                                                                            filter_name_1='NUV', filter_name_2='U',
                                                                            classify=classify)])
                    color_ub_vega = np.concatenate([color_ub_vega,
                                                    self.get_hst_cc_color(target=target, cluster_class=cluster_class,
                                                                          filter_name_1='U', filter_name_2='B',
                                                                          classify=classify)])
                    color_bv_vega = np.concatenate([color_bv_vega,
                                                    self.get_hst_cc_color(target=target, cluster_class=cluster_class,
                                                                          filter_name_1='B', filter_name_2='V',
                                                                          classify=classify)])

                    color_vi_ab = np.concatenate([color_vi_ab,
                                                  self.get_hst_cc_color(target=target, cluster_class=cluster_class,
                                                                        filter_name_1='V', filter_name_2='I',
                                                                        mag_sys='ab', classify=classify)])
                    color_nuvu_ab = np.concatenate([color_nuvu_ab,
                                                    self.get_hst_cc_color(target=target, cluster_class=cluster_class,
                                                                          filter_name_1='NUV', filter_name_2='U',
                                                                          mag_sys='ab', classify=classify)])
                    color_ub_ab = np.concatenate([color_ub_ab,
                                                  self.get_hst_cc_color(target=target, cluster_class=cluster_class,
                                                                        filter_name_1='U', filter_name_2='B',
                                                                        mag_sys='ab', classify=classify)])
                    color_bv_ab = np.concatenate([color_bv_ab,
                                                  self.get_hst_cc_color(target=target, cluster_class=cluster_class,
                                                                        filter_name_1='B', filter_name_2='V',
                                                                        mag_sys='ab', classify=classify)])

                    color_vi_err = np.concatenate([color_vi_err,
                                                   self.get_hst_cc_color_err(target=target, cluster_class=cluster_class,
                                                                             filter_name_1='V', filter_name_2='I',
                                                                             classify=classify)])
                    color_nuvu_err = np.concatenate([color_nuvu_err,
                                                     self.get_hst_cc_color_err(target=target,
                                                                               cluster_class=cluster_class,
                                                                               filter_name_1='NUV', filter_name_2='U',
                                                                               classify=classify)])
                    color_ub_err = np.concatenate([color_ub_err,
                                                   self.get_hst_cc_color_err(target=target, cluster_class=cluster_class,
                                                                             filter_name_1='U', filter_name_2='B',
                                                                             classify=classify)])
                    color_bv_err = np.concatenate([color_bv_err,
                                                   self.get_hst_cc_color_err(target=target, cluster_class=cluster_class,
                                                                             filter_name_1='B', filter_name_2='V',
                                                                             classify=classify)])

                    detect_nuv = np.concatenate([detect_nuv,
                                                 self.get_hst_cc_band_detect_mask(target=target, filter_name='NUV',
                                                                                  cluster_class=cluster_class,
                                                                                  classify=classify)])
                    detect_u = np.concatenate([detect_u,
                                               self.get_hst_cc_band_detect_mask(target=target, filter_name='U',
                                                                                cluster_class=cluster_class,
                                                                                classify=classify)])
                    detect_b = np.concatenate([detect_b,
                                               self.get_hst_cc_band_detect_mask(target=target, filter_name='B',
                                                                                cluster_class=cluster_class,
                                                                                classify=classify)])
                    detect_v = np.concatenate([detect_v,
                                               self.get_hst_cc_band_detect_mask(target=target, filter_name='V',
                                                                                cluster_class=cluster_class,
                                                                                classify=classify)])
                    detect_i = np.concatenate([detect_i,
                                               self.get_hst_cc_band_detect_mask(target=target, filter_name='I',
                                                                                cluster_class=cluster_class,
                                                                                classify=classify)])

                    # get V-band magnitude and absolute magnitude
                    v_mag_vega_ = self.get_hst_cc_band_vega_mag(target=target, filter_name='V',
                                                                cluster_class=cluster_class, classify=classify)
                    v_mag_ab_ = self.get_hst_cc_band_ab_mag(target=target, filter_name='V', cluster_class=cluster_class,
                                                            classify=classify)

                    v_mag_vega = np.concatenate([v_mag_vega, v_mag_vega_])
                    v_mag_ab = np.concatenate([v_mag_ab, v_mag_ab_])
                    # get distance
                    sample_access = SampleAccess()
                    target_dist = sample_access.get_target_dist(
                        target=helper_func.FileTools.get_sample_table_target_name(target=target))
                    abs_v_mag_vega = np.concatenate([abs_v_mag_vega,
                                                     helper_func.UnitTools.conv_mag2abs_mag(mag=v_mag_vega_,
                                                                                            dist=target_dist)])
                    abs_v_mag_ab = np.concatenate([abs_v_mag_ab,
                                                   helper_func.UnitTools.conv_mag2abs_mag(mag=v_mag_ab_,
                                                                                          dist=target_dist)])

                    ccd_class = np.concatenate([ccd_class, self.get_hst_ccd_class(target=target,
                                                                                  cluster_class=cluster_class,
                                                                                  classify=classify)])

                    age = np.concatenate([age, self.get_hst_cc_age(target=target, cluster_class=cluster_class,
                                                                   classify=classify)])
                    mstar = np.concatenate([mstar, self.get_hst_cc_mstar(target=target, cluster_class=cluster_class,
                                                                         classify=classify)])
                    ebv = np.concatenate([ebv, self.get_hst_cc_ebv(target=target, cluster_class=cluster_class,
                                                                   classify=classify)])

                    age_limlo_, age_limhi_ = self.get_hst_cc_age_err(target=target, cluster_class=cluster_class,
                                                                     classify=classify)
                    age_limlo = np.concatenate([age_limlo, age_limlo_])
                    age_limhi = np.concatenate([age_limhi, age_limhi_])

                    mstar_limlo_, mstar_limhi_ = self.get_hst_cc_mstar_err(target=target, cluster_class=cluster_class,
                                                                     classify=classify)
                    mstar_limlo = np.concatenate([mstar_limlo, mstar_limlo_])
                    mstar_limhi = np.concatenate([mstar_limhi, mstar_limhi_])

                    ebv_limlo_, ebv_limhi_ = self.get_hst_cc_ebv_err(target=target, cluster_class=cluster_class,
                                                                     classify=classify)
                    ebv_limlo = np.concatenate([ebv_limlo, ebv_limlo_])
                    ebv_limhi = np.concatenate([ebv_limhi, ebv_limhi_])

            quick_access_dict = {
                'phangs_cluster_id': phangs_cluster_id,
                'phangs_candidate_id': phangs_candidate_id,
                'index': index,
                'target_name': target_name,
                'ra': ra,
                'dec': dec,
                'x': x,
                'y': y,
                'cluster_class_hum': cluster_class_hum,
                'cluster_class_ml': cluster_class_ml,
                'cluster_class_ml_qual': cluster_class_ml_qual,
                'ci': ci,
                'ogc_floyd24_mask': ogc_floyd24_mask,
                'mask_hst_broad_band_covered': mask_hst_broad_band_covered,
                'mask_hst_ha_covered': mask_hst_ha_covered,
                'mask_nircam_covered': mask_nircam_covered,
                'mask_miri_covered': mask_miri_covered,
                'mask_astrosat_covered': mask_astrosat_covered,
                'mask_muse_covered': mask_muse_covered,
                'mask_alma_covered': mask_alma_covered,
                'color_vi_vega': color_vi_vega,
                'color_nuvu_vega': color_nuvu_vega,
                'color_ub_vega': color_ub_vega,
                'color_bv_vega': color_bv_vega,
                'color_vi_ab': color_vi_ab,
                'color_nuvu_ab': color_nuvu_ab,
                'color_ub_ab': color_ub_ab,
                'color_bv_ab': color_bv_ab,
                'color_vi_err': color_vi_err,
                'color_nuvu_err': color_nuvu_err,
                'color_ub_err': color_ub_err,
                'color_bv_err': color_bv_err,
                'detect_nuv': detect_nuv,
                'detect_u': detect_u,
                'detect_b': detect_b,
                'detect_v': detect_v,
                'detect_i': detect_i,
                'v_mag_vega': v_mag_vega,
                'abs_v_mag_vega': abs_v_mag_vega,
                'v_mag_ab': v_mag_ab,
                'abs_v_mag_ab': abs_v_mag_ab,
                'ccd_class': ccd_class,
                'age': age,
                'mstar': mstar,
                'ebv': ebv,
                'age_limlo': age_limlo,
                'age_limhi': age_limhi,
                'mstar_limlo': mstar_limlo,
                'mstar_limhi': mstar_limhi,
                'ebv_limlo': ebv_limlo,
                'ebv_limhi': ebv_limhi
            }

            if save_quick_access:
                np.save(quick_access_dict_path, quick_access_dict)
            return quick_access_dict

    def get_hst_cc_cross_match_mask(self, target, ra, dec, classify='human', cluster_class='class12',
                                    toleance_arcsec=0.04, allow_multiple_id=False):
        ra_hst_cc ,dec_hst_cc = self.get_hst_cc_coords_world(target=target, classify=classify,
                                                             cluster_class=cluster_class)
        separation = helper_func.CoordTools.calc_coord_separation(ra_1=ra_hst_cc, dec_1=dec_hst_cc, ra_2=ra, dec_2=dec)
        cross_match_map = separation < toleance_arcsec*u.arcsec
        if allow_multiple_id & (sum(cross_match_map) > 1):
            raise RuntimeError('Multiple obejcts were identified!')
        return cross_match_map

    def load_ccd_hull(self, ccd_type='ubvi', cluster_region='ycl', classify='human'):
        """
        Loading the color-color diagram hulls published in Maschmann+2024

        Parameters
        ----------
        ccd_type : str
        cluster_region: str
        classify: str
        """

        hull = np.genfromtxt(Path(phangs_access_config.phangs_config_dict['phangs_hst_cluster_cat_data_path']) /
                             self.phangs_hst_cluster_cat_sed_version / 'hull' /
                             ('hlsp_phangs-cat_hst_multi_hull_multi_%s_%s-%s-%s.txt' %
                              (self.phangs_hst_cluster_cat_ver, cluster_region, classify, ccd_type)))
        x_color_hull = hull[:, 0]
        y_color_hull = hull[:, 1]
        return x_color_hull, y_color_hull

    @staticmethod
    def points_in_hull(p, hull, tol=1e-12):
        """
        Parameters
        ----------
        p : vector
        hull: array-like
        tol: float
        """

        return np.all(hull.equations[:, :-1] @ p.T + np.repeat(hull.equations[:, -1][None, :], len(p), axis=0).T <= tol,
                      0)

    def load_extend_phot_table(self, target):
        """
        Loading extended photometric catalog

        Parameters
        ----------
        target : str
        """
        # check if already loaded
        if ('extend_phot_table_data_%s' % target) in self.extend_data.keys():
            return None

        file_path = (Path(phangs_access_config.phangs_config_dict['phangs_hst_cluster_cat_extend_photo_path']) /
                     ('%s_IR4' % phangs_access_config.phangs_config_dict['phangs_hst_cluster_cat_extend_photo_ver']))

        file_name = ('Phot_%s_%s_IR4_class12human.fits' %
                     (phangs_access_config.phangs_config_dict['phangs_hst_cluster_cat_extend_photo_ver'],
                      helper_func.FileTools.target_names_no_zeros(target=target)))

        table_data, table_header = helper_func.FileTools.load_fits_table(file_name=file_path / file_name, hdu_number=1)

        self.extend_data.update({
            'extend_phot_table_data_%s' % target: table_data,
            'extend_phot_table_header_%s' % target: table_header
        })

        # ra_photo_data = photo_data_table['raj2000']
        # dec_photo_data = photo_data_table['dej2000']
        # some_id_photo_data = photo_data_table['ID_phangs']

    def load_extend_sed_table(self, target):
        """
        Loading extended sed fit catalog

        Parameters
        ----------
        target : str
        """
        # check if already loaded
        if ('extend_sed_table_data_%s' % target) in self.extend_data.keys():
            return None

        file_path = Path(phangs_access_config.phangs_config_dict['phangs_hst_cluster_cat_extend_sed_fit_path'])
        file_name = '%s_%s_all_clusters_results.csv' % (helper_func.FileTools.target_names_no_zeros(target=target),
                                                        self.phangs_hst_cluster_cat_extend_sed_ver)

        print(file_path / file_name)
        table_data = helper_func.FileTools.load_ascii_table(file_name=file_path / file_name)

        self.extend_data.update({
            'extend_sed_table_data_%s' % target: table_data,
        })

    def get_extend_phot_cross_match_mask(self, target, ra, dec, toleance_arcsec=0.05, allow_multiple_id=False):
        ra_ext_phot ,dec_ext_phot = self.get_extend_phot_coords(target=target)
        separation = helper_func.CoordTools.calc_coord_separation(ra_1=ra_ext_phot, dec_1=dec_ext_phot, ra_2=ra, dec_2=dec)
        cross_match_map = separation < toleance_arcsec*u.arcsec
        if allow_multiple_id & (sum(cross_match_map) > 1):
            raise RuntimeError('Multiple obejcts were identified!')
        return cross_match_map


    def get_extend_phot_candidate_id(self, target):
        """
        access extended photometry candidate id

        Parameters
        ----------
        target : str
        """
        # check if loaded
        self.load_extend_phot_table(target=target)
        return np.array(self.extend_data['extend_phot_table_data_%s' % target]['ID_phangs'])

    def get_extend_phot_coords(self, target):
        """
        access extended photometry coordinates

        Parameters
        ----------
        target : str
        """
        # check if loaded
        self.load_extend_phot_table(target=target)
        return (np.array(self.extend_data['extend_phot_table_data_%s' % target]['raj2000']),
                np.array(self.extend_data['extend_phot_table_data_%s' % target]['dej2000']))

    def get_extend_phot_band_vega_mag(self, target, band):
        """
        access extended photometry Vega magnitude

        Parameters
        ----------
        target : str
        band : str
        """
        # check if loaded
        self.load_extend_phot_table(target=target)
        return np.array(self.extend_data['extend_phot_table_data_%s' % target]['%s_veg' % band])

    def get_extend_phot_band_flux(self, target, band):
        """
        access extended photometry flux

        Parameters
        ----------
        target : str
        band : str
        """
        # check if loaded
        self.load_extend_phot_table(target=target)
        return np.array(self.extend_data['extend_phot_table_data_%s' % target]['flux_%s' % band])

    def get_extend_phot_band_flux_err(self, target, band):
        """
        access extended photometry flux uncertainty

        Parameters
        ----------
        target : str
        band : str
        """
        # check if loaded
        self.load_extend_phot_table(target=target)
        return np.array(self.extend_data['extend_phot_table_data_%s' % target]['er_flux_%s' % band])

    def get_extend_phot_band_ab_mag(self, target, band):
        """
        access extended photometry AB magnitude

        Parameters
        ----------
        target : str
        band : str
        """
        # check if loaded
        self.load_extend_phot_table(target=target)
        return np.array(self.extend_data['extend_phot_table_data_%s' % target]['%s_ab' % band])

    def get_extend_phot_band_mag_err(self, target, band):
        """
        access extended photometry magnitude uncertainty

        Parameters
        ----------
        target : str
        band : str
        """
        # check if loaded
        self.load_extend_phot_table(target=target)
        return np.array(self.extend_data['extend_phot_table_data_%s' % target]['err_%s' % band])

    def get_extend_sed_candidate_id(self, target):
        """
        access extended sed fit id

        Parameters
        ----------
        target : str
        """
        # check if loaded
        self.load_extend_sed_table(target=target)
        return np.array(self.extend_data['extend_sed_table_data_%s' % target]['id'])

    def get_extend_sed_age(self, target):
        """
        access extended sed fit age

        Parameters
        ----------
        target : str
        """
        # check if loaded
        self.load_extend_sed_table(target=target)
        return np.array(self.extend_data['extend_sed_table_data_%s' % target]['best.sfh.age'])

    def get_extend_sed_mstar(self, target):
        """
        access extended sed fit stellar mass

        Parameters
        ----------
        target : str
        """
        # check if loaded
        self.load_extend_sed_table(target=target)
        return np.array(self.extend_data['extend_sed_table_data_%s' % target]['best.stellar.m_star'])

    def get_extend_sed_av(self, target):
        """
        access extended sed fit A_v

        Parameters
        ----------
        target : str
        """
        # check if loaded
        self.load_extend_sed_table(target=target)
        return np.array(self.extend_data['extend_sed_table_data_%s' % target]['best.attenuation.A550'])

    def get_extend_sed_ebv(self, target):
        """
        access extended sed fit E(B-V)

        Parameters
        ----------
        target : str
        """

        # convert to E(B-V)
        return dust_tools.DustTools.av2ebv(av=self.get_extend_sed_av(target))

    def identify_phangs_id_in_ext_phot_table(self, target, single_phangs_cluster_id):

        hst_cc_phangs_candidate_id = self.get_hst_cc_phangs_candidate_id(target=target)
        hst_cc_phangs_cluster_id = self.get_hst_cc_phangs_cluster_id(target=target)
        extend_phot_candidate_id = self.get_extend_phot_candidate_id(target=target)

        mask_select_candidate_id = hst_cc_phangs_cluster_id == single_phangs_cluster_id
        if np.sum(mask_select_candidate_id) == 0:
            raise IndexError(' the PHANGS Cluster ID ', single_phangs_cluster_id, ' is not in the ID list of ', target)

        select_candidate_id = hst_cc_phangs_candidate_id[hst_cc_phangs_cluster_id == single_phangs_cluster_id]

        index_obj = np.where(extend_phot_candidate_id == select_candidate_id)
        return index_obj[0][0]

    def identify_phangs_id_in_ext_sed_table(self, target, single_phangs_cluster_id):

        hst_cc_phangs_candidate_id = self.get_hst_cc_phangs_candidate_id(target=target)
        hst_cc_phangs_cluster_id = self.get_hst_cc_phangs_cluster_id(target=target)
        extend_sed_candidate_id = self.get_extend_sed_candidate_id(target=target)

        mask_select_candidate_id = hst_cc_phangs_cluster_id == single_phangs_cluster_id
        if np.sum(mask_select_candidate_id) == 0:
            raise IndexError(' the PHANGS Cluster ID ', single_phangs_cluster_id, ' is not in the ID list of ', target)

        select_candidate_id = hst_cc_phangs_candidate_id[hst_cc_phangs_cluster_id == single_phangs_cluster_id]

        index_obj = np.where(extend_sed_candidate_id == select_candidate_id)
        print(index_obj)
        return index_obj[0][0]
