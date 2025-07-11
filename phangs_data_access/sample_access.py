"""
Script to access the entire PHANGS galaxy sample and global properties
"""
from pathlib import Path
from astropy.table import Table
import numpy as np
from phangs_data_access import phangs_access_config, helper_func
from astroquery.simbad import Simbad
from astroquery.ipac.ned import Ned

class SampleAccess:
    """
    The PHANGS sample table reflect approximately all galaxies included in the original PHANGS-ALMA paper
    Leeroy+21 (2021ApJS..257...43L)
    Even though all columns from the sample table will be loaded into the constructor only a few are accessible via
    an individual method. This can be updated upon request.
    """

    def __init__(self):
        """

        """

        self.phangs_sample_table_path = phangs_access_config.phangs_config_dict['phangs_sample_table_path']
        self.phangs_sample_table_ver = phangs_access_config.phangs_config_dict['phangs_sample_table_ver']
        self.phangs_sample_table = None

        super().__init__()

    def load_phangs_sample_table(self):
        """

        Function to load Phangs sample table into the constructor.
        Required to access global sample data

        Parameters
        ----------

        Returns
        -------
        None
        """
        path_phangs_sample_table = (self.phangs_sample_table_path + '/' + self.phangs_sample_table_ver +
                                    '/phangs_sample_table_%s.fits' % self.phangs_sample_table_ver)

        self.phangs_sample_table = Table.read(path_phangs_sample_table)

    def check_load_phangs_data_table(self):
        """
        method to quickly check if sample table is loaded and load if not.
        """
        if self.phangs_sample_table is None:
            self.load_phangs_sample_table()

    def get_full_phangs_target_list(self):
        """
        returns all targets associated with PHANGS
        """
        self.check_load_phangs_data_table()
        return self.phangs_sample_table['name']

    def get_target_central_coords(self, target):
        """
        get target central coordinates
        Parameters
        ----------
        target : str
            Galaxy name
        """
        if target == 'ngc1808': return 76.926384, -37.512920
        self.check_load_phangs_data_table()
        mask_target = self.phangs_sample_table['name'] == target
        return (float(self.phangs_sample_table['orient_ra'][mask_target].value[0]),
                float(self.phangs_sample_table['orient_dec'][mask_target].value[0]))

    def get_target_sfr(self, target):
        """
        load SFR for a PHANGS target
        Parameters
        ----------
        target : str
            Galaxy name
        """
        self.check_load_phangs_data_table()
        mask_target = self.phangs_sample_table['name'] == target
        return self.phangs_sample_table['props_sfr'][mask_target].value[0]

    def get_target_sfr_err(self, target):
        """
        load SFR uncertainties
        """
        self.check_load_phangs_data_table()
        mask_target = self.phangs_sample_table['name'] == target
        return self.phangs_sample_table['props_sfr_unc'][mask_target].value[0]

    def get_target_log_sfr(self, target):
        """
        load log SFR
        """
        return np.log10(self.get_target_sfr(target=target))

    def get_target_mstar(self, target):
        """
        load stellar mass
        """
        self.check_load_phangs_data_table()
        mask_target = self.phangs_sample_table['name'] == target
        return self.phangs_sample_table['props_mstar'][mask_target].value[0]

    def get_target_mstar_err(self, target):
        """
        load stellar mass uncertainty
        """
        self.check_load_phangs_data_table()
        mask_target = self.phangs_sample_table['name'] == target
        return self.phangs_sample_table['props_mstar_unc'][mask_target].value[0]

    def get_target_log_mstar(self, target):
        """
        load log stellar mass
        """
        return np.log10(self.get_target_mstar(target=target))

    def get_target_ssfr(self, target):
        """
        load specific star-formation rate
        """
        sfr = self.get_target_sfr(target=target)
        mstar = self.get_target_mstar(target=target)
        return sfr / mstar

    def get_target_ssfr_err(self, target):
        """
        load specific star formation-rate uncertainties
        """
        sfr = self.get_target_sfr(target=target)
        sfr_err = self.get_target_sfr_err(target=target)
        mstar = self.get_target_mstar(target=target)
        mstar_err = self.get_target_mstar_err(target=target)
        # first degree error propagation
        return np.sqrt((sfr_err / mstar) ** 2 + (sfr * mstar_err / (mstar ** 2)) ** 2)

    def get_target_log_ssfr(self, target):
        """
        load log10 specific star-formation rate
        """
        return self.get_target_log_sfr(target=target) - self.get_target_log_mstar(target=target)

    def get_target_delta_ms(self, target):
        """
        load distance to main sequence of star-forming galaxies in dex
        """
        self.check_load_phangs_data_table()
        mask_target = self.phangs_sample_table['name'] == target
        return self.phangs_sample_table['props_deltams'][mask_target].value[0]

    def get_target_delta_ms_err(self, target):
        """
        load delta MS uncertainty in dex
        """
        self.check_load_phangs_data_table()
        mask_target = self.phangs_sample_table['name'] == target
        return self.phangs_sample_table['props_deltams_unc'][mask_target].value[0]

    def get_target_dist(self, target):
        """
        load target distance in Mpc
        """
        self.check_load_phangs_data_table()
        mask_target = self.phangs_sample_table['name'] == target
        if (sum(mask_target) == 0) & (target=='ngc1808'):
            # distance estimated from Tully fisher 2017A&A...598A..55B and 2009AJ....138..323T
            return 12.8

        return self.phangs_sample_table['dist'][mask_target].value[0]

    def get_target_dist_err(self, target):
        """
        load target distance uncertainty in Mpc
        """
        self.check_load_phangs_data_table()
        mask_target = self.phangs_sample_table['name'] == target
        dist_unc_dex = self.phangs_sample_table['dist_unc'][mask_target].value[0]
        if (sum(mask_target) == 0) & (target=='ngc1808'):
            # distance estimated from Tully fisher 2017A&A...598A..55B and 2009AJ....138..323T
            return 1.2

        return (10 ** (np.log10(self.get_target_dist(target=target)) + dist_unc_dex) -
                self.get_target_dist(target=target))

    def get_target_dist_method(self, target):
        """
        load method used to determine target distance
        """
        self.check_load_phangs_data_table()
        mask_target = self.phangs_sample_table['name'] == target
        if (sum(mask_target) == 0) & (target=='ngc1808'):
            # distance estimated from Tully fisher 2017A&A...598A..55B and 2009AJ....138..323T
            return 'TF'
        return self.phangs_sample_table['dist_label'][mask_target].value[0]

    def get_target_arcsec_re(self, target):
        """
        load the half-mass radius R_e [unit is arcsec]
        """
        self.check_load_phangs_data_table()
        mask_target = self.phangs_sample_table['name'] == target
        return self.phangs_sample_table['size_reff'][mask_target].value[0]

    def get_target_arcsec_re_err(self, target):
        """
        load the half-mass radius uncertainty [unit is arcsec]
        """
        self.check_load_phangs_data_table()
        mask_target = self.phangs_sample_table['name'] == target
        return self.phangs_sample_table['size_reff_unc'][mask_target].value[0]

    def get_target_kpc_re(self, target):
        """
        load the half-mass radius R_e [unit is kpc]
        """
        re_arcsec = self.get_target_arcsec_re(target=target)
        dist = self.get_target_dist(target=target)
        return helper_func.CoordTools.arcsec2kpc(diameter_arcsec=re_arcsec, target_dist_mpc=dist)

    def get_target_kpc_re_err(self, target):
        """
        load the half-mass radius uncertainty [unit is kpc]
        """
        re_arcsec_err = self.get_target_arcsec_re_err(target=target)
        dist = self.get_target_dist(target=target)
        return helper_func.CoordTools.arcsec2kpc(diameter_arcsec=re_arcsec_err, target_dist_mpc=dist)

    def get_target_arcsec_r25(self, target):
        """
        load 25th magnitude isophotal B-band radius [unit is arcsec]
        """
        self.check_load_phangs_data_table()
        mask_target = self.phangs_sample_table['name'] == target
        return self.phangs_sample_table['size_r25'][mask_target].value[0]

    def get_target_arcsec_r25_err(self, target):
        """
        load 25th magnitude isophotal B-band radius uncertainty [unit is arcsec]
        """
        self.check_load_phangs_data_table()
        mask_target = self.phangs_sample_table['name'] == target
        return self.phangs_sample_table['size_r25_unc'][mask_target].value[0]

    def get_target_kpc_r25(self, target):
        """
        load 25th magnitude isophotal B-band radius [unit is kpc]
        """
        r25_arcsec = self.get_target_arcsec_r25(target=target)
        dist = self.get_target_dist(target=target)
        return helper_func.CoordTools.arcsec2kpc(diameter_arcsec=r25_arcsec, target_dist_mpc=dist)

    def get_target_kpc_r25_err(self, target):
        """
        load 25th magnitude isophotal B-band radius uncertainty [unit is kpc]
        """
        r25_arcsec_err = self.get_target_arcsec_r25_err(target=target)
        dist = self.get_target_dist(target=target)
        return helper_func.CoordTools.arcsec2kpc(diameter_arcsec=r25_arcsec_err, target_dist_mpc=dist)

    def get_target_arcsec_r90(self, target):
        """
        load radius containing 90% of the mass [unit is arcsec]
        """
        self.check_load_phangs_data_table()
        mask_target = self.phangs_sample_table['name'] == target
        return self.phangs_sample_table['size_r90'][mask_target].value[0]

    def get_target_arcsec_r90_err(self, target):
        """
        load radius uncertainty containing 90% of the mass [unit is arcsec]
        """
        self.check_load_phangs_data_table()
        mask_target = self.phangs_sample_table['name'] == target
        return self.phangs_sample_table['size_r90_unc'][mask_target].value[0]

    def get_target_kpc_r90(self, target):
        """
        load radius containing 90% of the mass [unit is kpc]
        """
        r90_arcsec = self.get_target_arcsec_r90(target=target)
        dist = self.get_target_dist(target=target)
        return helper_func.CoordTools.arcsec2kpc(diameter_arcsec=r90_arcsec, target_dist_mpc=dist)

    def get_target_kpc_r90_err(self, target):
        """
        load radius containing 90% of the mass [unit is kpc]
        """
        r90_arcsec_err = self.get_target_arcsec_r90_err(target=target)
        dist = self.get_target_dist(target=target)
        return helper_func.CoordTools.arcsec2kpc(diameter_arcsec=r90_arcsec_err, target_dist_mpc=dist)

    def get_target_incl(self, target):
        """"
        load target inclination in degree
        """
        self.check_load_phangs_data_table()
        mask_target = self.phangs_sample_table['name'] == target
        return self.phangs_sample_table['orient_incl'][mask_target].value[0]

    def get_target_incl_err(self, target):
        """"
        load target inclination uncertainty in degree
        """
        self.check_load_phangs_data_table()
        mask_target = self.phangs_sample_table['name'] == target
        return self.phangs_sample_table['orient_incl_unc'][mask_target].value[0]

    def get_target_pos_ang(self, target):
        """"
        load target position angle in degree
        """
        self.check_load_phangs_data_table()
        mask_target = self.phangs_sample_table['name'] == target
        return self.phangs_sample_table['orient_posang'][mask_target].value[0]

    def get_target_pos_ang_err(self, target):
        """"
        load target position angle uncertainty in degree
        """
        self.check_load_phangs_data_table()
        mask_target = self.phangs_sample_table['name'] == target
        return self.phangs_sample_table['orient_posang_unc'][mask_target].value[0]

    def get_target_mw_extinction(self, target):
        """
        This function is meant to eventually provide extinction levels for PHANGS sources.
        there are multiple values reported as
        mwext_sfd98
        mwext_sfd98_unc
        mwext_sf11
        mwext_sf11_unc
        this needs to be done
        """
        raise ModuleNotFoundError('This module is not finished yet. Do not use it. It is only a placeholder')

    def get_target_mhi(self, target):
        """"
        load target HI mass
        """
        self.check_load_phangs_data_table()
        mask_target = self.phangs_sample_table['name'] == target
        return self.phangs_sample_table['props_mhi'][mask_target].value[0]

    def get_target_mhi_err(self, target):
        """"
        load target HI mass uncertainty
        """
        self.check_load_phangs_data_table()
        mask_target = self.phangs_sample_table['name'] == target
        return self.phangs_sample_table['props_mhi_unc'][mask_target].value[0]

    def get_target_mh2(self, target):
        """"
        load target H2 mass in units of M_sun
        """
        self.check_load_phangs_data_table()
        mask_target = self.phangs_sample_table['name'] == target
        return self.phangs_sample_table['mh2_phangs'][mask_target].value[0]

    def get_target_mh2_err(self, target):
        """"
        load target H2 mass uncertainty
        """
        self.check_load_phangs_data_table()
        mask_target = self.phangs_sample_table['name'] == target
        return self.phangs_sample_table['mh2_phangs_unc'][mask_target].value[0]

    def get_target_co_luminosity(self, target):
        """"
        load target CO luminosity
        """
        self.check_load_phangs_data_table()
        mask_target = self.phangs_sample_table['name'] == target
        return self.phangs_sample_table['lco_phangs'][mask_target].value[0]

    def get_target_co_luminosity_err(self, target):
        """"
        load target CO luminosity uncertainty
        """
        self.check_load_phangs_data_table()
        mask_target = self.phangs_sample_table['name'] == target
        return self.phangs_sample_table['lco_phangs_unc'][mask_target].value[0]

    def get_target_co_aperture_corr(self, target):
        """"
        load target aperture correction for CO observations
        """
        self.check_load_phangs_data_table()
        mask_target = self.phangs_sample_table['name'] == target
        return self.phangs_sample_table['appcor_phangs'][mask_target].value[0]

    def get_target_co10_conv_fact(self, target):
        """"
        load target CO10 conversion factor
        """
        self.check_load_phangs_data_table()
        mask_target = self.phangs_sample_table['name'] == target
        return self.phangs_sample_table['aco10_phangs'][mask_target].value[0]

    def get_target_morph_t_type(self, target):
        """"
        load target Hubble type T
        (see Hubble stage at https://en.wikipedia.org/wiki/Galaxy_morphological_classification)
        """
        self.check_load_phangs_data_table()
        mask_target = self.phangs_sample_table['name'] == target
        return self.phangs_sample_table['morph_t'][mask_target].value[0]

    def get_target_morph_t_type_err(self, target):
        """"
        load target Hubble type T uncertainty
        (see Hubble stage at https://en.wikipedia.org/wiki/Galaxy_morphological_classification)
        """
        self.check_load_phangs_data_table()
        mask_target = self.phangs_sample_table['name'] == target
        return self.phangs_sample_table['morph_t_unc'][mask_target].value[0]

    @staticmethod
    def get_hst_obs_zp_mag(target, band, mag_sys='vega'):
        """"
        load zero point magnitude for a HST target
        Parameters
        ----------
        target : str
        band : str
        mag_sys : str
        """
        header_df = helper_func.FileTools.load_ascii_table_from_txt(
            file_name=(Path(phangs_access_config.phangs_config_dict['hst_obs_hdr_file_path']) /
                       ('header_info_%s_prime.txt' % helper_func.FileTools.target_names_no_zeros(target=target))))
        filter_set = np.array(header_df['filter'].to_list())
        mask_filter = filter_set == band

        if mag_sys == 'vega':
            return np.array(header_df['zpVEGA'].to_list())[mask_filter]
        elif mag_sys == 'AB':
            return np.array(header_df['zpAB'].to_list())[mask_filter]
        else:
            raise KeyError('mag_sys must be vega or AB')
    @staticmethod
    def get_hst_obs_date(target, band):
        """"
        load zero point magnitude for a HST target
        Parameters
        ----------
        target : str
        band : str
        """
        header_df = helper_func.FileTools.load_ascii_table_from_txt(
            file_name=(Path(phangs_access_config.phangs_config_dict['hst_obs_hdr_file_path']) /
                       ('header_info_%s_prime.txt' % helper_func.FileTools.target_names_no_zeros(target=target))))
        filter_set = np.array(header_df['filter'].to_list())
        mask_filter = filter_set == band

        return np.array(header_df['date'].to_list())[mask_filter]
