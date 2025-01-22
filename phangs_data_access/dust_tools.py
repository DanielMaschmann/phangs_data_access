import numpy as np
from astroquery.ipac.irsa.irsa_dust import IrsaDust
from astropy.coordinates import SkyCoord
import astropy.units as u

from dust_extinction import parameter_averages

from phangs_data_access import helper_func, sample_access, phys_params


class DustTools:
    """
    All functionalities for dust extinction and attenuation calculations
    """
    def __int__(self):
        pass

    @staticmethod
    def get_coord_gal_ext_evb(ra, dec, rad_deg=None, method='SandF', ebv_estimator='mean'):
        """
        Get the Galactic E(B-V) for a given coordinate
        Parameters
        ----------
        ra, dec: float or list or `numpy.ndarray`
            coordinates in degree
        rad_deg:  `astropy.units.Quantity`
        method: str
            must be SFD or SandF and specifies the reference
            Schlafly, E.F. & Finkbeiner, D.P. 2011, ApJ 737, 103 (SandF).
            Schlegel, D.J., Finkbeiner, D.P. Davis, M. 1998, ApJ 500, 525 (SFD).
        ebv_estimator: str
            must be in ['mean', 'std', 'ref', 'min', 'max']

        Returns
        -------
        gal_ext_ebv: float or list or `numpy.ndarray`
        """
        assert ebv_estimator in ['mean', 'std', 'ref', 'min', 'max']
        assert method in ['SandF', 'SFD']

        target_coords = SkyCoord(ra=ra * u.deg, dec=dec * u.deg)
        gal_ebv_table = IrsaDust.get_query_table(target_coords, section='ebv', radius=rad_deg)
        return gal_ebv_table['ext %s %s' % (method, ebv_estimator)]

    @staticmethod
    def get_target_gal_ext_ebv(target, method='SandF', ebv_estimator='mean', rad_deg=None):
        """
        Function to get Galactic E(B-V) for phangs target
        """
        phangs_sample = sample_access.SampleAccess()
        ra_target, dec_target = phangs_sample.get_target_central_coords(target=target)

        return DustTools.get_coord_gal_ext_evb(ra=ra_target, dec=dec_target, rad_deg=rad_deg, method=method,
                                               ebv_estimator=ebv_estimator)

    @staticmethod
    def get_gal_ext_at_wave(ra, dec, wave_mu, rad_deg=None, method='SandF', ebv_estimator='mean', ext_law='F99', r_v=3.1):
        gal_ext_ebv = DustTools.get_coord_gal_ext_evb(ra=ra, dec=dec, method=method, ebv_estimator=ebv_estimator,
                                                      rad_deg=rad_deg)
        ext_model = getattr(parameter_averages, ext_law)(Rv=r_v)
        return ext_model(wave_mu*u.micron) * r_v * gal_ext_ebv

    @staticmethod
    def get_gal_ext(ra, dec, wave_mu, rad_deg=None, method='SandF', ebv_estimator='mean', ext_law='F99', r_v=3.1):
        gal_ext_ebv = DustTools.get_coord_gal_ext_evb(ra=ra, dec=dec, rad_deg=rad_deg, method=method,
                                                      ebv_estimator=ebv_estimator)

        ext_model = getattr(parameter_averages, ext_law)(Rv=r_v)
        return ext_model(wave_mu) * r_v * gal_ext_ebv

    @staticmethod
    def get_target_gal_ext_band(target, obs, band, rad_deg=None, method='SandF', ebv_estimator='mean', ext_law='F99',
                                r_v=3.1, wave_estimator='pivot_wave'):
        gal_ext_ebv = DustTools.get_target_gal_ext_ebv(
            target=helper_func.FileTools.target_name_no_directions(target=target), method=method,
            ebv_estimator=ebv_estimator, rad_deg=rad_deg)

        # get wavelength
        if obs in ['hst', 'hst_ha']:
            wave = helper_func.ObsTools.get_hst_band_wave(
                band=band, instrument=helper_func.ObsTools.get_hst_instrument(target=target, band=band),
                wave_estimator=wave_estimator, unit='mu') * u.micron
        elif obs in ['nircam', 'miri']  :
            wave = helper_func.ObsTools.get_jwst_band_wave(
                band=band, instrument=obs,
                wave_estimator=wave_estimator, unit='mu') * u.micron
        elif obs == 'astrosat':
            wave = helper_func.ObsTools.get_astrosat_band_wave(band=band, wave_estimator=wave_estimator, unit='mu') * u.micron
        else:
            raise KeyError('obs keywaord not understood!')

        ext_model = getattr(parameter_averages, ext_law)(Rv=r_v)

        if (wave.value < (1/ext_model.x_range[1])) | (wave.value > (1/ext_model.x_range[0])):
            return 0
        else:
            return (ext_model(wave) * r_v * gal_ext_ebv).value[0]

    @staticmethod
    def mag_ext2ebv(mag, wave, ext_law='CCM89', r_v=3.1):
        ext_model = getattr(parameter_averages, ext_law)(Rv=r_v)
        return mag / (ext_model(wave * u.micron) * r_v)






    @staticmethod
    def c00_redd_curve(wavelength=6565, r_v=3.1):
        r"""
        calculate reddening curve
         following  Calzetti et al. (2000) doi:10.1086/308692
         using eq. 4

        :param wavelength: rest frame wavelength in angstrom of spectral part of which to compute the reddening curve
        :type wavelength: float or int
        :param r_v: default 3.1  total extinction at V
        :type r_v: float

        :return extinction E(B - V) in mag
        :rtype: array_like
        """

        # change wavelength from Angstrom to microns
        wavelength *= 1e-4

        # eq. 4
        if (wavelength > 0.63) & (wavelength < 2.20):
            # suitable for 0.63 micron < wavelength < 2.20 micron
            k_lambda = 2.659 * (-1.857 + 1.040/wavelength) + r_v
        elif (wavelength > 0.12) & (wavelength < 0.63):
            # suitable for 0.12 micron < wavelength < 0.63 micron
            k_lambda = 2.659 * (- 2.156 + 1.509 / wavelength - 0.198 / wavelength**2 + 0.011 / wavelength**3) + r_v
        else:
            raise KeyError('wavelength must be > 1200 Angstrom and < 22000 Angstrom')

        return k_lambda

    @staticmethod
    def calc_stellar_extinct(wavelength, ebv, r_v):
        return ExtinctionTools.compute_reddening_curve(wavelength=wavelength, r_v=r_v) * ebv

    @staticmethod
    def color_ext_ccm89_ebv(wave1, wave2, ebv, r_v=3.1):

        model_ccm89 = parameter_averages.CCM89(Rv=r_v)
        reddening1 = model_ccm89(wave1 * u.micron) * r_v
        reddening2 = model_ccm89(wave2 * u.micron) * r_v

        return (reddening1 - reddening2) * ebv

    @staticmethod
    def band_ext_ccm89_ebv(wave, ebv, r_v=3.1):

        model_ccm89 = parameter_averages.CCM89(Rv=r_v)
        reddening = model_ccm89(wave * u.micron) * r_v

        return reddening * ebv

    @staticmethod
    def ebv2av(ebv, r_v=3.1):
        wave_v = 5388.55 * 1e-4
        model_ccm89 = parameter_averages.CCM89(Rv=r_v)
        return model_ccm89(wave_v*u.micron) * r_v * ebv

    @staticmethod
    def av2ebv(av, r_v=3.1):
        wave_v = 5388.55 * 1e-4
        model_ccm89 = parameter_averages.CCM89(Rv=r_v)
        return av / (model_ccm89(wave_v*u.micron) * r_v)

    @staticmethod
    def color_ext_ccm89_av(wave1, wave2, av, r_v=3.1):

        model_ccm89 = parameter_averages.CCM89(Rv=r_v)
        reddening1 = model_ccm89(wave1*u.micron) * r_v
        reddening2 = model_ccm89(wave2*u.micron) * r_v

        wave_v = 5388.55 * 1e-4
        reddening_v = model_ccm89(wave_v*u.micron) * r_v

        return (reddening1 - reddening2)*av/reddening_v

    @staticmethod
    def color_ext_f99_av(wave1, wave2, av, r_v=3.1):

        model_f99 = parameter_averages.F99(Rv=r_v)
        reddening1 = model_f99(wave1*u.micron) * r_v
        reddening2 = model_f99(wave2*u.micron) * r_v

        wave_v = 5388.55 * 1e-4
        reddening_v = model_f99(wave_v*u.micron) * r_v

        return (reddening1 - reddening2)*av/reddening_v

