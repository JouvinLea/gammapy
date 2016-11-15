# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, \
    unicode_literals
import logging
import numpy as np
from astropy.units import Quantity
from astropy.table import QTable, Table
from astropy.coordinates import Angle
from ..utils.energy import EnergyBounds, Energy
from ..stats import significance
from ..background import fill_acceptance_image
from ..image import SkyImage, SkyImageList, disk_correlate
from ..cube import SkyCube
from .exposure import exposure_cube

__all__ = ['SingleObsCubeMaker', 'StackedObsCubeMaker']

log = logging.getLogger(__name__)


class SingleObsCubeMaker(object):
    """Compute '~gammapy.cube.SkyCube' images for one observation.

    The computed images are stored in a ``images`` attribute of
    type `~gammapy.image.SkyImageList` with the following keys:

    * ``counts`` : Counts
    * ``bkg`` : Background model
    * ``exposure`` : Exposure
    * ``excess`` : Excess
    * ``significance`` : Significance

    Parameters
    ----------
    obs : `~gammapy.data.DataStoreObservation`
        Observation data
    empty_cube_images : `~gammapy.cube.SkyCube`
        Reference Cube for images in reco energy
    empty_exposure_cube : `~gammapy.cube.SkyCube`
        Reference Cube for exposure in true energy
    offset_band : `astropy.coordinates.Angle`
        Offset band selection
    exclusion_mask : `~gammapy.image.SkyMask`
        Exclusion mask
    save_bkg_scale: bool
        True if you want to save the normalisation of the bkg computed outside the exlusion region in a Table
    """

    def __init__(self, obs, empty_cube_images, empty_exposure_cube,
                 offset_band, header, exclusion_mask=None, save_bkg_scale=True):
        # Select the events in the given energy and offset range
        self.energy_reco_bins = empty_cube_images.energy_axis.energy
        self.offset_band = offset_band
        self.counts_cube = SkyCube.empty_like(empty_cube_images)
        self.bkg_cube = SkyCube.empty_like(empty_cube_images)
        self.significance_cube = SkyCube.empty_like(empty_cube_images)
        self.excess_cube = SkyCube.empty_like(empty_cube_images)
        self.exposure_cube = SkyCube.empty_like(empty_exposure_cube)

        self.obs_id = obs.obs_id
        events = obs.events
        # events = events.select_energy(self.energy_band)
        self.events = events.select_offset(self.offset_band)

        # self.images = SkyImageList()
        # self.empty_image = empty_image
        self.header = header
        if exclusion_mask:
            self.cube_exclusion_mask = np.tile(exclusion_mask.data,
                                               (len(self.energy_reco_bins) - 1, 1, 1))
        self.aeff = obs.aeff
        self.edisp = obs.edisp
        self.psf = obs.psf
        self.bkg = obs.bkg
        self.obs_center = obs.pointing_radec
        self.livetime = obs.observation_live_time_duration
        self.save_bkg_scale = save_bkg_scale
        if self.save_bkg_scale:
            self.table_bkg_scale = Table(names=["OBS_ID", "bkg_scale"])

    def make_counts_cube(self):
        """Fill the counts image for the events of one observation."""
        self.counts_cube.fill_events(self.events)

    def make_bkg_cube(self, bkg_norm=True):
        """
        Make the background image for one observation from a bkg model.

        Parameters
        ----------
        bkg_norm : bool
            If true, apply the scaling factor from the number of counts
            outside the exclusion region to the bkg image
        """
        for i_E in range(len(self.energy_reco_bins)-1):
            energy_band = Energy(
                [self.energy_reco_bins[i_E].value, self.energy_reco_bins[i_E + 1].value],
                self.energy_reco_bins.unit)
            table = self.bkg.acceptance_curve_in_energy_band(
                energy_band=energy_band)
            center = self.obs_center.galactic
            bkg_hdu = fill_acceptance_image(self.header, center,
                                            table["offset"],
                                            table["Acceptance"],
                                            self.offset_band[1])
            bkg_image = Quantity(bkg_hdu.data, table[
                "Acceptance"].unit) * self.bkg_cube.sky_image_ref.solid_angle() * self.livetime
            self.bkg_cube.data[i_E, :, :] = bkg_image.decompose().value

        if bkg_norm:
            scale = self.background_norm_factor(self.counts_cube,
                                                self.bkg_cube)
            self.bkg_cube.data = scale * self.bkg_cube.data
            if self.save_bkg_scale:
                self.table_bkg_scale.add_row([self.obs_id, scale])

    def background_norm_factor(self, counts, bkg):
        """Determine the scaling factor to apply to the background image.

        Compares the events in the counts images and the bkg image outside the exclusion images.

        Parameters
        ----------
        counts : `~gammapy.cube.SkyCube`
            counts images cube
        bkg : `~gammapy.cube.SkyCube`
            bkg images cube

        Returns
        -------
        scale : float
            scaling factor between the counts and the bkg images outside the exclusion region.
        """
        counts_sum = np.sum(
            self.counts_cube.data * self.cube_exclusion_mask)
        bkg_sum = np.sum(self.bkg_cube.data * self.cube_exclusion_mask)
        scale = counts_sum / bkg_sum

        return scale

    def make_exposure_cube(self):
        """
        Compute the exposure cube
        """
        self.exposure_cube = exposure_cube(pointing=self.obs_center,
                                           livetime=self.livetime,
                                           aeff2d=self.aeff,
                                           ref_cube=self.exposure_cube,
                                           offset_max=self.offset_band[1])

    def make_significance_cube(self, radius):
        """Make the significance image from the counts and bkg images.

        Parameters
        ----------
        radius : float
            Disk radius in pixels.
        """
        for i_E in range(len(self.energy_reco_bins)-1):
            counts = disk_correlate(self.counts_cube[i_E, :, :], radius)
            bkg = disk_correlate(self.bkg_cube[i_E, :, :], radius)
            self.significance_cube.data[i_E, :, :] = significance(counts, bkg)

    def make_excess_cube(self):
        """Compute excess between counts and bkg image."""
        for i_E in range(len(self.energy_reco_bins)-1):
            self.excess_cube.data[i_E, :, :] = self.counts_cube[i_E, :,
                                               :] - self.bkg_cube[i_E, :, :]


class StackedObsCubeMaker(object):
    """Compute stacked images for many observations.

    The computed images are stored in a ``images`` attribute of
    type `~gammapy.image.SkyImageList` with the following keys:

    * ``counts`` : Counts
    * ``bkg`` : Background model
    * ``exposure`` : Exposure
    * ``excess`` : Excess
    * ``significance`` : Significance

    Parameters
    ----------
    empty_cube_images : `~gammapy.cube.SkyCube`
        Reference Cube for images in reco energy
    empty_exposure_cube : `~gammapy.cube.SkyCube`
        Reference Cube for exposure in true energy
    offset_band : `astropy.coordinates.Angle`
        Offset band selection
    data_store : `~gammapy.data.DataStore`
        Data store
    obs_table : `~astropy.table.Table`
        Required columns: OBS_ID
    exclusion_mask : `~gammapy.image.SkyMask`
        Exclusion mask
    save_bkg_scale: bool
        True if you want to save the normalisation of the bkg for each run in a `Table` table_bkg_norm with two columns:
         "OBS_ID" and "bkg_scale"
    """

    def __init__(self, empty_cube_images, header,empty_exposure_cube=None,
                 offset_band=None,
                 data_store=None, obs_table=None, exclusion_mask=None,
                 ncounts_min=0, save_bkg_scale=True):

        self.empty_cube_images = empty_cube_images
        self.energy_reco_bins = empty_cube_images.energy_axis.energy
        if not empty_exposure_cube:
            self.empty_exposure_cube = SkyCube.empty_like(empty_cube_images)
        else:
            self.empty_exposure_cube = empty_exposure_cube

        self.counts_cube = SkyCube.empty_like(empty_cube_images)
        self.bkg_cube = SkyCube.empty_like(empty_cube_images)
        self.significance_cube = SkyCube.empty_like(empty_cube_images)
        self.excess_cube = SkyCube.empty_like(empty_cube_images)
        self.exposure_cube = SkyCube.empty_like(empty_exposure_cube)

        self.data_store = data_store
        self.obs_table = obs_table
        self.offset_band = offset_band

        self.header = header
        self.exclusion_mask = exclusion_mask
        if exclusion_mask:
            self.exclusion_mask = exclusion_mask

        self.save_bkg_scale = save_bkg_scale
        if self.save_bkg_scale:
            self.table_bkg_scale = Table(names=["OBS_ID", "bkg_scale"])

    def make_images(self, make_background_image=False, bkg_norm=True, radius=10):
        """Compute the counts, bkg, exposure, excess and significance images for a set of observation.

        Parameters
        ----------
        make_background_image : bool
            True if you want to compute the background and exposure images
        bkg_norm : bool
            If true, apply the scaling factor to the bkg image
        spectral_index : float
            Assumed power-law spectral index
        for_integral_flux : bool
            True if you want that the total excess / exposure gives the integrated flux
        radius : float
            Disk radius in pixels for the significance image
        """

        for obs_id in self.obs_table['OBS_ID']:
            obs = self.data_store.obs(obs_id)
            cube_images = SingleObsCubeMaker(obs=obs,
                                             empty_cube_images=self.empty_cube_images,
                                             empty_exposure_cube=self.empty_exposure_cube,
                                             header=self.header,
                                             offset_band=self.offset_band,
                                             exclusion_mask=self.exclusion_mask,
                                             save_bkg_scale=self.save_bkg_scale)
            cube_images.make_counts_cube()
            self.counts_cube.data += cube_images.counts_cube.data
            if make_background_image:
                cube_images.make_bkg_cube(bkg_norm)
                if self.save_bkg_scale:
                    self.table_bkg_scale.add_row(
                        cube_images.table_bkg_scale[0])
                cube_images.make_exposure_cube()
                self.bkg_cube.data += cube_images.bkg_cube.data
                self.exposure_cube.data += cube_images.exposure_cube.data.to("m2 s")
        if make_background_image:
            self.make_significance_cube(radius)
            self.make_excess_cube()

    def make_significance_cube(self, radius):
        """Make the significance image from the counts and bkg images.

        Parameters
        ----------
        radius : float
            Disk radius in pixels.
        """
        for i_E in range(len(self.energy_reco_bins)-1):
            counts = disk_correlate(self.counts_cube.data[i_E, :, :], radius)
            bkg = disk_correlate(self.bkg_cube.data[i_E, :, :], radius)
            self.significance_cube.data[i_E, :, :] = significance(counts, bkg)

    def make_excess_cube(self):
        """Compute excess between counts and bkg image."""
        for i_E in range(len(self.energy_reco_bins)-1):
            self.excess_cube.data[i_E, :, :] = self.counts_cube.data[i_E, :,
                                               :] - self.bkg_cube.data[i_E, :, :]
