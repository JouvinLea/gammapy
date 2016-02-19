# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import numpy as np
from astropy.tests.helper import pytest
from astropy.utils.compat import NUMPY_LT_1_9
from numpy.testing import assert_allclose
from astropy.coordinates import SkyCoord, Angle

from ...spectrum.results import SpectrumFitResult, SpectrumStats
from ...utils.testing import requires_dependency, requires_data, SHERPA_LT_4_8
from ...region import SkyCircleRegion
from ...datasets import gammapy_extra
from ...utils.scripts import read_yaml
from ...utils.energy import EnergyBounds
from ...image import ExclusionMask
from ...data import DataStore
from ...spectrum import (
    SpectrumExtraction,
    run_spectrum_extraction_using_config,
)
from ...spectrum.spectrum_extraction import SpectrumObservationList

def make_spectrum_extraction():
    # Construct w/o config file
    center = SkyCoord(83.63, 22.01, unit='deg', frame='icrs')
    radius = Angle('0.3 deg')
    on_region = SkyCircleRegion(pos=center, radius=radius)

    bkg_method = dict(type='reflected', n_min=2)

    exclusion_file = gammapy_extra.filename(
        "datasets/exclusion_masks/tevcat_exclusion.fits")
    excl = ExclusionMask.from_fits(exclusion_file)

    bounds = EnergyBounds.equal_log_spacing(1, 10, 40, unit='TeV')

    obs = [23523, 23559, 11111]
    store = gammapy_extra.filename("datasets/hess-crab4-hd-hap-prod2")
    ds = DataStore.from_dir(store)

    ana = SpectrumExtraction(datastore=ds, obs_ids=obs, on_region=on_region,
                           bkg_method=bkg_method, exclusion=excl,
                           ebounds=bounds)
    return ana

@requires_dependency('scipy')
@requires_data('gammapy-extra')
def test_spectrum_extraction(tmpdir):
    # Construct w/o config file
    center = SkyCoord(83.63, 22.01, unit='deg', frame='icrs')
    radius = Angle('0.3 deg')
    on_region = SkyCircleRegion(pos=center, radius=radius)

    bkg_method = dict(type='reflected', n_min=2)

    exclusion_file = gammapy_extra.filename(
        "datasets/exclusion_masks/tevcat_exclusion.fits")
    excl = ExclusionMask.from_fits(exclusion_file)

    bounds = EnergyBounds.equal_log_spacing(1, 10, 40, unit='TeV')

    obs = [23523, 23559, 11111, 23592]
    store = gammapy_extra.filename("datasets/hess-crab4-hd-hap-prod2")
    ds = DataStore.from_dir(store)

    ana = SpectrumExtraction(datastore=ds, obs_ids=obs, on_region=on_region,
                           bkg_method=bkg_method, exclusion=excl,
                           ebounds=bounds)

    #test methods on SpectrumObservationList
    obs = ana.observations
    assert len(obs) == 3
    obs23523 = obs.get_obslist_from_obsid([23523])[0]
    assert obs23523.on_vector.total_counts == 123
    new_list = obs.get_obslist_from_obsid([23523, 23592])
    assert new_list[0].obs_id == 23523
    assert new_list[1].obs_id == 23592

# @pytest.mark.xfail
@requires_dependency('yaml')
@requires_dependency('scipy')
@requires_data('gammapy-extra')
def test_spectrum_extraction_from_configfile(tmpdir):
    configfile = gammapy_extra.filename(
        'test_datasets/spectrum/spectrum_analysis_example.yaml')
    config = read_yaml(configfile)
    config['extraction']['results']['outdir'] = str(tmpdir)
    tmpfile = tmpdir / 'spectrum.yaml'
    config['extraction']['results']['result_file'] = str(tmpfile)

    run_spectrum_extraction_using_config(config)
    actual = SpectrumStats.from_yaml(str(tmpfile))

    desired = SpectrumStats.from_yaml(
        gammapy_extra.filename('test_datasets/spectrum/spectrum.yaml'))

    assert actual.n_on == desired.n_on


