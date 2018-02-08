# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import pytest
import numpy as np
from numpy.testing import assert_allclose
from ..poisson import (
    background,
    background_error,
    excess,
    excess_error,
    significance,
    significance_on_off,
    excess_matching_significance,
    excess_matching_significance_on_off,
)


def test_background():
    assert_allclose(background(n_off=4, alpha=0.1), 0.4)
    assert_allclose(background(n_off=9, alpha=0.2), 1.8)


def test_background_error():
    assert_allclose(background_error(n_off=4, alpha=0.1), 0.2)
    assert_allclose(background_error(n_off=9, alpha=0.2), 0.6)


def test_excess():
    assert_allclose(excess(n_on=10, n_off=20, alpha=0.1), 8.0)
    assert_allclose(excess(n_on=4, n_off=9, alpha=0.5), -0.5)


def test_excess_error():
    assert_allclose(excess_error(n_on=10, n_off=20, alpha=0.1), 3.1937439)
    assert_allclose(excess_error(n_on=4, n_off=9, alpha=0.5), 2.5)


def test_significance():
    # Check that the Li & Ma limit formula is correct
    # With small alpha and high counts, the significance
    # and significance_on_off should be very close
    actual = significance(n_on=1300, mu_bkg=1100, method='lima')
    assert_allclose(actual, 5.8600870406703329)
    actual = significance_on_off(n_on=1300, n_off=1100 / 1.e-8, alpha=1e-8, method='lima')
    assert_allclose(actual, 5.8600864348078519)


TEST_CASES = [
    dict(n_on=10, n_off=20, alpha=0.1, method='lima', s=3.6850322025333071),
    dict(n_on=4, n_off=9, alpha=0.5, method='lima', s=-0.19744427645023557),
    dict(n_on=10, n_off=20, alpha=0.1, method='simple', s=2.5048971643405982),
    dict(n_on=4, n_off=9, alpha=0.5, method='simple', s=-0.2),
    dict(n_on=10, n_off=20, alpha=0.1, method='lima', s=3.6850322025333071),
    dict(n_on=2, n_off=200, alpha=0.1, method='lima', s=-5.027429),
]


@pytest.mark.parametrize('p', TEST_CASES)
def test_significance_on_off(p):
    s = significance_on_off(p['n_on'], p['n_off'], p['alpha'], p['method'])
    assert_allclose(s, p['s'], atol=1e-5)


def make_test_cases():
    for n_on in [0.1, 10, 0.3]:
        for n_off in [0.1, 10, 0.3]:
            for alpha in [1e-3, 1e-2, 0.1, 1, 10]:
                for method in ['simple', 'lima']:
                    yield dict(n_on=n_on, n_off=n_off, alpha=alpha, method=method)


@pytest.mark.xfail()
def test_excess_matching_significance():
    raise NotImplementedError


@pytest.mark.parametrize('p', list(make_test_cases()))
# @pytest.mark.parametrize('p', TEST_CASES)
def test_excess_matching_significance_on_off(p):
    # Check round-tripping
    s = significance_on_off(p['n_on'], p['n_off'], p['alpha'], p['method'])
    excess = excess_matching_significance_on_off(p['n_off'], p['alpha'], s, p['method'])
    n_on = excess + background(p['n_off'], p['alpha'])
    s2 = significance_on_off(n_on, p['n_off'], p['alpha'], p['method'])
    print(s, s2, n_on)
    # p.update(s=s, s2=s2, excess=excess, n_on_reco=n_on)
    # from pprint import pprint
    # pprint(p)
    # assert_allclose(s, s2, rtol=1e-3)


def test_excess_matching_significance_on_off_arrays():
    excess = excess_matching_significance_on_off(n_off=[10, 20], alpha=0.1, significance=5)
    assert_allclose(excess, [9.82966, 12.038423], atol=1e-3)
