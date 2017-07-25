# Licensed under a 3-clause BSD style license - see LICENSE.rst
import pytest
import numpy as np
from numpy.testing import assert_allclose
from astropy.coordinates import SkyCoord
from astropy.tests.helper import remote_data
from ...tiles import HipsSurveyProperties
from ..simple import make_sky_image, SimpleTilePainter, plot_mpl_single_tile, _is_tile_distorted, _measure_tile_shape
from ...utils.wcs import WCSGeometry
from ...utils.testing import make_test_wcs_geometry, requires_hips_extra

make_sky_image_pars = [
    dict(
        file_format='fits',
        shape=(1000, 2000),
        url='http://alasky.unistra.fr/DSS/DSS2Merged/properties',
        data_1=2213,
        data_2=2296,
        data_sum=8756493140,
        dtype='>i2',
    ),
    dict(
        file_format='jpg',
        shape=(1000, 2000, 3),
        url='https://raw.githubusercontent.com/hipspy/hips-extra/master/datasets/samples/FermiColor/properties',
        data_1=[133, 117, 121],
        data_2=[137, 116, 114],
        data_sum=828908873,
        dtype='uint8',
    ),
    dict(
        file_format='png',
        shape=(1000, 2000, 4),
        url='https://raw.githubusercontent.com/hipspy/hips-extra/master/datasets/samples/AKARI-FIS/properties',
        data_1=[224, 216, 196, 255],
        data_2=[227, 217, 205, 255],
        data_sum=1635622838,
        dtype='uint8',
    ),
]


@remote_data
@pytest.mark.parametrize('pars', make_sky_image_pars)
def test_make_sky_image(pars):
    hips_survey = HipsSurveyProperties.fetch(url=pars['url'])
    geometry = make_test_wcs_geometry()
    image = make_sky_image(geometry=geometry, hips_survey=hips_survey, tile_format=pars['file_format'])
    assert image.shape == pars['shape']
    assert image.dtype == pars['dtype']
    assert_allclose(np.sum(image), pars['data_sum'])
    assert_allclose(image[200, 994], pars['data_1'])
    assert_allclose(image[200, 995], pars['data_2'])


@remote_data
class TestSimpleTilePainter:
    @classmethod
    def setup_class(cls):
        url = 'http://alasky.unistra.fr/DSS/DSS2Merged/properties'
        cls.hips_survey = HipsSurveyProperties.fetch(url)
        cls.geometry = WCSGeometry.create(
            skydir=SkyCoord(0, 0, unit='deg', frame='icrs'),
            width=2000, height=1000, fov="3 deg",
            coordsys='icrs', projection='AIT',
        )
        cls.painter = SimpleTilePainter(cls.geometry, cls.hips_survey, 'fits')

    def test_draw_hips_order(self):
        assert self.painter.draw_hips_order == 7

    def test_tile_indices(self):
        assert list(self.painter.tile_indices)[:4] == [69623, 69627, 69628, 69629]

    draw_hips_order_pars = [
        dict(order=7, fov="3 deg"),
        dict(order=5, fov="10 deg"),
        dict(order=4, fov="15 deg"),
    ]

    @requires_hips_extra()
    @pytest.mark.parametrize('pars', draw_hips_order_pars)
    def test_compute_matching_hips_order(self, pars):
        geometry = WCSGeometry.create(
            skydir=SkyCoord(0, 0, unit='deg', frame='icrs'),
            width=2000, height=1000, fov=pars['fov'],
            coordsys='icrs', projection='AIT',
        )

        simple_tile_painter = SimpleTilePainter(geometry, self.hips_survey, 'fits')
        assert simple_tile_painter.draw_hips_order == pars['order']

    def test_run(self):
        self.painter.run()
        assert self.painter.image.shape == (1000, 2000)
        assert_allclose(self.painter.image[200, 994], 2120)

    def test_draw_hips_tile_grid(self):
        self.painter.plot_mpl_hips_tile_grid()

    def test_draw_debug_image(self):
        tile = self.painter.tiles[3]
        image = self.painter.image
        plot_mpl_single_tile(self.geometry, tile, image)

    def test_corners(self):
        tile = self.painter.tiles[3]
        x, y = tile.meta.skycoord_corners.to_pixel(self.geometry.wcs)

        assert_allclose(x, [764.627476, 999., 764.646551, 530.26981])
        assert_allclose(y, [300.055412, 101.107245, -97.849955, 101.105373])

    def test_is_tile_distorted(self):
        tile = self.painter.tiles[3]
        corners = tile.meta.skycoord_corners.to_pixel(self.geometry.wcs)
        assert _is_tile_distorted(corners) == True

    def test_measure_tile_shape(self):
        tile = self.painter.tiles[3]
        corners = tile.meta.skycoord_corners.to_pixel(self.geometry.wcs)
        edges, diagonals, ratio = _measure_tile_shape(corners)

        edges_precomp = [307.426175, 307.417479, 307.434024, 307.41606]
        diagonals_precomp = [397.905367, 468.73019]
        ratio_precomp = 0.848900658905216

        assert_allclose(edges_precomp, edges)
        assert_allclose(diagonals_precomp, diagonals)
        assert_allclose(ratio_precomp, ratio)
