import astropy.io.fits as pyfits
import os

os.environ['ASTROPY_SKIP_CONFIG_UPDATE'] = "1"
import hips
from hips.utils.healpix import hips_tile_healpix_ipix_array, healpix_order_to_npix

shift_order = 9
width = 2**shift_order
npix_per_tile = width * width

ipix_of_tile = hips_tile_healpix_ipix_array(shift_order)

from hips.tiles.tile import HipsTileMeta


import pandas as pd
import numpy as np
df = pd.read_parquet("spherex_survey_202303_hpx_nside2048_nexp_simple.parquet")
nside=2048
from healpy.pixelfunc import nest2ring
# The original data is in ring order. We convert it to nest order.
arr_ring = df["nexp"].array
arr_nest = arr_ring[nest2ring(nside, np.arange(len(arr_ring)))]

import healpy as hp
hp.fitsfunc.write_map("test_hips2/test_healpix2.fits", arr_nest.astype("int32"),
                      nest=True, coord="C", overwrite=True, dtype="int32")

# f = pyfits.open("test_hips/HFI_SkyMap_100_2048_R3.01_full.fits")
# f[1].data["HITS"][:] = arr_nest

# f.writeto("test_hips2/test_healpix.fits", overwrite=True)

