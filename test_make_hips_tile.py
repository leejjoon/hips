import astropy.io.fits as pyfits
import os

os.environ['ASTROPY_SKIP_CONFIG_UPDATE'] = "1"
import hips
from hips.utils.healpix import hips_tile_healpix_ipix_array, healpix_order_to_npix

shift_order = 8
width = 2**shift_order
npix_per_tile = width * width

ipix_of_tile = hips_tile_healpix_ipix_array(shift_order).T[::-1]

from hips.tiles.tile import HipsTileMeta


import pandas as pd
import numpy as np
df = pd.read_parquet("spherex_survey_202303_hpx_nside2048_nexp_simple.parquet")
nside=2048
from healpy.pixelfunc import nest2ring
# The original data is in ring order. We convert it to nest order.
arr_ring = df["nexp"].array
arr_nest = arr_ring[nest2ring(nside, np.arange(len(arr_ring)))].astype("int32")

order = 3
ntiles = healpix_order_to_npix(order)

for ipix in range(ntiles):
    tile = HipsTileMeta(order, ipix, # nest2ring(2**order, ipix),
                        file_format="fits", frame="icrs", width=width)
    p = tile.tile_default_path
    p.parent.mkdir(parents=True, exist_ok=True)
    im = arr_nest[ipix*npix_per_tile + ipix_of_tile]

    pyfits.PrimaryHDU(data=im).writeto(p, overwrite=True)

ipix = 18
def get_im(ipix):
    tile = HipsTileMeta(order, ipix, file_format="fits", frame="icrs", width=512)
    im = arr_nest[ipix*npix_per_tile + ipix_of_tile]
    return im

# downsample

order = 2
ntiles = healpix_order_to_npix(order)
arr_nest_o2 = np.mean([arr_nest[i::4] for i in range(4)], axis=0).astype("int32")

for ipix in range(ntiles):
    tile = HipsTileMeta(order, ipix, file_format="fits", frame="icrs", width=512)
    p = tile.tile_default_path
    p.parent.mkdir(parents=True, exist_ok=True)
    im = arr_nest_o2[ipix*npix_per_tile + ipix_of_tile]

    pyfits.PrimaryHDU(data=im).writeto(p, overwrite=True)

def get_im_o1(ipix):
    tile = HipsTileMeta(order, ipix, file_format="fits", frame="icrs", width=512)

    im = arr_nest_o1[ipix*npix_per_tile + ipix_of_tile]
    return im

order = 1
ntiles = healpix_order_to_npix(order)
arr_nest_o1 = np.mean([arr_nest_o2[i::4] for i in range(4)], axis=0).astype("int32")

for ipix in range(ntiles):
    tile = HipsTileMeta(order, ipix, file_format="fits", frame="icrs", width=512)
    p = tile.tile_default_path
    p.parent.mkdir(parents=True, exist_ok=True)
    im = arr_nest_o1[ipix*npix_per_tile + ipix_of_tile]

    pyfits.PrimaryHDU(data=im).writeto(p, overwrite=True)

# downsample

order = 0
ntiles = healpix_order_to_npix(order)
arr_nest_o0 = np.mean([arr_nest_o1[i::4] for i in range(4)], axis=0).astype("int32")


for ipix in range(ntiles):
    tile = HipsTileMeta(order, ipix, file_format="fits", frame="icrs", width=512)
    p = tile.tile_default_path
    p.parent.mkdir(parents=True, exist_ok=True)
    im = arr_nest_o0[ipix*npix_per_tile + ipix_of_tile]

    pyfits.PrimaryHDU(data=im).writeto(p, overwrite=True)

# make allsky
# for order 3
# 64x64


order = 3
ntiles = healpix_order_to_npix(order)

ncol = int(np.sqrt(ntiles))
tilewidth = 64
nsample = width // tilewidth
subtiles = []
for ipix in range(ntiles):
    im0 = arr_nest[ipix*npix_per_tile + ipix_of_tile]
    im = np.mean([im0[iy::nsample, ix::nsample]
                  for ix in range(nsample) for iy in range(nsample)],
                 axis=0)
    subtiles.append(im)

b = [subtiles[i: i+ncol] for i in range(0, ntiles, ncol)]

b[-1].extend([np.zeros_like(b[0][0])] * (ncol - len(b[-1])))


bb = np.block(b[::-1])

pyfits.PrimaryHDU(data=bb).writeto(f"Norder{order}/Allsky.fits", overwrite=True)
