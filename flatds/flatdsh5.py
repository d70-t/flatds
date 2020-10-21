import h5py
import numpy as np
from . import read_header

import logging
logger = logging.getLogger(__name__)

def is_compatible(header1, header2, concat_dim):
    d1 = dict(header1["dims"])
    d2 = dict(header2["dims"])
    if concat_dim not in d1:
        logger.info("concat dim %s is missing in header 1 (%s)", concat_dim, d1)
        return False
    if concat_dim not in d2:
        logger.info("concat dim %s is missing in header 2 (%s)", concat_dim, d2)
        return False
    del d1[concat_dim]
    del d2[concat_dim]
    if d1 != d2:
        logger.info("other dims are not of equal size (%s != %s)", d1, d2)
        return False

    loneley_vars = set(header1["vars"]) ^ set(header2["vars"])
    if len(loneley_vars) > 0:
        logger.info("vars %s are only in one header", loneley_vars)
        return False

    for var in header1["vars"]:
        for prop in ["t", "st", "is", "d"]:
            if header1["vars"][var][prop] != header2["vars"][var][prop]:
                logger.info("prop %s on var %s differs between headers", prop, var)
                return False

    return True

def concat_flatds(flatds_files, concat_dim, outfile):
    headers = [read_header(fn) for fn in flatds_files]
    print(len(headers))
    print(next(iter(headers[0]["vars"].items())))

    if not all(is_compatible(headers[0], h, concat_dim) for h in headers[1:]):
        raise ValueError("some headers are not compatible")

    concat_dim_id = [i for i, (n, s) in enumerate(headers[0]["dims"]) if n == concat_dim][0]
    variables = headers[0]["vars"]
    concat_vars = {k for k, v in variables.items() if concat_dim_id in v["d"]}
    concat_size = [h["dims"][concat_dim_id][1] for h in headers]
    total_concat_size = sum(concat_size)
    varmeta = {}
    for var in concat_vars:
        varinfo = headers[0]["vars"][var]
        max_stride = np.argmax(varinfo["st"])
        concat_dim_index = varinfo["d"].index(concat_dim_id)
        if max_stride != concat_dim_index:
            logger.error("variable %s is not stored along concat dimension. need virtual datasets to do this, which is not yet implemented", var)
            raise ValueError("invalid dataset orientation")
        blocks = [
            {"file": filename,
             "size": h["dims"][concat_dim_id][1] * h["vars"][var]["st"][concat_dim_index],
             "ofs": h["vars"][var]["ofs"],
            }
            for filename, h in zip(flatds_files, headers)
        ]
        dims = [headers[0]["dims"][d][0] for d in varinfo["d"]]
        shape = [headers[0]["dims"][d][1] for d in varinfo["d"]]
        shape[concat_dim_index] = total_concat_size
        varmeta[var] = {
            "name": var,
            "shape": shape,
            "strides": varinfo["st"],
            "dtype": varinfo["t"],
            "dims": dims,
            "blocks": blocks,
            "attrs": varinfo["attrs"],
        }

    dims = dict(headers[0]["dims"])
    dims[concat_dim] = total_concat_size
    global_attrs = headers[0]["attrs"]

    of = h5py.File(outfile, "w")

    def make_ds(meta):
        print("creating {}".format(meta["name"]))
        ds = of.create_dataset(meta["name"],
                               shape=meta["shape"],
                               dtype=meta["dtype"],
                               external=[(b["file"], b["ofs"], b["size"]) for b in meta["blocks"]])
        for k, v in meta["attrs"].items():
            ds.attrs[k] = v
        return ds

    scales = {}
    for dim, size in dims.items():
        if dim in varmeta:
            ds = make_ds(varmeta[dim])
            ds.make_scale(dim)
        else:
            ds = of.create_dataset(dim, shape=(size,), dtype="int8")
            ds.make_scale("This is a netCDF dimension but not a netCDF variable.")
        scales[dim] = ds

    for var, meta in sorted(varmeta.items()):
        if var in dims:
            continue
        ds = make_ds(varmeta[var])
        for i, dim in enumerate(meta["dims"]):
            ds.dims[i].attach_scale(scales[dim])

    for k, v in global_attrs.items():
        of.attrs[k] = v

    of.close()


def _main():
    logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("infiles", nargs="+", type=str)
    parser.add_argument("-o", "--outfile", type=str, help="output hdf5 / nc file")
    args = parser.parse_args()

    concat_flatds(sorted(args.infiles), "time", args.outfile)

if __name__ == "__main__":
    exit(_main())
