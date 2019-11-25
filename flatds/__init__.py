import xarray as xr
import numpy as np
import msgpack
import struct
from contextlib import closing

MAGIC = np.array(list(map(ord, "FLATDS")), dtype="uint8")


class FlatdsWriter(object):
    def __init__(self, filename):
        self.filename = filename
        self.f = open(self.filename, "wb")
        self.f.write(b"FLATDS\x00\x00")
        self._vars = {}
        self._dims = []

    def close(self):
        self._write_header()
        self.f.close()

    def _add_dim(self, name, size):
        for i, entry in enumerate(self._dims):
            current_name, current_size = entry
            if current_name == name:
                if current_size != size:
                    raise ValueError("size missmatch for existing dim: {} {} != {}".format(name, current_size, size))
                return i
        self._dims.append((name, size))
        return len(self._dims) - 1

    def write_variale(self, name, array, dims, attrs=None):
        if name in self._vars:
            raise ValueError("array {} has already been written!")
        if len(dims) != len(array.shape):
            raise ValueError("dimension size and shape size missmatch")

        dim_indices = list(map(self._add_dim, dims, array.shape))

        start = self.f.tell()
        align = array.dtype.alignment
        misalign = start % align
        if misalign != 0:
            ofs = align - misalign
            self.f.write('\x00'*ofs)
            start += ofs
        self.f.write(array)
        self._vars[name] = {"ofs": start,
                            "t": array.dtype.name,
                            "st": array.strides,
                            "is": array.dtype.itemsize,
                            "d": dim_indices}
        if attrs:
            self._vars[name]["attrs"] = attrs

    def write_data_array(self, name, arr):
        return self.write_variale(name, arr.data, arr.dims, arr.attrs)

    def _write_header(self):
        start = self.f.tell()
        header = {"vars": self._vars, "dims": self._dims}
        msgpack.pack(header, self.f, use_bin_type=False)
        self.f.write(struct.pack("<Q", start))


def write_xarray_dataset(filename, ds):
    with closing(FlatdsWriter(filename)) as w:
        for name, var in ds.variables.items():
            w.write_data_array(name, var)


def open_flatds(filename, writeable=False):
    data = np.memmap(filename, dtype="uint8", mode="r+" if writeable else "r")
    if np.any(data[:len(MAGIC)] != MAGIC):
        raise ValueError("file \"{}\" is not a flatds file".format(filename))
    if data[len(MAGIC)] != 0:
        raise ValueError("unknown header location")
    # header is in the back
    header_location = data[-8:].view("uint64")[0]
    header = msgpack.unpackb(data[header_location:-8], raw=False)

    def get_var(props):
        if len(props["d"]) > 0:
            dims, shape = zip(*[header["dims"][d] for d in props["d"]])
            size = np.prod(shape) * props["is"]
        else:
            dims = ()
            shape = ()
            size = props["is"]
        ofs = props["ofs"]
        d = data[ofs:ofs+size]
        d = d.view(dtype=props["t"])
        d = np.lib.stride_tricks.as_strided(d, shape, props["st"], subok=True, writeable=writeable)
        attrs = props.get("attrs", {})
        return xr.DataArray(d, dims=dims, attrs=attrs)

    variables = {name: get_var(p) for name, p in header["vars"].items()}
    attrs = header.get("attrs", {})
    return xr.decode_cf(xr.Dataset(variables, attrs=attrs))
