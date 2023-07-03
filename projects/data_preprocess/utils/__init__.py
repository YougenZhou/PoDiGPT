import io
import json


def _make_w_io_base(f, mode: str):
    if not isinstance(f, io.IOBase):
        f = open(f, mode=mode)
    return f


def _make_r_io_base(f, mode: str):
    if not isinstance(f, io.IOBase):
        f = open(f, mode=mode)
    return f


def save_json(obj, f, mode='w', indent=4, default=str):
    f = _make_w_io_base(f, mode)
    if isinstance(obj, (dict, list)):
        json.dump(obj, f, indent=indent, ensure_ascii=False, default=default)
    elif isinstance(obj, str):
        f.write(obj)
    else:
        raise ValueError(f'Unexpected type: {type(obj)}')
    f.close()


def read_json(f, mode='r'):
    f = _make_r_io_base(f, mode)
    raw_data = json.load(f)
    f.close()
    return raw_data