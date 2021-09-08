#!/usr/bin/env python3

import json
import warnings

from sys import float_info
from pathlib import PosixPath
from datetime import datetime


def approx_eq_float(a, b, abs_tol=2 * float_info.epsilon):
    '''Define approximate float equality using an absolute tolerance;
       for direct fp subtraction, the error upper bound is 2 epsilon,
       but this can increase with additional operations, e.g. a =? b*c.'''
    return abs(a - b) <= abs_tol


def to_timestamp(dt, rtype=int, epoch=datetime(1970, 1, 1)):
    '''Converts a datetime object into a POSIX timestamp (either of (1) integer
       type to millisecond precision, or (2) float type, to microsecond precision).'''
    if rtype not in [int, float]:
        raise ValueError(
            'Return type must be either integer or (double-precision) float!')
    td = dt - epoch
    ts = td.total_seconds()
    if rtype == int:
        return int(round(ts))
    return ts


def safe_filename(dir, savename, file_extension='pdf', overwrite=False):
    '''Helper function to optionally avoid overwriting duplicate filenames.'''
    filename = PosixPath(dir) / f'{savename}.{file_extension}'
    if not overwrite:
        dup_count = 1
        while filename.is_file():
            filename = PosixPath(dir) / f'{savename}({dup_count}).{file_extension}'
            dup_count += 1
    return filename


def json_pretty_dumps(j_obj, file_ptr=None, sort_keys=True, wrap_line_length=0,
                      keep_array_indentation=True, indent_size=4):
    '''Helper function using jsbeautifier (if available) to serialize JSON with more human-readable
       formatting. If file_ptr is supplied, also writes the resulting JSON string to file.'''
    try:
        import jsbeautifier
        # Setup beautifier options
        options = jsbeautifier.default_options()
        options.indent_size = indent_size
        options.wrap_line_length = wrap_line_length
        options.keep_array_indentation = keep_array_indentation
        # Convert j_obj to a JSON string, then beautify it
        j_string = jsbeautifier.beautify(json.dumps(j_obj, sort_keys=sort_keys), options)
    except ImportError:
        warnings.warn(
            "jsbeautifier module not found (to install it, try, e.g., 'pip install jsbeautifier'),"
            + "falling back to default serializer...", ImportWarning)
        j_string = json.dumps(j_obj, sort_keys=sort_keys)
    if file_ptr:
        file_ptr.write(j_string)
    return j_string
