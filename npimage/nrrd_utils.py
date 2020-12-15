#!/usr/bin/env python3

# Utility functions for nrrd files

import sys
import os
import glob

import nrrd  # pip install pynrrd


def compress(*fn_patterns, keep_uncompressed=False):
    if len(fn_patterns) == 0:
        fn_patterns = ['*.nrrd']
        i = input(f'Compressing all {len(glob.glob("*nrrd"))} nrrd'
                  ' files in this directory. Continue? [Y/n]')
        if i.lower() != 'y':
            return

    for fn_pattern in fn_patterns:
        for fn in glob.glob(fn_pattern):

            print(f'Loading input file {fn} into memory.')
            data, header = nrrd.read(fn)
            if 'encoding' in header and header['encoding'] == 'gzip':
                print('That file is already compressed. Skipping.')
                continue
            elif 'encoding' in header and header['encoding'] == 'raw':
                header['encoding'] = 'gzip'

            if keep_uncompressed:
                fn_moved = fn.replace('.nrrd', '.uncompressed.nrrd')
                print(f'Moving {fn} to {fn_moved} and writing compressed version to {fn}')
                os.rename(fn, fn_moved)
            else:
                print(f'Overwriting {fn} with a compressed version of itself.')

            nrrd.write(fn, data, header=header)


def read_headers(*fn_patterns):
    if len(fn_patterns) == 0:
        fn_patterns = ['*.nrrd']
        print('Reading headers of all nrrd files in this directory.\n')
    for fn_pattern in fn_patterns:
        for fn in glob.glob(fn_pattern):
            print(fn)
            print(nrrd.read_header(fn), '\n')


if __name__ == '__main__':
    l = locals()
    public_functions = [f for f in l if callable(l[f]) and f[0] != '_']
    if len(sys.argv) <= 1 or not sys.argv[1] in public_functions:
        from inspect import signature
        print('Functions available:')
        for f_name in public_functions:
            print('  '+f_name+str(signature(l[f_name])))
            docstring = l[f_name].__doc__
            if not isinstance(docstring, type(None)):
                print(docstring.strip('\n'))
    else:
        func = l[sys.argv[1]]
        args = []
        kwargs = {}
        for arg in sys.argv[2:]:
            if '=' in arg:
                split = arg.split('=')
                kwargs[split[0]] = split[1]
            else:
                args.append(arg)
        #print(f'args, {args}')
        #print(f'kwargs, {kwargs}')
        func(*args, **kwargs)
