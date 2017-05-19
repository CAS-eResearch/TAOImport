#!/usr/bin/env python

from __future__ import print_function

import h5py
import numpy as np
import numbers
import os


def compare_fields_by_name(field1, field2, fieldname, failed=0,
                           failed_fields=[], field_prefix=''):

    import numpy as np
    if isinstance(field1[0], numbers.Integral):
        compare = bool(np.all((field1- field2) == 0))
    else:
        finite = np.isfinite(field1)
        compare = bool(np.allclose(field1[finite], field2[finite]))
        
    if not compare:
        diff = field1 - field2
        print('Field = {0} : min. diff = {0} max. diff = {1}'.
              format(fieldname, min(diff), max(diff)))

        failed += int(not compare)
        if not compare:
            if fieldname not in failed_fields:
                failed_fields.append('{0}{1}'.format(field_prefix, fieldname))

    return failed, failed_fields


def compare_galaxy_vars(serial_galaxies, par_galaxies,
                        failed=0, failed_fields=[]):
    assert par_galaxies.dtype == serial_galaxies.dtype
    dtype = par_galaxies.dtype

    for f in dtype.names:
        failed, failed_fields = compare_fields_by_name(par_galaxies[f],
                                                       serial_galaxies[f],
                                                       str(f),
                                                       failed,
                                                       failed_fields,
                                                       field_prefix='galaxies/'
                                                       )

    return failed, failed_fields


def compare_serial_runs(file1, file2):
    print("########################################################################################################")
    print("#                     Filename              Ngalaxies   Ntrees     Status      NumFailed    FailedFields")
    print("########################################################################################################")
    with h5py.File(file1, 'r') as ser1, h5py.File(file2, 'r') as ser2:
        failed, failed_fields = compare_galaxy_vars(ser1['galaxies'],
                                                    ser2['galaxies'])

        
        for f in ['tree_counts', 'tree_displs', 'snapshot_redshifts']:
            failed, failed_fields = compare_fields_by_name(ser1[f],
                                                           ser2[f],
                                                           f,
                                                           failed,
                                                           failed_fields)
        
    RED   = "\033[1;31m"
    GREEN = "\033[0;32m"
    RESET = "\033[0;0m"
    status = (RED+'FAILED'+RESET) if failed > 0 else (GREEN + 'PASSED' + RESET)
    if failed == 0:
        failed_fields = None
    print("{0:30s} {1:10d}  {2:8d}     {3:6s}    {4:8d}         {5}"
          .format(os.path.basename(file2),
                  ngal,
                  ntrees,
                  status,
                  failed,
                  failed_fields
                  ))
    

def compare_serial_and_parallel_runs(serial_file, parallel_files):
    serial_galaxy_offset = 0
    serial_tree_offset = 0

    print("########################################################################################################")
    print("#                     Filename              Ngalaxies   Ntrees     Status      NumFailed    FailedFields")
    print("########################################################################################################")
    with h5py.File(serial_file, 'r') as ser:
        serial_galaxies = ser['galaxies']
        serial_tree_counts = ser['tree_counts']
        serial_tree_displs = ser['tree_displs']
        for pfile in parallel_files:
            with h5py.File(pfile, 'r') as par:
                par_galaxies = par['galaxies']
                par_tree_counts = par['tree_counts']
                par_tree_displs = par['tree_displs']

                ngal = par_galaxies.shape[0]
                ntrees = par_tree_counts.shape[0]
                serial_galaxies_view = serial_galaxies[serial_galaxy_offset:serial_galaxy_offset + ngal]
                serial_tree_counts_view = serial_tree_counts[serial_tree_offset:serial_tree_offset + ntrees]
                serial_tree_displs_view = serial_tree_displs[serial_tree_offset:serial_tree_offset + ntrees + 1]

                failed, failed_fields = compare_galaxy_vars(serial_galaxies_view, par_galaxies)
                failed, failed_fields = compare_fields_by_name(par_tree_counts,
                                                               serial_tree_counts_view,
                                                               'tree_counts',
                                                               failed,
                                                               failed_fields)
                failed, failed_fields = compare_fields_by_name(par_tree_displs,
                                                               serial_tree_displs_view,
                                                               'tree_displs',
                                                               failed,
                                                               failed_fields)

                serial_galaxy_offset += ngal
                serial_tree_offset += ntrees

                RED   = "\033[1;31m"
                GREEN = "\033[0;32m"
                RESET = "\033[0;0m"
                status = (RED+'FAILED'+RESET) if failed > 0 else (GREEN + 'PASSED' + RESET)
                if failed == 0:
                    failed_fields = None
                print("{0:>35s}     {1:10d}  {2:8d}     {3:6s}    {4:8d}         {5}"
                      .format(os.path.basename(pfile),
                              ngal,
                              ntrees,
                              status,
                              failed,
                              failed_fields
                              ))

        if serial_galaxy_offset != serial_galaxies.shape[0]:
            print("Error: The serial file has {0} galaxies but the parallel "
                  "files combined have {1} galaxies"
                  .format(serial_galaxies.shape[0],
                          serial_galaxy_offset))


def check_if_parallel_file(filename):
    import h5py
    parallel_files = None
    with h5py.File(filename, 'r') as hf:
        if 'filenames' in hf.attrs.keys():
            parallel_files = hf.attrs['filenames']
            parallel_run = 1
        else:
            parallel_run = 0

    return parallel_run, parallel_files

def main():
    import argparse

    parser = argparse.ArgumentParser(description='Verify the parallel run of the TAO converter against a (known-good) serial run')
    parser.add_argument('-s', '--serial', help='filename for the hdf5 file produced during the serial run', required=True)
    parser.add_argument('-p', '--parallel', help='filename for the hdf5 file produced during the parallel run', required=True)

    args = parser.parse_args()

    serial_file = args.serial
    par_file = args.parallel

    if (check_if_parallel_file(serial_file))[0]:
        msg = 'Error: The serial file was generated by a parallel writing '\
            'process.\nPlease specify a serial run -- either by using a '\
            'serial run file directly\nor by a single core parallel run '\
            '(use the `filename_0.h5`)'
        raise ValueError(msg)
    
    parallel_run, parallel_files = check_if_parallel_file(par_file)

    print("Comparing between these two files:")
    print("Serial   : {0}".format(serial_file))
    if parallel_run:
        print("Parallel : {0}".format(par_file))
        print("")
        compare_serial_and_parallel_runs(serial_file, parallel_files)
    else:
        print("Serial : {0}".format(par_file))
        print("")
        compare_serial_runs(serial_file, par_file)


if __name__ == '__main__':
    main()
