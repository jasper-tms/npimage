#!/usr/bin/env python3

import os

import numpy as np

import npimage


def test_pbm_load_save():
    test_cases = [
        {
            "data": np.array([
                [1, 0, 1, 0, 1],
                [0, 1, 0, 1, 0],
                [1, 1, 1, 0, 0]
            ], dtype=bool),
            "filename": "test_arbitrary_width.pbm",
            "description": "PBM read/write, arbitrary width"
        },
        {
            "data": np.array([
                [1, 0, 1, 0, 1, 0, 1, 0],
                [0, 1, 0, 1, 0, 1, 0, 1],
                [1, 1, 1, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 1, 1, 1]
            ], dtype=bool),
            "filename": "test_multiple_of_8.pbm",
            "description": "PBM read/write, width a multiple of 8"
        }
    ]

    for case in test_cases:
        test_data = case["data"]
        test_filename = case["filename"]
        description = case["description"]

        try:
            # Save the test data to a PBM file
            npimage.save(test_data, test_filename)

            # Verify the file size
            expected_file_size = npimage.filetypes.pbm.predict_file_size(test_data)
            actual_file_size = os.path.getsize(test_filename)
            assert actual_file_size == expected_file_size, f"File size mismatch: expected {expected_file_size}, got {actual_file_size}"

            # Load the data back from the PBM file
            loaded_data = npimage.load(test_filename)

            # Assert that the loaded data matches the original data
            assert np.array_equal(test_data, loaded_data), f"Loaded data does not match original data for {description}"

            print(f"Test passed: {description} works correctly.")
        finally:
            # Clean up the test file
            if os.path.exists(test_filename):
                os.remove(test_filename)


if __name__ == '__main__':
    test_pbm_load_save()
