#!/usr/bin/env python3

import os
import tempfile
import numpy as np
import npimage


def test_heic_load_save():
    """Test loading and saving HEIC images."""
    # Create a simple test image
    test_data = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Test saving as HEIC
        heic_filename = os.path.join(tmpdir, "test_image.heic")
        
        try:
            # Save the test data as HEIC
            npimage.save(test_data, heic_filename)
            
            # Verify the file was created
            assert os.path.exists(heic_filename), "HEIC file was not created"
            
            # Load the HEIC file back
            loaded_data = npimage.load(heic_filename)
            
            # Check that the loaded data has the same shape
            assert loaded_data.shape == test_data.shape, f"Shape mismatch: expected {test_data.shape}, got {loaded_data.shape}"
            
            # Check that the data type is correct
            assert loaded_data.dtype == np.uint8, f"Data type mismatch: expected uint8, got {loaded_data.dtype}"
            
            print("Test passed: HEIC load/save functionality works correctly.")
            
        except Exception as e:
            print(f"Test failed with error: {e}")
            raise


def test_heic_extension_support():
    """Test that HEIC extension is properly recognized."""
    # Test that HEIC is in the supported extensions
    assert 'heic' in npimage.core.supported_extensions, "HEIC extension not in supported extensions"
    
    # Test that the extension is properly parsed
    filename = "test_image.heic"
    extension = filename.split('.')[-1].lower()
    assert extension == 'heic', f"Extension parsing failed: expected 'heic', got '{extension}'"
    
    print("Test passed: HEIC extension is properly supported.")


if __name__ == '__main__':
    try:
        test_heic_extension_support()
        test_heic_load_save()
        print("All HEIC tests passed!")
    except Exception as e:
        print(f"HEIC tests failed: {e}")
        raise