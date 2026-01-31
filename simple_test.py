"""
Simple test for QA Analysis Dashboard
Tests basic functionality without unicode characters
"""

import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

def test_imports():
    """Test if all imports work correctly"""
    print("Testing imports...")
    
    try:
        import streamlit as st
        print("Streamlit imported successfully")
    except Exception as e:
        print(f"Streamlit import failed: {e}")
        return False
    
    try:
        import pandas as pd
        print("Pandas imported successfully")
    except Exception as e:
        print(f"Pandas import failed: {e}")
        return False
    
    try:
        import numpy as np
        print("NumPy imported successfully")
    except Exception as e:
        print(f"NumPy import failed: {e}")
        return False
    
    try:
        from config.settings import Config
        print("Config imported successfully")
    except Exception as e:
        print(f"Config import failed: {e}")
        return False
    
    try:
        from utils.data_manager import DataManager
        print("DataManager imported successfully")
    except Exception as e:
        print(f"DataManager import failed: {e}")
        return False
    
    try:
        from utils.parallel_processor import ParallelProcessingManager
        print("ParallelProcessingManager imported successfully")
    except Exception as e:
        print(f"ParallelProcessingManager import failed: {e}")
        return False
    
    try:
        from utils.import_export import ImportExportManager
        print("ImportExportManager imported successfully")
    except Exception as e:
        print(f"ImportExportManager import failed: {e}")
        return False
    
    return True

def test_config():
    """Test configuration setup"""
    print("\nTesting configuration...")
    
    try:
        from config.settings import Config
        
        print(f"Data directory: {Config.DATA_DIR}")
        print(f"Intervalos directory: {Config.INTERVALOS_DIR}")
        print(f"Output directory: {Config.OUTPUT_DIR}")
        
        # Test directory creation
        Config.setup_dirs()
        print("Directory creation successful")
        
        return True
        
    except Exception as e:
        print(f"Configuration test failed: {e}")
        return False

def test_data_manager():
    """Test data manager functionality"""
    print("\nTesting data manager...")
    
    try:
        from utils.data_manager import DataManager
        
        # Create data manager
        dm = DataManager()
        print("DataManager created successfully")
        
        # Test cache
        cache_info = dm.get_cache_info()
        print(f"Cache info: {cache_info['total_cached_items']} items")
        
        return True
        
    except Exception as e:
        print(f"Data manager test failed: {e}")
        return False

def test_parallel_processor():
    """Test parallel processing manager"""
    print("\nTesting parallel processor...")
    
    try:
        from utils.parallel_processor import ParallelProcessingManager
        
        # Create manager
        ppm = ParallelProcessingManager()
        print("ParallelProcessingManager created successfully")
        
        # Test model initialization (without actual models)
        print("ParallelProcessingManager ready for model loading")
        
        return True
        
    except Exception as e:
        print(f"Parallel processor test failed: {e}")
        return False

def test_import_export():
    """Test import/export manager"""
    print("\nTesting import/export manager...")
    
    try:
        from utils.import_export import ImportExportManager
        
        # Create manager
        iem = ImportExportManager()
        print("ImportExportManager created successfully")
        
        # Test file detection
        browse_results = iem.browse_analysis_files()
        print(f"File browsing works: {browse_results['summary']['total_files']} files found")
        
        return True
        
    except Exception as e:
        print(f"Import/export test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("=" * 60)
    print("QA Analysis Dashboard - Pipeline Test")
    print("=" * 60)
    
    tests = [
        ("Import Test", test_imports),
        ("Configuration Test", test_config),
        ("Data Manager Test", test_data_manager),
        ("Parallel Processor Test", test_parallel_processor),
        ("Import/Export Test", test_import_export)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\nRunning {test_name}...")
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"{test_name} crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = 0
    for test_name, success in results:
        status = "PASS" if success else "FAIL"
        print(f"{status} {test_name}")
        if success:
            passed += 1
    
    print(f"\nResults: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("All tests passed! Pipeline is ready.")
        print("\nTo start the dashboard:")
        print("   poetry run streamlit run streamlit_app.py")
    else:
        print("Some tests failed. Check the errors above.")
    
    return passed == len(results)

if __name__ == "__main__":
    main()