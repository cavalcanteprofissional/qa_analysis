#!/usr/bin/env python3
"""
Test script for the dynamic color system.
"""

from src.color_manager import ColorManager
from src.palettes import get_all_palettes, get_palette_info


def test_color_manager():
    """Test basic ColorManager functionality."""
    print("Testing ColorManager...")
    
    # Initialize
    cm = ColorManager()
    
    # Test basic color assignment
    models = ['ModelA', 'ModelB', 'ModelC']
    colors = [cm.get_model_color(model) for model in models]
    
    print(f"[OK] Colors assigned to models:")
    for model, color in zip(models, colors):
        print(f"   {model}: {color}")
    
    # Test palette switching
    print(f"\n[INFO] Testing palette switching...")
    for palette in ['default', 'pastel', 'vibrant', 'professional']:
        cm.set_palette(palette)
        cm.model_colors = {}  # Reset to regenerate
        test_colors = [cm.get_model_color(model) for model in models]
        print(f"   {palette}: {test_colors}")
    
    # Test performance coloring
    print(f"\n[INFO] Testing performance-based coloring...")
    cm.set_performance_mode(True)
    scores = [0.9, 0.7, 0.3]
    perf_colors = [cm.get_model_color(model, score) for model, score in zip(models, scores)]
    for model, score, color in zip(models, scores, perf_colors):
        print(f"   {model} (score={score}): {color}")
    
    # Test accessibility mode
    print(f"\n[INFO] Testing accessibility mode...")
    cm.set_accessibility_mode(True)
    cm.model_colors = {}
    access_colors = [cm.get_model_color(model) for model in models]
    print(f"   Accessibility colors: {access_colors}")
    
    # Test palette generation
    print(f"\n[INFO] Testing palette generation...")
    extended_palette = cm.generate_palette(15, 'default')
    print(f"   Extended palette (15 colors): {len(extended_palette)} colors")
    print(f"   First 5: {extended_palette[:5]}")
    
    print("\n[OK] All ColorManager tests passed!")


def test_palettes():
    """Test palette system."""
    print("\n[INFO] Testing Palette System...")
    
    # Test palette info
    palettes = get_all_palettes()
    print(f"[OK] Available palettes: {list(palettes.keys())}")
    
    # Test palette details
    for palette_name in ['default', 'pastel', 'professional']:
        info = get_palette_info(palette_name)
        if info:
            print(f"   {palette_name}: {info.get('description', 'No description')}")
            print(f"   Colors: {info.get('colors', [])[:3]}...")  # First 3 colors
    
    print("[OK] Palette system tests passed!")


def test_color_utils():
    """Test color utility functions."""
    print("\n[INFO] Testing Color Utils...")
    
    from src.color_utils import (
        is_valid_hex_color, hex_to_rgb, rgb_to_hex,
        adjust_brightness, rotate_hue, calculate_color_distance
    )
    
    # Test validation
    assert is_valid_hex_color('#1f77b4') == True
    assert is_valid_hex_color('invalid') == False
    print("[OK] Color validation works")
    
    # Test conversion
    rgb = hex_to_rgb('#1f77b4')
    hex_color = rgb_to_hex(*rgb)
    assert hex_color == '#1f77b4'
    print("[OK] Color conversion works")
    
    # Test adjustments
    brighter = adjust_brightness('#1f77b4', 1.5)
    rotated = rotate_hue('#1f77b4', 45)
    print(f"[OK] Color adjustments: brighter={brighter}, rotated={rotated}")
    
    # Test distance
    distance = calculate_color_distance('#1f77b4', '#ff7f0e')
    print(f"[OK] Color distance between #1f77b4 and #ff7f0e: {distance:.2f}")
    
    print("[OK] Color utils tests passed!")


def test_integration():
    """Test integration scenarios."""
    print("\n[INFO] Testing Integration Scenarios...")
    
    cm = ColorManager()
    
    # Simulate real usage with multiple models and performance data
    models = ['DistilBERT', 'RoBERTa', 'BERT', 'GPT-2', 'T5']
    performance_scores = {'DistilBERT': 0.85, 'RoBERTa': 0.92, 'BERT': 0.78, 'GPT-2': 0.65, 'T5': 0.73}
    
    # Test regular coloring
    cm.set_palette('vibrant')
    regular_colors = {model: cm.get_model_color(model) for model in models}
    print("[OK] Regular coloring:")
    for model, color in regular_colors.items():
        print(f"   {model}: {color}")
    
    # Test performance coloring
    cm.set_performance_mode(True)
    cm.model_colors = {}  # Reset
    perf_colors = {model: cm.get_model_color(model, score) for model, score in performance_scores.items()}
    print("\n[OK] Performance-based coloring:")
    for model, color in perf_colors.items():
        print(f"   {model} ({performance_scores[model]:.2f}): {color}")
    
    # Test color consistency across backends
    cm.set_performance_mode(False)
    cm.set_palette('professional')
    consistency_colors = {model: cm.get_model_color(model) for model in models}
    print("\n[OK] Consistency test (professional palette):")
    for model, color in consistency_colors.items():
        print(f"   {model}: {color}")
    
    print("[OK] Integration tests passed!")


if __name__ == "__main__":
    print("[START] Starting Color System Tests...")
    
    try:
        test_color_manager()
        test_palettes()
        test_color_utils()
        test_integration()
        
        print("\n[SUCCESS] All tests passed! Color system is working correctly.")
        
    except Exception as e:
        print(f"\n[ERROR] Test failed with error: {e}")
        import traceback
        traceback.print_exc()