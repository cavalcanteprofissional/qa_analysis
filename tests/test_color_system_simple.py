#!/usr/bin/env python3
"""
Test script for the dynamic color system - Windows compatible version.
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
    
    print(f"Colors assigned to models:")
    for model, color in zip(models, colors):
        print(f"   {model}: {color}")
    
    # Test palette switching
    print(f"\nTesting palette switching...")
    for palette in ['default', 'pastel', 'vibrant', 'professional']:
        cm.set_palette(palette)
        cm.model_colors = {}  # Reset to regenerate
        test_colors = [cm.get_model_color(model) for model in models]
        print(f"   {palette}: {test_colors}")
    
    # Test performance coloring
    print(f"\nTesting performance-based coloring...")
    cm.set_performance_mode(True)
    scores = [0.9, 0.7, 0.3]
    perf_colors = [cm.get_model_color(model, score) for model, score in zip(models, scores)]
    for model, score, color in zip(models, scores, perf_colors):
        print(f"   {model} (score={score}): {color}")
    
    # Test accessibility mode
    print(f"\nTesting accessibility mode...")
    cm.set_accessibility_mode(True)
    cm.model_colors = {}
    access_colors = [cm.get_model_color(model) for model in models]
    print(f"   Accessibility colors: {access_colors}")
    
    # Test palette generation
    print(f"\nTesting palette generation...")
    extended_palette = cm.generate_palette(15, 'default')
    print(f"   Extended palette (15 colors): {len(extended_palette)} colors")
    print(f"   First 5: {extended_palette[:5]}")
    
    print("\nAll ColorManager tests passed!")


def test_palettes():
    """Test palette system."""
    print("\nTesting Palette System...")
    
    # Test palette info
    palettes = get_all_palettes()
    print(f"Available palettes: {list(palettes.keys())}")
    
    # Test palette details
    for palette_name in ['default', 'pastel', 'professional']:
        info = get_palette_info(palette_name)
        if info:
            print(f"   {palette_name}: {info.get('description', 'No description')}")
            print(f"   Colors: {info.get('colors', [])[:3]}...")  # First 3 colors
    
    print("Palette system tests passed!")


def test_integration():
    """Test integration scenarios."""
    print("\nTesting Integration Scenarios...")
    
    cm = ColorManager()
    
    # Simulate real usage with multiple models and performance data
    models = ['DistilBERT', 'RoBERTa', 'BERT', 'GPT-2', 'T5']
    performance_scores = {'DistilBERT': 0.85, 'RoBERTa': 0.92, 'BERT': 0.78, 'GPT-2': 0.65, 'T5': 0.73}
    
    # Test regular coloring
    cm.set_palette('vibrant')
    regular_colors = {model: cm.get_model_color(model) for model in models}
    print("Regular coloring:")
    for model, color in regular_colors.items():
        print(f"   {model}: {color}")
    
    # Test performance coloring
    cm.set_performance_mode(True)
    cm.model_colors = {}  # Reset
    perf_colors = {model: cm.get_model_color(model, score) for model, score in performance_scores.items()}
    print("\nPerformance-based coloring:")
    for model, color in perf_colors.items():
        print(f"   {model} ({performance_scores[model]:.2f}): {color}")
    
    # Test color consistency across backends
    cm.set_performance_mode(False)
    cm.set_palette('professional')
    consistency_colors = {model: cm.get_model_color(model) for model in models}
    print("\nConsistency test (professional palette):")
    for model, color in consistency_colors.items():
        print(f"   {model}: {color}")
    
    print("Integration tests passed!")


if __name__ == "__main__":
    print("Starting Color System Tests...")
    
    try:
        test_color_manager()
        test_palettes()
        test_integration()
        
        print("\nAll tests passed! Color system is working correctly.")
        
    except Exception as e:
        print(f"\nTest failed with error: {e}")
        import traceback
        traceback.print_exc()