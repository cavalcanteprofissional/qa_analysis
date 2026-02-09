"""
Color utilities for the QA Analysis Dashboard.
Helper functions for color manipulation and validation.
"""

import colorsys
import re
from typing import List, Tuple, Optional


def is_valid_hex_color(hex_color: str) -> bool:
    """
    Check if a string is a valid hex color.
    
    Args:
        hex_color: Color string to validate
        
    Returns:
        True if valid hex color, False otherwise
    """
    hex_pattern = re.compile(r'^#([A-Fa-f0-9]{6}|[A-Fa-f0-9]{3})$')
    return bool(hex_pattern.match(hex_color))


def hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
    """
    Convert hex color to RGB tuple.
    
    Args:
        hex_color: Hex color string (e.g., "#1f77b4")
        
    Returns:
        Tuple of RGB values (0-255)
    """
    hex_color = hex_color.lstrip('#')
    if len(hex_color) == 3:  # Short form like "#abc"
        hex_color = ''.join([c*2 for c in hex_color])
    
    rgb_tuple = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    return (rgb_tuple[0], rgb_tuple[1], rgb_tuple[2])


def rgb_to_hex(r: int, g: int, b: int) -> str:
    """
    Convert RGB values to hex color.
    
    Args:
        r, g, b: RGB values (0-255)
        
    Returns:
        Hex color string
    """
    return f'#{r:02x}{g:02x}{b:02x}'


def hex_to_hsv(hex_color: str) -> Tuple[float, float, float]:
    """
    Convert hex color to HSV.
    
    Args:
        hex_color: Hex color string
        
    Returns:
        Tuple of HSV values (0-1 for all)
    """
    r, g, b = hex_to_rgb(hex_color)
    r_norm, g_norm, b_norm = r/255.0, g/255.0, b/255.0
    return colorsys.rgb_to_hsv(r_norm, g_norm, b_norm)


def hsv_to_hex(h: float, s: float, v: float) -> str:
    """
    Convert HSV to hex color.
    
    Args:
        h: Hue (0-1)
        s: Saturation (0-1)
        v: Value (0-1)
        
    Returns:
        Hex color string
    """
    r_norm, g_norm, b_norm = colorsys.hsv_to_rgb(h, s, v)
    r, g, b = int(r_norm*255), int(g_norm*255), int(b_norm*255)
    return rgb_to_hex(r, g, b)


def adjust_brightness(hex_color: str, factor: float) -> str:
    """
    Adjust brightness of a color.
    
    Args:
        hex_color: Original hex color
        factor: Brightness factor (>1 = brighter, <1 = darker)
        
    Returns:
        Adjusted hex color
    """
    h, s, v = hex_to_hsv(hex_color)
    v = min(1.0, v * factor)  # Ensure v doesn't exceed 1.0
    return hsv_to_hex(h, s, v)


def adjust_saturation(hex_color: str, factor: float) -> str:
    """
    Adjust saturation of a color.
    
    Args:
        hex_color: Original hex color
        factor: Saturation factor (>1 = more saturated, <1 = less)
        
    Returns:
        Adjusted hex color
    """
    h, s, v = hex_to_hsv(hex_color)
    s = min(1.0, s * factor)  # Ensure s doesn't exceed 1.0
    return hsv_to_hex(h, s, v)


def rotate_hue(hex_color: str, degrees: float) -> str:
    """
    Rotate the hue of a color.
    
    Args:
        hex_color: Original hex color
        degrees: Degrees to rotate (positive = clockwise)
        
    Returns:
        Color with rotated hue
    """
    h, s, v = hex_to_hsv(hex_color)
    h = (h + degrees / 360.0) % 1.0
    return hsv_to_hex(h, s, v)


def get_contrast_color(hex_color: str) -> str:
    """
    Get black or white color that contrasts best with the given color.
    
    Args:
        hex_color: Background color
        
    Returns:
        "#000000" (black) or "#ffffff" (white)
    """
    r, g, b = hex_to_rgb(hex_color)
    # Calculate luminance using standard formula
    luminance = (0.299 * r + 0.587 * g + 0.114 * b) / 255
    return "#000000" if luminance > 0.5 else "#ffffff"


def calculate_color_distance(color1: str, color2: str) -> float:
    """
    Calculate Euclidean distance between two colors in RGB space.
    
    Args:
        color1: First hex color
        color2: Second hex color
        
    Returns:
        Distance value (0-441.67, where 441.67 is max distance)
    """
    r1, g1, b1 = hex_to_rgb(color1)
    r2, g2, b2 = hex_to_rgb(color2)
    
    distance = ((r2 - r1)**2 + (g2 - g1)**2 + (b2 - b1)**2)**0.5
    return distance


def generate_color_variations(base_color: str, num_variations: int) -> List[str]:
    """
    Generate color variations from a base color.
    
    Args:
        base_color: Starting hex color
        num_variations: Number of variations to generate
        
    Returns:
        List of hex color variations
    """
    variations = [base_color]
    
    for i in range(1, num_variations):
        # Rotate hue and adjust brightness
        rotated = rotate_hue(base_color, (360 / num_variations) * i)
        brightness_factor = 0.8 + (0.4 * i / num_variations)  # Vary between 0.8 and 1.2
        adjusted = adjust_brightness(rotated, brightness_factor)
        variations.append(adjusted)
    
    return variations


def ensure_minimum_contrast(background: str, foreground: str, min_ratio: float = 4.5) -> str:
    """
    Ensure foreground color has minimum contrast ratio with background.
    
    Args:
        background: Background hex color
        foreground: Foreground hex color
        min_ratio: Minimum contrast ratio (WCAG AA standard is 4.5)
        
    Returns:
        Adjusted foreground color that meets contrast requirement
    """
    # This is a simplified version - full implementation would calculate actual contrast ratio
    # For now, we'll adjust brightness to ensure better contrast
    bg_luminance = sum(hex_to_rgb(background)) / 3 / 255
    fg_luminance = sum(hex_to_rgb(foreground)) / 3 / 255
    
    if abs(bg_luminance - fg_luminance) < 0.3:  # Too similar
        if bg_luminance > 0.5:
            return "#000000"  # Use black for light backgrounds
        else:
            return "#ffffff"  # Use white for dark backgrounds
    
    return foreground


def create_gradient_colors(start_color: str, end_color: str, steps: int) -> List[str]:
    """
    Create a gradient of colors between two endpoints.
    
    Args:
        start_color: Starting hex color
        end_color: Ending hex color
        steps: Number of colors in the gradient
        
    Returns:
        List of hex colors forming a gradient
    """
    gradient = []
    
    # Convert to HSV for smoother transitions
    h1, s1, v1 = hex_to_hsv(start_color)
    h2, s2, v2 = hex_to_hsv(end_color)
    
    for i in range(steps):
        # Linear interpolation in HSV space
        t = i / (steps - 1) if steps > 1 else 0
        
        # Handle hue wrapping (for shortest path)
        if abs(h2 - h1) <= 0.5:
            h = h1 + t * (h2 - h1)
        else:
            if h1 > h2:
                h = h1 + t * ((h2 + 1.0) - h1)
                h = h % 1.0
            else:
                h = h1 + t * ((h1 + 1.0) - h2)
                h = (h + 1.0) % 1.0
        
        s = s1 + t * (s2 - s1)
        v = v1 + t * (v2 - v1)
        
        gradient.append(hsv_to_hex(h, s, v))
    
    return gradient


def validate_palette(colors: List[str]) -> Tuple[bool, List[str]]:
    """
    Validate a list of colors as a palette.
    
    Args:
        colors: List of hex color strings
        
    Returns:
        Tuple of (is_valid, error_messages)
    """
    errors = []
    
    if not colors:
        errors.append("Palette cannot be empty")
        return False, errors
    
    if len(colors) < 2:
        errors.append("Palette must have at least 2 colors")
    
    for i, color in enumerate(colors):
        if not is_valid_hex_color(color):
            errors.append(f"Color {i+1} ('{color}') is not a valid hex color")
    
    # Check for duplicate colors
    unique_colors = set(colors)
    if len(unique_colors) != len(colors):
        errors.append("Palette contains duplicate colors")
    
    # Check for sufficient contrast between adjacent colors
    for i in range(len(colors) - 1):
        distance = calculate_color_distance(colors[i], colors[i+1])
        if distance < 30:  # Arbitrary minimum distance
            errors.append(f"Colors {i+1} and {i+2} are too similar")
    
    return len(errors) == 0, errors