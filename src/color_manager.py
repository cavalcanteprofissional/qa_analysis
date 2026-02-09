import streamlit as st
import colorsys
from typing import List, Dict, Optional
import json


class ColorManager:
    """Advanced color management system with full functionality."""
    
    def __init__(self):
        # Fixed palettes
        self.palettes = {
            'default': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'],
            'pastel': ['#AEC7E8', '#FFBB78', '#98DF8A', '#FF9896', '#C5B0D5', '#FF9896', '#C49E94', '#F7B6D2', '#C7C7C7', '#DBDB8D'],
            'vibrant': ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD', '#98D8C8', '#F7DC6F', '#BB8FCE', '#85C1E2'],
            'professional': ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6C4A6D', '#3D5A80', '#98C1D9', '#E0FBFC', '#EE6C4D', '#293241'],
            'monochrome': ['#1f1f1f', '#404040', '#666666', '#8c8c8c', '#b3b3b3', '#cccccc', '#000000', '#333333', '#666666', '#999999'],
            'colorblind': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        }
        
        self.current_palette = 'default'
        self.model_colors = {}
        self.custom_palettes = {}
        self.accessibility_mode = False
        self.performance_mode = False
        self._performance_colors = {
            'excellent': '#2ca02c',  # Verde
            'good': '#ff7f0e',       # Laranja  
            'average': '#d62728',    # Vermelho
            'poor': '#8c564b'        # Marrom
        }
        
    def get_model_color(self, model_name: str, performance_score: Optional[float] = None) -> str:
        """Get color for model with optional performance-based coloring."""
        
        # Performance-based coloring takes priority
        if self.performance_mode and performance_score is not None:
            return self.get_performance_color(model_name, performance_score)
        
        # Return existing custom color
        if model_name in self.model_colors:
            return self.model_colors[model_name]
        
        # Generate new color from current palette
        models = sorted(list(self.model_colors.keys()) + [model_name])
        index = models.index(model_name)
        palette = self.generate_palette(index + 1, self.current_palette)
        color = palette[index]
        
        # Apply accessibility mode if enabled
        if self.accessibility_mode:
            color = self.apply_accessibility(color)
        
        self.model_colors[model_name] = color
        return color
    
    def get_performance_color(self, model_name: str, score: float) -> str:
        """Get color based on model performance."""
        if score >= 0.8:
            return self._performance_colors.get('excellent', '#2ca02c')
        elif score >= 0.6:
            return self._performance_colors.get('good', '#ff7f0e')
        elif score >= 0.4:
            return self._performance_colors.get('average', '#d62728')
        else:
            return self._performance_colors.get('poor', '#8c564b')
    
    def generate_palette(self, num_colors: int, palette_name: Optional[str] = None) -> List[str]:
        """Generate extended palette for unlimited models."""
        palette_name = palette_name or self.current_palette
        base_palette = self.palettes.get(palette_name, self.palettes['default'])
        
        if num_colors <= len(base_palette):
            return base_palette[:num_colors]
        
        # Generate additional colors using HSV rotation
        extended = base_palette.copy()
        while len(extended) < num_colors:
            base_color = extended[len(extended) % len(base_palette)]
            new_color = self._rotate_hue(base_color, 30)  # 30 degree rotation
            extended.append(new_color)
        
        return extended[:num_colors]
    
    def _rotate_hue(self, hex_color: str, degrees: int) -> str:
        """Rotate hue of a hex color by specified degrees."""
        # Convert hex to RGB
        hex_color = hex_color.lstrip('#')
        r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        
        # Convert RGB to HSV
        r_norm, g_norm, b_norm = r/255.0, g/255.0, b/255.0
        h, s, v = colorsys.rgb_to_hsv(r_norm, g_norm, b_norm)
        
        # Rotate hue
        h = (h + degrees/360.0) % 1.0
        
        # Convert back to RGB
        r_norm, g_norm, b_norm = colorsys.hsv_to_rgb(h, s, v)
        r, g, b = int(r_norm*255), int(g_norm*255), int(b_norm*255)
        
        # Convert to hex
        return f'#{r:02x}{g:02x}{b:02x}'
    
    def apply_accessibility(self, hex_color: str) -> str:
        """Apply accessibility-friendly modifications to increase contrast."""
        hex_color = hex_color.lstrip('#')
        r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        
        # Increase brightness/darkness for better contrast
        # Simple approach: if average color is dark, make it lighter; if light, make it darker
        avg = (r + g + b) / 3
        
        if avg < 128:  # Dark color - make it lighter
            factor = 1.3
        else:  # Light color - make it darker
            factor = 0.7
        
        r = min(255, int(r * factor))
        g = min(255, int(g * factor))
        b = min(255, int(b * factor))
        
        return f'#{r:02x}{g:02x}{b:02x}'
    
    def save_custom_palette(self, name: str, colors: List[str]) -> None:
        """Save user-defined custom palette."""
        self.custom_palettes[name] = colors
        self.save_to_session_state()
    
    def get_color_map(self, models: List[str], performance_scores: Optional[Dict] = None) -> Dict[str, str]:
        """Get complete color mapping for all models."""
        color_map = {}
        for model in models:
            score = performance_scores.get(model) if performance_scores else None
            color_map[model] = self.get_model_color(model, score)
        return color_map
    
    def reset_colors(self) -> None:
        """Reset all color assignments to defaults."""
        self.model_colors = {}
        self.current_palette = 'default'
        self.accessibility_mode = False
        self.performance_mode = False
        self.save_to_session_state()
    
    def set_performance_mode(self, enabled: bool) -> None:
        """Enable or disable performance-based coloring."""
        self.performance_mode = enabled
        self.save_to_session_state()
    
    def set_accessibility_mode(self, enabled: bool) -> None:
        """Enable or disable accessibility mode."""
        self.accessibility_mode = enabled
        self.save_to_session_state()
    
    def set_palette(self, palette_name: str) -> None:
        """Set the current palette and regenerate model colors."""
        if palette_name in self.palettes or palette_name in self.custom_palettes:
            self.current_palette = palette_name
            self.model_colors = {}  # Reset to regenerate with new palette
            self.save_to_session_state()
    
    def save_to_session_state(self) -> None:
        """Save preferences to Streamlit session state."""
        if 'color_preferences' not in st.session_state:
            st.session_state.color_preferences = {}
        
        st.session_state.color_preferences.update({
            'current_palette': self.current_palette,
            'model_colors': self.model_colors,
            'custom_palettes': self.custom_palettes,
            'accessibility_mode': self.accessibility_mode,
            'performance_mode': self.performance_mode
        })
    
    def load_from_session_state(self) -> None:
        """Load preferences from Streamlit session state."""
        prefs = st.session_state.get('color_preferences', {})
        self.current_palette = prefs.get('current_palette', 'default')
        self.model_colors = prefs.get('model_colors', {})
        self.custom_palettes = prefs.get('custom_palettes', {})
        self.accessibility_mode = prefs.get('accessibility_mode', False)
        self.performance_mode = prefs.get('performance_mode', False)
    
    def get_all_palette_options(self) -> List[str]:
        """Get all available palette options (built-in + custom)."""
        return list(self.palettes.keys()) + list(self.custom_palettes.keys())
    
    def update_model_color(self, model_name: str, color: str) -> None:
        """Manually update a model's color."""
        self.model_colors[model_name] = color
        self.save_to_session_state()
    
    def get_model_performance_stats(self, df, model_col, score_col) -> Dict[str, float]:
        """Get average performance scores for each model."""
        if model_col not in df.columns or score_col not in df.columns:
            return {}
        
        return df.groupby(model_col)[score_col].mean().to_dict()