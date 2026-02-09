"""
Predefined color palettes for the QA Analysis Dashboard.
High-quality color schemes optimized for data visualization.
"""

# Category: Default/Standard Palettes
DEFAULT_PALETTES = {
    'default': {
        'name': 'Plotly Default',
        'colors': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                   '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'],
        'description': 'Standard Plotly color sequence',
        'usage': 'General purpose, good contrast'
    },
    
    'pastel': {
        'name': 'Soft Pastels',
        'colors': ['#AEC7E8', '#FFBB78', '#98DF8A', '#FF9896', '#C5B0D5',
                   '#FF9896', '#C49E94', '#F7B6D2', '#C7C7C7', '#DBDB8D'],
        'description': 'Soft, muted colors suitable for professional presentations',
        'usage': 'Corporate presentations, print-friendly'
    },
    
    'vibrant': {
        'name': 'Vibrant Energy',
        'colors': ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7',
                   '#DDA0DD', '#98D8C8', '#F7DC6F', '#BB8FCE', '#85C1E2'],
        'description': 'High-contrast, energetic colors',
        'usage': 'Modern dashboards, interactive visualizations'
    },
    
    'professional': {
        'name': 'Corporate Professional',
        'colors': ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6C4A6D',
                   '#3D5A80', '#98C1D9', '#E0FBFC', '#EE6C4D', '#293241'],
        'description': 'Business-friendly professional colors',
        'usage': 'Business analytics, executive dashboards'
    },
    
    'monochrome': {
        'name': 'Monochrome Grayscale',
        'colors': ['#1f1f1f', '#404040', '#666666', '#8c8c8c', '#b3b3b3',
                   '#cccccc', '#000000', '#333333', '#666666', '#999999'],
        'description': 'Grayscale variations for formal documents',
        'usage': 'Academic papers, black and white printing'
    },
    
    'colorblind': {
        'name': 'Colorblind Friendly',
        'colors': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                   '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'],
        'description': 'Optimized for colorblind accessibility (simulated)',
        'usage': 'Accessible dashboards, diverse audiences'
    }
}

# Category: Specialized Palettes
SPECIALIZED_PALETTES = {
    'sequential': {
        'name': 'Sequential Blue',
        'colors': ['#f7fbff', '#deebf7', '#c6dbef', '#9ecae1', '#6baed6',
                   '#4292c6', '#2171b5', '#08519c', '#08306b'],
        'description': 'Single-hue progression for ordered data',
        'usage': 'Heat maps, performance metrics'
    },
    
    'diverging': {
        'name': 'Diverging Red-Blue',
        'colors': ['#67001f', '#b2182b', '#d6604d', '#f4a582', '#fddbc7',
                   '#f7f7f7', '#d1e5f0', '#92c5de', '#4393c3', '#2166ac', '#053061'],
        'description': 'Two-hue diverging for centered data',
        'usage': 'Deviation plots, comparisons to baseline'
    },
    
    'qualitative': {
        'name': 'Qualitative Research',
        'colors': ['#8dd3c7', '#ffffb3', '#bebada', '#fb8072', '#80b1d3',
                   '#fdb462', '#b3de69', '#fccde5', '#d9d9d9', '#bc80bd'],
        'description': 'Optimized for categorical data',
        'usage': 'Category comparisons, classification'
    }
}

# Category: Performance-Based Palettes
PERFORMANCE_PALETTES = {
    'performance': {
        'name': 'Performance Scale',
        'colors': {
            'excellent': '#2ca02c',  # Green
            'good': '#ff7f0e',       # Orange  
            'average': '#d62728',    # Red
            'poor': '#8c564b'        # Brown
        },
        'description': 'Color coding based on performance levels',
        'usage': 'Model performance visualization',
        'mapping': {
            'excellent': 'Score ≥ 0.8',
            'good': 'Score 0.6-0.79',
            'average': 'Score 0.4-0.59',
            'poor': 'Score < 0.4'
        }
    },
    
    'traffic_light': {
        'name': 'Traffic Light System',
        'colors': {
            'go': '#00ff00',          # Bright green
            'caution': '#ffff00',     # Yellow
            'stop': '#ff0000'         # Red
        },
        'description': 'Universal traffic light colors',
        'usage': 'Quick status indicators',
        'mapping': {
            'go': 'Good performance',
            'caution': 'Moderate performance',
            'stop': 'Poor performance'
        }
    }
}

# Category: Theme-Based Palettes
THEME_PALETTES = {
    'nature': {
        'name': 'Nature Inspired',
        'colors': ['#2d5016', '#73a942', '#aad576', '#d4e4bc', '#f7f7f7'],
        'description': 'Earthy, natural tones',
        'usage': 'Environmental data, organic themes'
    },
    
    'ocean': {
        'name': 'Ocean Depths',
        'colors': ['#08519c', '#3182bd', '#6baed6', '#9ecae1', '#deebf7'],
        'description': 'Blue ocean gradient',
        'usage': 'Water-related data, cooling systems'
    },
    
    'sunset': {
        'name': 'Sunset Warmth',
        'colors': ['#54278f', '#998ec3', '#d8b365', '#fec44f', '#fee391'],
        'description': 'Warm sunset colors',
        'usage': 'Energy metrics, temperature data'
    },
    
    'forest': {
        'name': 'Forest Canopy',
        'colors': ['#00441b', '#006d2c', '#238b45', '#41ab5d', '#74c476'],
        'description': 'Rich green forest colors',
        'usage': 'Growth metrics, environmental data'
    }
}

# Category: Accessibility Palettes
ACCESSIBILITY_PALETTES = {
    'high_contrast': {
        'name': 'High Contrast',
        'colors': ['#000000', '#ffffff', '#ff0000', '#00ff00', '#0000ff',
                   '#ffff00', '#ff00ff', '#00ffff', '#ff8800', '#8800ff'],
        'description': 'Maximum contrast for accessibility',
        'usage': 'Vision-impaired users, presentation projectors'
    },
    
    'daltonism': {
        'name': 'Colorblind Safe',
        'colors': ['#1b9e77', '#d95f02', '#7570b3', '#e7298a', '#66a61e',
                   '#e6ab02', '#a6761d', '#666666', '#999999', '#cccccc'],
        'description': 'Colorblind-friendly palette',
        'usage': 'Public dashboards, diverse audiences'
    }
}

# Utility functions
def get_palette_by_category(category: str) -> dict:
    """Get all palettes from a specific category."""
    categories = {
        'default': DEFAULT_PALETTES,
        'specialized': SPECIALIZED_PALETTES,
        'performance': PERFORMANCE_PALETTES,
        'theme': THEME_PALETTES,
        'accessibility': ACCESSIBILITY_PALETTES
    }
    return categories.get(category, {})


def get_all_palettes() -> dict:
    """Get all available palettes."""
    all_palettes = {}
    for category_dict in [DEFAULT_PALETTES, SPECIALIZED_PALETTES, 
                         PERFORMANCE_PALETTES, THEME_PALETTES, 
                         ACCESSIBILITY_PALETTES]:
        all_palettes.update(category_dict)
    return all_palettes


def get_palette_colors(palette_name: str) -> list:
    """Get color list from a palette name."""
    all_palettes = get_all_palettes()
    
    if palette_name in all_palettes:
        palette = all_palettes[palette_name]
        if isinstance(palette.get('colors'), dict):
            # Performance palette with levels
            return list(palette['colors'].values())
        else:
            # Regular color list
            return palette['colors']
    
    # Fallback to default palette
    return DEFAULT_PALETTES['default']['colors']


def get_palette_info(palette_name: str) -> dict:
    """Get detailed information about a palette."""
    all_palettes = get_all_palettes()
    return all_palettes.get(palette_name, {})


def validate_palette_selection(palette_name: str) -> bool:
    """Check if a palette name is valid."""
    return palette_name in get_all_palettes()


def get_palette_recommendations(use_case: str) -> list:
    """Get recommended palettes for specific use cases."""
    recommendations = {
        'presentation': ['pastel', 'professional', 'high_contrast'],
        'interactive': ['vibrant', 'default', 'ocean'],
        'print': ['monochrome', 'pastel', 'daltonism'],
        'accessibility': ['colorblind', 'daltonism', 'high_contrast'],
        'performance': ['performance', 'traffic_light'],
        'business': ['professional', 'default', 'pastel'],
        'research': ['qualitative', 'sequential', 'diverging']
    }
    
    return recommendations.get(use_case, ['default'])


# Color meaning mappings for performance palettes
PERFORMANCE_MAPPINGS = {
    'performance': {
        'excellent': {'range': '≥ 0.8', 'description': 'Outstanding performance'},
        'good': {'range': '0.6 - 0.79', 'description': 'Good performance'},
        'average': {'range': '0.4 - 0.59', 'description': 'Average performance'},
        'poor': {'range': '< 0.4', 'description': 'Poor performance'}
    },
    'traffic_light': {
        'go': {'range': 'Good', 'description': 'Green light - proceed'},
        'caution': {'range': 'Moderate', 'description': 'Yellow light - be careful'},
        'stop': {'range': 'Poor', 'description': 'Red light - needs attention'}
    }
}