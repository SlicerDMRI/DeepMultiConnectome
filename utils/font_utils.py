#!/usr/bin/env python3
"""
Font utility for matplotlib plots

This module provides a robust way to set matplotlib fonts with fallbacks
to avoid font warnings when Times New Roman is not available.
"""

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import warnings

def configure_plot_fonts(preferred_family='Times New Roman', size=16):
    """
    Configure matplotlib fonts with fallback options
    
    Args:
        preferred_family: Preferred font family (default: 'Times New Roman')
        size: Font size (default: 16)
    """
    
    # Check if the preferred font is available
    available_fonts = [f.name for f in fm.fontManager.ttflist]
    
    if preferred_family in available_fonts:
        font_family = preferred_family
    else:
        # Fallback fonts based on preference
        if 'serif' in preferred_family.lower() or 'times' in preferred_family.lower():
            # For serif fonts, try common serif alternatives
            serif_alternatives = ['DejaVu Serif', 'Liberation Serif', 'Bitstream Vera Serif', 'serif']
            font_family = None
            for font in serif_alternatives:
                if font in available_fonts or font == 'serif':
                    font_family = font
                    break
            if font_family is None:
                font_family = 'serif'
        else:
            # For sans-serif fonts, try common alternatives
            sans_alternatives = ['DejaVu Sans', 'Liberation Sans', 'Bitstream Vera Sans', 'sans-serif']
            font_family = None
            for font in sans_alternatives:
                if font in available_fonts or font == 'sans-serif':
                    font_family = font
                    break
            if font_family is None:
                font_family = 'sans-serif'
    
    # Configure matplotlib
    try:
        plt.rcParams.update({
            'font.family': font_family,
            'axes.labelsize': size,
            'xtick.labelsize': size,
            'ytick.labelsize': size,
            'legend.fontsize': size,
            'figure.titlesize': size + 2,
            'axes.titlesize': size + 1
        })
        print(f"✓ Configured plots with font: {font_family}")
    except Exception as e:
        # Ultimate fallback
        warnings.warn(f"Font configuration failed, using matplotlib defaults: {e}")
        plt.rcParams.update({
            'axes.labelsize': size,
            'xtick.labelsize': size,
            'ytick.labelsize': size,
            'legend.fontsize': size,
            'figure.titlesize': size + 2,
            'axes.titlesize': size + 1
        })


def get_available_fonts():
    """
    Get list of available fonts on the system
    
    Returns:
        List of available font names
    """
    return sorted([f.name for f in fm.fontManager.ttflist])


def check_font_availability(font_name):
    """
    Check if a specific font is available
    
    Args:
        font_name: Name of the font to check
        
    Returns:
        Boolean indicating if font is available
    """
    available_fonts = [f.name for f in fm.fontManager.ttflist]
    return font_name in available_fonts


if __name__ == "__main__":
    """Test the font configuration"""
    
    print("Testing font configuration...")
    
    # Test with Times New Roman
    print("\n1. Testing with Times New Roman:")
    configure_plot_fonts('Times New Roman')
    
    # Test with a non-existent font
    print("\n2. Testing with non-existent font:")
    configure_plot_fonts('NonExistentFont')
    
    # Show available fonts (first 20)
    print("\n3. Available fonts (first 20):")
    fonts = get_available_fonts()
    for i, font in enumerate(fonts[:20]):
        print(f"   {font}")
    
    print(f"\nTotal available fonts: {len(fonts)}")
    
    # Check specific fonts
    test_fonts = ['Times New Roman', 'DejaVu Sans', 'Liberation Sans', 'Arial']
    print("\n4. Font availability check:")
    for font in test_fonts:
        available = check_font_availability(font)
        status = "✓" if available else "✗"
        print(f"   {status} {font}")