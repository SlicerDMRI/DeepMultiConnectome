#!/usr/bin/env python3
"""
Test the improved multi-subject analysis with a few subjects
"""

import sys
import os
from pathlib import Path

# Add path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from improved_multi_subject_analysis import ImprovedMultiSubjectAnalysis

def test_improved_analysis():
    """Test the improved analysis with 3 subjects"""
    
    # Create a temporary subject list
    test_subjects = ["100206", "100307", "100408"]
    test_file = Path("/tmp/test_subjects.txt")
    
    with open(test_file, 'w') as f:
        for subject in test_subjects:
            f.write(f"{subject}\n")
    
    print(f"Created test subject list: {test_file}")
    print(f"Testing with subjects: {test_subjects}")
    
    # Run analysis
    analysis = ImprovedMultiSubjectAnalysis(
        subject_list_file=str(test_file),
        max_subjects=3
    )
    
    print("\nRunning analysis...")
    analysis.run_analysis(n_processes=1)  # Use single process for testing
    
    print("\nTest completed!")


if __name__ == "__main__":
    test_improved_analysis()