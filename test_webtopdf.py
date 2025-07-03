import os
from webtopdf import webpage_to_pdf, advanced_webpage_to_pdf

def test_basic_conversion():
    """Test basic webpage to PDF conversion"""
    url = "https://github.com/microsoft/playwright-python"  # More complex website
    output_path = "test_basic.pdf"
    
    # Clean up any existing test file
    if os.path.exists(output_path):
        os.remove(output_path)
    
    # Run conversion
    webpage_to_pdf(url, output_path, wait_time=10)  # Increased wait time for complex site
    
    # Verify file was created
    assert os.path.exists(output_path), f"PDF file {output_path} was not created"
    assert os.path.getsize(output_path) > 0, f"PDF file {output_path} is empty"
    print("Basic conversion test passed!")

def test_advanced_conversion():
    """Test advanced webpage to PDF conversion with custom options"""
    url = "https://github.com/microsoft/playwright-python"  # More complex website
    output_path = "test_advanced.pdf"
    
    # Clean up any existing test file
    if os.path.exists(output_path):
        os.remove(output_path)
    
    # Custom options
    custom_options = {
        'wait_time': 10,  # Increased wait time
        'viewport': {'width': 1920, 'height': 1080},
        'pdf_options': {
            'format': 'A4',
            'landscape': True,
            'print_background': True,
            'margin': {'top': '2cm', 'bottom': '2cm', 'left': '2cm', 'right': '2cm'},
            'scale': 0.8  # Slightly smaller scale to fit more content
        }
    }
    
    # Run conversion
    advanced_webpage_to_pdf(url, output_path, custom_options)
    
    # Verify file was created
    assert os.path.exists(output_path), f"PDF file {output_path} was not created"
    assert os.path.getsize(output_path) > 0, f"PDF file {output_path} is empty"
    print("Advanced conversion test passed!")

if __name__ == "__main__":
    print("Running web-to-PDF conversion tests with complex website...")
    test_basic_conversion()
    test_advanced_conversion()
    print("All tests passed!") 