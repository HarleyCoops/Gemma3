"""
Test Gemma Package

This script tests the gemma package to see what attributes and methods it has.
"""

import os
import sys
import inspect
import gemma

def print_module_info(module, indent=0):
    """Print information about a module and its attributes."""
    indent_str = "  " * indent
    
    # Print module attributes
    print(f"{indent_str}Module attributes:")
    for attr_name in dir(module):
        if not attr_name.startswith('__'):
            try:
                attr = getattr(module, attr_name)
                attr_type = type(attr).__name__
                print(f"{indent_str}- {attr_name} ({attr_type})")
                
                # If it's a module, recursively print its attributes
                if inspect.ismodule(attr):
                    print_module_info(attr, indent + 1)
            except Exception as e:
                print(f"{indent_str}- {attr_name} (Error: {str(e)})")

def main():
    """Test the gemma package."""
    print(f"Python version: {sys.version}")
    print(f"Gemma package version: {gemma.__version__}")
    print(f"Gemma package path: {gemma.__file__}")
    
    print("\nExploring gemma package:")
    print_module_info(gemma)
    
    # Try to import specific modules that might be used in text_generation.py
    print("\nTrying to import specific modules:")
    try:
        from gemma import nn
        print("Successfully imported gemma.nn")
        print("nn attributes:")
        for attr in dir(nn):
            if not attr.startswith('__'):
                print(f"- {attr}")
    except ImportError as e:
        print(f"Failed to import gemma.nn: {str(e)}")
    
    try:
        from gemma import ckpts
        print("Successfully imported gemma.ckpts")
        print("ckpts attributes:")
        for attr in dir(ckpts):
            if not attr.startswith('__'):
                print(f"- {attr}")
    except ImportError as e:
        print(f"Failed to import gemma.ckpts: {str(e)}")
    
    try:
        from gemma import text
        print("Successfully imported gemma.text")
        print("text attributes:")
        for attr in dir(text):
            if not attr.startswith('__'):
                print(f"- {attr}")
    except ImportError as e:
        print(f"Failed to import gemma.text: {str(e)}")
    
    try:
        from gemma import Tokenizer
        print("Successfully imported gemma.Tokenizer")
    except ImportError as e:
        print(f"Failed to import gemma.Tokenizer: {str(e)}")

if __name__ == "__main__":
    main() 