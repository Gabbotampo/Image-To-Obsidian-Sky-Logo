# Image-To-Obsidian-Sky-Logo

Image-To-Obsidian-Sky-Logo is a Python-based tool that converts images into Minecraft schematics. It processes PNG/JPG images, automatically scales them based on pixel density, corrects small structural gaps, and maps image colors to Minecraft blocks. The final output can be exported as a .litematic file (or .schem as a fallback).

Features
Converts PNG and JPG images into Minecraft schematics
Automatic scaling based on target number of non-white pixels
Image preprocessing with gap correction for improved structural quality
Color-to-block mapping with Euclidean distance matching
Direct .litematic export using litemapy
Optional .schem export fallback using mcschematic
Optimized processing using NumPy for large images
Requirements

Install the required dependencies:

pip install Pillow litemapy numpy

Optional fallback support:

pip install mcschematic
How It Works

SkyLogo Generator follows a multi-stage pipeline:

Loads an image from the current working directory
Computes the number of non-white pixels (image density estimation)
Scales the image to match the requested target pixel count
Applies gap correction to improve visual continuity
Maps each pixel to the closest Minecraft block color
Generates a Minecraft schematic file (.litematic or .schem)
Block Mapping System

The tool uses a configurable color palette defined in COLOR_TO_BLOCK.

Default mappings include:

Obsidian: dark tones and near-black colors
Crying Obsidian: purple shades around #9F81D6

The system uses Euclidean distance in RGB space to determine the closest block match.

How it works
Color recognition:

Black (#000000) = obsidian
Purple (#9G81D6 it MUST be this one) = crying obsidian
Any other pixel including trasparent ones = air

Usage
Run the script:

python skylogo_generator.py

Then follow the interactive prompts:

Enter the desired number of non-white pixels
Select an image if multiple are present in the directory
Wait for processing to complete
Retrieve the output file from the output/ directory

Output
All generated schematics are saved in:

output/

Example output file:

output/logo_sky.litematic
Fallback Mode

If litemapy is not installed or fails during execution, the program automatically attempts to generate a .schem file using mcschematic.

Performance Considerations
Processing time increases with image resolution
NumPy-based vectorization is used for efficiency
Large images may require significant memory and processing time
Block placement is optimized in batch operations where possible
Limitations
Only supports flat (Y = 0) schematic generation
Limited default block palette (extendable manually)
Very large images may lead to high memory usage
No built-in GUI (CLI only)
Future Improvements
3D schematic support (multi-layer depth generation)
Extended and biome-aware block palettes
Graphical user interface (GUI)
Real-time preview of generated schematics
Advanced dithering for improved visual fidelity
License

This project is free to use, modify, and distribute. Attribution is appreciated but not required.

If you don't understend something write me on discord (gabbotampo).
