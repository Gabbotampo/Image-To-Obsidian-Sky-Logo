# Image-To-Obsidian-Sky-Logo
A Python script that converts an image into a Minecraft schematic using obsidian and crying obsidian, preserving transparency and allowing easy import into Litematica for sky-logos.


1. How the Python Script Works

The script converts an image into a Minecraft schematic with specific blocks.

How it works:

Open the image

Reads it in RGBA (red, green, blue, alpha).

Transparent pixels become air in the schematic.

Each pixel is equal to a block so for smaller logo just leave a low resolutuion

Color recognition

Black (0, 0, 0) → obsidian

Purple #9F81D6 → crying obsidian

All other pixels → air

Important: To have crying obsidian appear in the logo exactly where you want, the pixels must be colored #9F81D6 in your image.

Block placement

Schematic is horizontal (X-Z plane), Y=0.

Saving

Saved in the output folder as sky_logo.schem.

2. How to Use the Script

Make sure Python is installed.

Install the libraries:

pip install pillow mcschematic


Image file

The image must be named logo.png and placed in the same folder as the script.

Run the script:

python program.py


The schematic will be in output/sky_logo.schem.

Supported colors:

Black → obsidian

Purple #9F81D6 → crying obsidian

Transparent → air

3. Importing the Schematic in Amulet

Create a creative superflat world (Empty preset).

Open it in Amulet.

Go to import → Import file and select sky_logo.schem.

Place it where you want, then save the world.

Transparent pixels stay empty, no unwanted blocks appear.

4. Create a Litematic for Litematica

Open Minecraft with Litematica.

Load the world with the imported schematic.

Open M → Area Selection, select the schematic → Save → Litematic.

You can now place it automatically in creative mode.

5. Tips

Keep the schematic horizontal for logos.

Transparent areas in the image correspond to air.

Avoid very large images for better performance.

Pixels for crying obsidian must be exactly #9F81D6 for it to appear correctly.

If you don't understend something write me on discord (gabbotampo).
