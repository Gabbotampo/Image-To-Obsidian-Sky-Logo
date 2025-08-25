from PIL import Image
import mcschematic
import os

# Palette colore → blocco con sfumature
COLOR_TO_BLOCK = {
    "obsidian": [(0, 0, 0)],          # nero → ossidiana normale
    "crying_obsidian": [
        (159, 129, 214),              # viola dall'immagine #9F81D6
    ]
}

OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def get_block_from_color(color, tolerance=20):
    """Restituisce il blocco corrispondente al colore oppure None se sfondo"""
    for block, colors in COLOR_TO_BLOCK.items():
        for c in colors:
            if all(abs(color[i] - c[i]) <= tolerance for i in range(3)):
                return f"minecraft:{block}"
    return None  # pixel non corrispondente → vuoto

def image_to_schematic(image_path, output_name="logo_sky", use_original_size=True, size=None):
    img = Image.open(image_path).convert("RGBA")  # Usa RGBA per trasparenza
    if not use_original_size and size:
        img = img.resize(size, Image.NEAREST)

    width, height = img.size
    schem = mcschematic.MCSchematic()

    for y in range(height):
        for x in range(width):
            r, g, b, a = img.getpixel((x, y))
            
            # Pixel trasparenti → vuoto
            if a == 0:
                continue

            block = get_block_from_color((r, g, b))
            if block:
                schem.setBlock((x, 0, y), block)  # X-Z orizzontale, Y=0

    schem.save(OUTPUT_DIR, output_name, mcschematic.Version.JE_1_20)
    print(f"Schematic salvato in: {OUTPUT_DIR}/{output_name}.schem")
    print(f"Dimensioni: {width}x{height} blocchi (XxZ)")

if __name__ == "__main__":
    image_to_schematic("logo.png", "sky_logo", use_original_size=True)

