"""
SkyLogo Generator - Unified Version
=====================================
1. Takes a logo image (PNG/JPG) from the current folder
2. Scales it so it has the desired number of non-white pixels
3. Fixes gaps (isolated missing pixels) for crying_obsidian and other blocks
4. Generates a .litematic file directly (requires: pip install litemapy Pillow numpy)

Dependencies:
    pip install Pillow litemapy numpy
"""

from PIL import Image, ImageFilter
import math
import os
import sys
import numpy as np

# ──────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────

# Threshold to consider a pixel as "white/background"
WHITE_THRESHOLD = 240

# Color → Minecraft block palette
# Add more colors here if needed
COLOR_TO_BLOCK = {
    "obsidian": [
        (0, 0, 0),          # pure black
        (10, 10, 10),       # near-black
    ],
    "crying_obsidian": [
        (159, 129, 214),    # purple #9F81D6
        (140, 110, 200),    # darker variant
        (170, 145, 220),    # lighter variant
    ],
}

# Color tolerance (max Euclidean distance per R/G/B channel)
COLOR_TOLERANCE = 30

# Search radius for gap filling (in image pixels)
# Increase for larger gaps, decrease to be more conservative
GAP_FILL_RADIUS = 2

# Output directory
OUTPUT_DIR = "output"


# ──────────────────────────────────────────────
# PART 1: IMAGE SCALING
# ──────────────────────────────────────────────

def count_non_white_pixels(img: Image.Image) -> int:
    """Counts non-white (non-background) pixels in the image."""
    img_rgb = img.convert("RGB")
    pixels = img_rgb.load()
    width, height = img_rgb.size
    count = 0
    for y in range(height):
        for x in range(width):
            r, g, b = pixels[x, y]
            if r < WHITE_THRESHOLD or g < WHITE_THRESHOLD or b < WHITE_THRESHOLD:
                count += 1
    return count


def upscale_to_target_nonwhite(img: Image.Image, target_nonwhite: int):
    """
    Resizes the image so it contains approximately `target_nonwhite`
    non-white pixels, preserving aspect ratio.
    """
    width, height = img.size
    total_pixels = width * height
    current_nonwhite = count_non_white_pixels(img)

    if current_nonwhite == 0:
        print("⚠️  WARNING: The image appears to be completely white/empty!")
        return img, 0, 0.0, 1.0, width, height

    density = current_nonwhite / total_pixels
    required_total_pixels = target_nonwhite / density
    scale_factor = math.sqrt(required_total_pixels / total_pixels)

    new_width = int(width * scale_factor)
    new_height = int(height * scale_factor)

    # NEAREST to preserve sharp edges after scaling
    resized = img.resize((new_width, new_height), Image.NEAREST)

    return resized, current_nonwhite, density, scale_factor, new_width, new_height


# ──────────────────────────────────────────────
# PART 2: GAP CORRECTION
# ──────────────────────────────────────────────

def color_matches_block(color: tuple, tolerance: int = COLOR_TOLERANCE) -> str | None:
    """
    Returns the block name if the color matches, otherwise None.
    Uses Euclidean distance for more accurate matching.
    """
    best_block = None
    best_dist = float('inf')

    for block, colors in COLOR_TO_BLOCK.items():
        for c in colors:
            dist = math.sqrt(sum((color[i] - c[i]) ** 2 for i in range(3)))
            if dist < best_dist:
                best_dist = dist
                best_block = block

    # Accept only if distance is within tolerance
    if best_dist <= tolerance * math.sqrt(3):
        return best_block
    return None


def fill_gaps(img: Image.Image, radius: int = GAP_FILL_RADIUS) -> Image.Image:
    """
    Fixes gaps in the image using numpy for efficient processing on large images.
    Isolated white/transparent pixels surrounded by colored pixels are filled
    with the average color of their neighborhood.
    """
    arr = np.array(img.convert("RGBA"), dtype=np.uint8)
    height, width = arr.shape[:2]

    for iteration in range(2):
        rgb = arr[:, :, :3].astype(np.int32)
        alpha = arr[:, :, 3]

        # Mask empty pixels (transparent or white)
        is_empty = (alpha == 0) | (
            (rgb[:, :, 0] >= WHITE_THRESHOLD) &
            (rgb[:, :, 1] >= WHITE_THRESHOLD) &
            (rgb[:, :, 2] >= WHITE_THRESHOLD)
        )

        # Count filled neighbors using array shifts (very fast)
        filled_count = np.zeros((height, width), dtype=np.int32)
        color_sum = np.zeros((height, width, 3), dtype=np.int32)

        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                if dx == 0 and dy == 0:
                    continue
                # Shift array to look at neighbor (dy, dx)
                shifted_empty = np.roll(np.roll(is_empty, dy, axis=0), dx, axis=1)
                shifted_rgb = np.roll(np.roll(rgb, dy, axis=0), dx, axis=1)

                # Edges: zero out pixels coming from wrap-around
                if dy > 0:
                    shifted_empty[:dy, :] = True
                    shifted_rgb[:dy, :] = 0
                elif dy < 0:
                    shifted_empty[dy:, :] = True
                    shifted_rgb[dy:, :] = 0
                if dx > 0:
                    shifted_empty[:, :dx] = True
                    shifted_rgb[:, :dx] = 0
                elif dx < 0:
                    shifted_empty[:, dx:] = True
                    shifted_rgb[:, dx:] = 0

                # Non-empty neighbor contributes to count and color sum
                neighbor_filled = ~shifted_empty
                filled_count += neighbor_filled.astype(np.int32)
                color_sum += shifted_rgb * neighbor_filled[:, :, np.newaxis]

        # Threshold: at least 40% of neighbors must be filled
        total_neighbors = (2 * radius + 1) ** 2 - 1
        threshold = total_neighbors * 0.40
        should_fill = is_empty & (filled_count >= threshold)

        changes = int(np.sum(should_fill))
        if changes == 0:
            break

        # Compute average neighbor color and apply
        safe_count = np.where(filled_count > 0, filled_count, 1)
        avg_color = (color_sum / safe_count[:, :, np.newaxis]).astype(np.uint8)

        # Apply only to pixels that should be filled
        mask = should_fill[:, :, np.newaxis]
        arr[:, :, :3] = np.where(mask, avg_color, arr[:, :, :3])
        arr[:, :, 3] = np.where(should_fill, 255, arr[:, :, 3])

        print(f"  -> Gap fill iteration {iteration+1}: fixed {changes:,} pixels")

    return Image.fromarray(arr, "RGBA")


# ──────────────────────────────────────────────
# PART 3: LITEMATIC GENERATION
# ──────────────────────────────────────────────

def image_to_litematic(image_path: str, output_name: str = "sky_logo"):
    """
    Converts an image to a .litematic file using litemapy.
    Each non-empty pixel becomes a Minecraft block on a horizontal plane (Y=0).
    Uses numpy to efficiently handle very large images.
    """
    try:
        from litemapy import Schematic, Region, BlockState
    except ImportError:
        print("\n ERROR: litemapy is not installed.")
        print("   Install with: pip install litemapy")
        print("   Skipping .litematic generation.\n")
        return False

    img = Image.open(image_path).convert("RGBA")
    width, height = img.size

    print(f"\n Building litematic: {width}x{height} blocks ({width*height:,} total)...")
    print(f"   This may take a few minutes for large images, please wait...")

    # Convert to numpy array for fast processing
    arr = np.array(img)  # shape: (height, width, 4) -> RGBA

    # Precompute block masks using vectorized Euclidean distance
    rgb = arr[:, :, :3].astype(np.int32)
    alpha = arr[:, :, 3]

    # Mask transparent and white (background) pixels
    is_transparent = alpha == 0
    is_white = (rgb[:, :, 0] >= WHITE_THRESHOLD) & \
               (rgb[:, :, 1] >= WHITE_THRESHOLD) & \
               (rgb[:, :, 2] >= WHITE_THRESHOLD)
    is_empty = is_transparent | is_white

    # For each block, compute Euclidean distance from all reference colors
    # and build a block_id map for each pixel
    block_list = list(COLOR_TO_BLOCK.keys())
    best_dist = np.full((height, width), float('inf'))
    best_block_idx = np.full((height, width), -1, dtype=np.int32)

    for block_idx, (block_name, colors) in enumerate(COLOR_TO_BLOCK.items()):
        for ref_color in colors:
            ref = np.array(ref_color, dtype=np.int32)
            diff = rgb - ref  # shape (H, W, 3)
            dist = np.sqrt(np.sum(diff ** 2, axis=2))  # shape (H, W)
            closer = dist < best_dist
            best_dist = np.where(closer, dist, best_dist)
            best_block_idx = np.where(closer, block_idx, best_block_idx)

    # Tolerance: discard pixels too far from any reference color
    max_dist = COLOR_TOLERANCE * math.sqrt(3)
    too_far = best_dist > max_dist

    # Final map: -1 = empty/background/unrecognized, otherwise block index
    final_map = np.where(is_empty | too_far, -1, best_block_idx)

    # Find coordinates of pixels to place
    ys, xs = np.where(final_map >= 0)
    placed = len(xs)

    print(f"   Recognized pixels: {placed:,} -- building region...")

    # Create the litematic region
    region = Region(0, 0, 0, width, 1, height)

    # Pre-instantiate BlockStates (one per block type, not one per pixel)
    block_states = {
        block_name: BlockState(f"minecraft:{block_name}")
        for block_name in block_list
    }

    # Place blocks -- only loop over non-empty pixels
    chunk_size = 100_000
    total = len(xs)
    for i in range(0, total, chunk_size):
        batch_xs = xs[i:i+chunk_size]
        batch_ys = ys[i:i+chunk_size]
        for x, y in zip(batch_xs, batch_ys):
            block_idx = final_map[y, x]
            block_name = block_list[block_idx]
            region.setblock(int(x), 0, int(y), block_states[block_name])
        print(f"   Progress: {min(i+chunk_size, total):,} / {total:,} blocks ({100*min(i+chunk_size,total)//total}%)")

    schem = Schematic(
        name=output_name,
        author="SkyLogoGenerator",
        description="Auto-generated by skylogo_generator.py",
        regions={output_name: region},
    )

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, f"{output_name}.litematic")
    print(f"   Saving .litematic file (may take a moment)...")
    schem.save(output_path)

    print(f"Litematic saved: {output_path}")
    print(f"   Blocks placed: {placed:,}")
    print(f"   Dimensions: {width} x {height} blocks (X x Z)")
    return True


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────

def main():
    print("=" * 50)
    print("  SkyLogo Generator - Unified Version")
    print("=" * 50)

    # -- Step 1: ask for target --
    try:
        target_nonwhite = int(input("\nEnter the desired number of non-white pixels (e.g. 3500000): "))
    except ValueError:
        print("Invalid value. Please enter an integer.")
        sys.exit(1)

    # -- Step 2: find images in current folder --
    files = [f for f in os.listdir() if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    if not files:
        print("No images found in the current folder.")
        sys.exit(1)

    if len(files) > 1:
        print(f"\nFound {len(files)} images:")
        for i, f in enumerate(files):
            print(f"  [{i}] {f}")
        try:
            choice = int(input("Which one do you want to use? (number): "))
            files = [files[choice]]
        except (ValueError, IndexError):
            print("Invalid choice, using the first image.")
            files = [files[0]]

    source_file = files[0]
    print(f"\nSource image: {source_file}")

    # -- Step 3: scale the image --
    print("\nScaling image...")
    img = Image.open(source_file)
    upscaled, current_nw, density, scale, new_w, new_h = upscale_to_target_nonwhite(img, target_nonwhite)

    print(f"  Original non-white pixels: {current_nw:,}")
    print(f"  Density: {density:.4f}")
    print(f"  Scale factor: {scale:.4f}")
    print(f"  New dimensions: {new_w} x {new_h}")

    # -- Step 4: fix gaps --
    print("\nFixing gaps...")
    filled_img = fill_gaps(upscaled, radius=GAP_FILL_RADIUS)

    logo_path = "logo.png"
    filled_img.save(logo_path)
    print(f"  Processed image saved as: {logo_path}")

    # -- Step 5: generate litematic --
    output_name = os.path.splitext(source_file)[0] + "_sky"
    success = image_to_litematic(logo_path, output_name)

    if not success:
        # Fallback: try .schem if litemapy is unavailable
        print("\nAttempting fallback with mcschematic (.schem)...")
        try:
            import mcschematic
            img_final = Image.open(logo_path).convert("RGBA")
            width, height = img_final.size
            schem = mcschematic.MCSchematic()

            for y in range(height):
                for x in range(width):
                    r, g, b, a = img_final.getpixel((x, y))
                    if a == 0:
                        continue
                    is_bg = r >= WHITE_THRESHOLD and g >= WHITE_THRESHOLD and b >= WHITE_THRESHOLD
                    if is_bg:
                        continue
                    block_name = color_matches_block((r, g, b))
                    if block_name:
                        schem.setBlock((x, 0, y), f"minecraft:{block_name}")

            os.makedirs(OUTPUT_DIR, exist_ok=True)
            schem.save(OUTPUT_DIR, output_name, mcschematic.Version.JE_1_20)
            print(f"Schematic saved: {OUTPUT_DIR}/{output_name}.schem")
        except ImportError:
            print("mcschematic is also unavailable.")
            print("   Install with: pip install litemapy  (recommended)")
            print("              or: pip install mcschematic")

    print("\nDone!")


if __name__ == "__main__":
    main()