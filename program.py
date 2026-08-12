"""
SkyLogo Generator - Unified Version
=====================================
1. Takes a logo image (PNG/JPG) from the current folder
2. Scales it so it has the desired number of non-white pixels
3. Fixes gaps (isolated missing pixels) for crying_obsidian and other blocks
4. Generates a .litematic file directly (requires: pip install litemapy Pillow numpy)

The logo can be generated in two orientations:

    horizontal  the logo lies flat on the X/Z plane (image X -> world X,
                image row -> world Z), a single block layer at Y = 0.
    vertical    the logo stands up like a wall (image X -> world X,
                image row -> world Y, flipped so the top of the picture is up),
                a single block layer at Z = 0.

Dependencies:
    pip install Pillow litemapy numpy
"""

from PIL import Image
import math
import os
import sys
import numpy as np

# Pillow refuses images above ~179 megapixels by default, as a guard against
# maliciously crafted files that try to exhaust memory. This generator is meant
# to be pointed at very large local images the user owns, and an upscaled logo
# easily goes past that limit, so the guard is lifted here.
# Only run this program on images you trust.
Image.MAX_IMAGE_PIXELS = None

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
# ORIENTATION AND MINECRAFT WORLD LIMITS
# ──────────────────────────────────────────────

ORIENTATION_HORIZONTAL = "horizontal"
ORIENTATION_VERTICAL = "vertical"

# Overworld build range: the lowest placeable block is at Y = -64 and the
# highest one at Y = 319 (Y = 320 is the exclusive upper bound), which leaves
# 384 blocks of vertical space for the build.
MC_WORLD_MIN_Y = -64
MC_WORLD_MAX_Y = 320
MC_MAX_BUILD_HEIGHT = MC_WORLD_MAX_Y - MC_WORLD_MIN_Y  # 384


class LogoTooTallError(ValueError):
    """Raised when a vertical logo would not fit in the Minecraft build height."""


class NoRecognizedPixelsError(ValueError):
    """Raised when no pixel of the image matches the block palette."""


class CoordinateOutOfBoundsError(RuntimeError):
    """Raised when computed block coordinates do not fit the region dimensions.

    This should never happen; it exists so that a mapping mistake is reported
    with useful context instead of surfacing as an opaque numpy IndexError from
    deep inside litemapy.
    """


# ──────────────────────────────────────────────
# PART 1: IMAGE SCALING
# ──────────────────────────────────────────────

def background_mask(arr: np.ndarray) -> np.ndarray:
    """
    Returns a boolean mask that is True for pixels considered background.

    A pixel is background when it is fully transparent, or when all of its
    RGB channels are at or above WHITE_THRESHOLD. This is the single definition
    of "empty" used by every stage of the pipeline (counting, gap filling and
    block placement), so the three stages can never disagree.

    :param arr: an (H, W, 4) uint8 RGBA array
    """
    rgb = arr[:, :, :3]
    alpha = arr[:, :, 3]
    is_white = (
        (rgb[:, :, 0] >= WHITE_THRESHOLD) &
        (rgb[:, :, 1] >= WHITE_THRESHOLD) &
        (rgb[:, :, 2] >= WHITE_THRESHOLD)
    )
    return (alpha == 0) | is_white


def count_non_white_pixels(img: Image.Image) -> int:
    """Counts non-white (non-background) pixels in the image."""
    arr = np.array(img.convert("RGBA"), dtype=np.uint8)
    return int(np.count_nonzero(~background_mask(arr)))


def compute_scaled_size(width: int, height: int, current_nonwhite: int,
                        target_nonwhite: int) -> tuple[int, int, float]:
    """
    Computes the image size that contains approximately `target_nonwhite`
    non-white pixels, preserving the aspect ratio.

    :returns: (new_width, new_height, scale_factor)
    """
    scale_factor = math.sqrt(target_nonwhite / current_nonwhite)
    # At least one pixel per axis: a zero-sized image cannot be resized and
    # would produce an invalid (zero-length) litematic region.
    new_width = max(1, int(width * scale_factor))
    new_height = max(1, int(height * scale_factor))
    return new_width, new_height, scale_factor


def upscale_to_target_nonwhite(img: Image.Image, target_nonwhite: int,
                               current_nonwhite: int | None = None):
    """
    Resizes the image so it contains approximately `target_nonwhite`
    non-white pixels, preserving aspect ratio.

    :param current_nonwhite: an already computed non-white pixel count, to avoid
                             scanning the image twice
    """
    width, height = img.size
    total_pixels = width * height
    if current_nonwhite is None:
        current_nonwhite = count_non_white_pixels(img)

    if current_nonwhite == 0:
        print("⚠️  WARNING: The image appears to be completely white/empty!")
        return img, 0, 0.0, 1.0, width, height

    density = current_nonwhite / total_pixels
    new_width, new_height, scale_factor = compute_scaled_size(
        width, height, current_nonwhite, target_nonwhite
    )

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

        # Mask empty pixels (transparent or white)
        is_empty = background_mask(arr)

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
# PART 3: COORDINATE MAPPING
# ──────────────────────────────────────────────

def compute_region_dimensions(image_width: int, image_height: int,
                              orientation: str) -> tuple[int, int, int]:
    """
    Returns the litematic region size as (size_x, size_y, size_z) for the given
    image size and orientation. The region is exactly one block thick on the
    axis the logo does not use.
    """
    if orientation == ORIENTATION_HORIZONTAL:
        return image_width, 1, image_height
    if orientation == ORIENTATION_VERTICAL:
        return image_width, image_height, 1
    raise ValueError(f"Unknown orientation: {orientation!r}")


def map_image_pixels_to_region(xs_img: np.ndarray, ys_img: np.ndarray,
                               image_height: int, orientation: str):
    """
    Maps image pixel coordinates to region-local block coordinates.

    Region-local coordinates always start at 0 and run up to size-1 on each
    axis; the position of the logo in the world is decided by the region origin,
    never by shifting these coordinates. Offsetting the coordinates themselves
    without growing the region is what pushes indices past the end of the
    region's block array.

    :param xs_img: image column indices (0 .. image_width-1)
    :param ys_img: image row indices (0 .. image_height-1)
    """
    if orientation == ORIENTATION_HORIZONTAL:
        # Image column -> X, image row -> Z, single layer at Y = 0.
        return xs_img, np.zeros_like(xs_img), ys_img
    if orientation == ORIENTATION_VERTICAL:
        # Image column -> X, image row -> Y, single layer at Z = 0.
        # Rows are flipped because row 0 is the top of the picture while
        # Minecraft's Y axis grows upwards.
        return xs_img, (image_height - 1) - ys_img, np.zeros_like(xs_img)
    raise ValueError(f"Unknown orientation: {orientation!r}")


def check_vertical_height(image_height: int, orientation: str,
                          current_nonwhite: int | None = None,
                          original_height: int | None = None) -> None:
    """
    Refuses a vertical logo that does not fit in the Minecraft build height.

    :raises LogoTooTallError: if the logo is taller than MC_MAX_BUILD_HEIGHT
    """
    if orientation != ORIENTATION_VERTICAL:
        return
    if image_height <= MC_MAX_BUILD_HEIGHT:
        return

    message = (
        f"The vertical logo would be {image_height:,} blocks tall, but Minecraft "
        f"only offers {MC_MAX_BUILD_HEIGHT} blocks of vertical space "
        f"(Y from {MC_WORLD_MIN_Y} to {MC_WORLD_MAX_Y - 1}).\n"
        f"The logo is not resized or cropped automatically."
    )
    if current_nonwhite and original_height:
        max_target = max_target_for_vertical(current_nonwhite, original_height)
        message += (
            f"\nOptions:\n"
            f"  - lower the target to at most {max_target:,} non-white pixels, or\n"
            f"  - choose the horizontal orientation, which has no height limit."
        )
    else:
        message += (
            f"\nOptions: lower the target number of non-white pixels, or choose "
            f"the horizontal orientation, which has no height limit."
        )
    raise LogoTooTallError(message)


def max_target_for_vertical(current_nonwhite: int, original_height: int) -> int:
    """
    Largest target number of non-white pixels whose vertical logo still fits in
    the Minecraft build height.

    The scaling keeps the aspect ratio, so the number of non-white pixels grows
    with the square of the scale factor while the height grows linearly.
    """
    max_scale = MC_MAX_BUILD_HEIGHT / original_height
    return max(1, int(current_nonwhite * max_scale ** 2))


def validate_region_coordinates(xs: np.ndarray, ys: np.ndarray, zs: np.ndarray,
                                dimensions: tuple[int, int, int],
                                orientation: str) -> None:
    """
    Checks every computed coordinate against the region dimensions before any
    block is written.

    :raises CoordinateOutOfBoundsError: if any coordinate falls outside
    """
    if len(xs) == 0:
        return

    for axis_name, coords, size in (("X", xs, dimensions[0]),
                                    ("Y", ys, dimensions[1]),
                                    ("Z", zs, dimensions[2])):
        lowest = int(coords.min())
        highest = int(coords.max())
        if lowest < 0 or highest >= size:
            raise CoordinateOutOfBoundsError(
                f"Block coordinates do not fit the litematic region.\n"
                f"  orientation: {orientation}\n"
                f"  region size: {dimensions[0]} x {dimensions[1]} x {dimensions[2]} (X x Y x Z)\n"
                f"  axis {axis_name}: computed coordinates span [{lowest}, {highest}], "
                f"but valid values are 0 to {size - 1}.\n"
                f"This is a bug in the coordinate mapping, not something you did wrong."
            )


# ──────────────────────────────────────────────
# PART 4: LITEMATIC GENERATION
# ──────────────────────────────────────────────

def place_blocks(region, dimensions: tuple[int, int, int], coords_by_block: dict) -> int:
    """
    Writes all blocks into the region and returns how many were placed.

    Every block state is first registered in the region palette through the
    public API. When the region exposes its block array in the expected shape,
    the remaining positions are filled with a vectorized numpy assignment, which
    is orders of magnitude faster than a per-pixel Python loop on multi-million
    block logos. Otherwise the public per-block API is used instead.
    """
    from litemapy import BlockState

    if any(size <= 0 for size in dimensions):
        raise ValueError(f"Region dimensions must all be positive, got {dimensions}")

    palette_index = {}
    for block_name, (bxs, bys, bzs) in coords_by_block.items():
        state = BlockState(f"minecraft:{block_name}")
        # Registers the state in the palette and places the first block.
        region[int(bxs[0]), int(bys[0]), int(bzs[0])] = state
        palette_index[block_name] = region.palette.index(state)

    block_store = getattr(region, "_Region__blocks", None)
    can_bulk_write = (
        isinstance(block_store, np.ndarray)
        and block_store.shape == tuple(dimensions)
        and block_store.dtype.kind == "u"
    )

    placed = 0
    if can_bulk_write:
        # Region dimensions are positive, so region coordinates and storage
        # indices are identical and the array can be indexed directly.
        for block_name, (bxs, bys, bzs) in coords_by_block.items():
            block_store[bxs, bys, bzs] = palette_index[block_name]
            placed += len(bxs)
            print(f"   Placed {len(bxs):,} {block_name} blocks")
    else:
        print("   (using the slower per-block API)")
        for block_name, (bxs, bys, bzs) in coords_by_block.items():
            state = BlockState(f"minecraft:{block_name}")
            total = len(bxs)
            for index in range(total):
                region[int(bxs[index]), int(bys[index]), int(bzs[index])] = state
                placed += 1
                if placed % 250_000 == 0:
                    print(f"   Progress: {placed:,} blocks")
            print(f"   Placed {total:,} {block_name} blocks")

    written = region.count_blocks()
    if written != placed:
        raise RuntimeError(
            f"Block count mismatch: {placed:,} blocks were requested but the region "
            f"contains {written:,}. The generated litematic would be wrong."
        )
    return placed


def image_to_litematic(image_path: str, output_name: str = "sky_logo",
                       orientation: str = ORIENTATION_HORIZONTAL) -> bool:
    """
    Converts an image to a .litematic file using litemapy.

    Each recognized pixel becomes a Minecraft block on a single-block-thick
    plane: horizontal (X/Z, at Y=0) or vertical (X/Y, at Z=0).
    Uses numpy to efficiently handle very large images.
    """
    try:
        from litemapy import Schematic, Region
    except ImportError:
        print("\n ERROR: litemapy is not installed.")
        print("   Install with: pip install litemapy")
        print("   Skipping .litematic generation.\n")
        return False

    img = Image.open(image_path).convert("RGBA")
    width, height = img.size

    # Authoritative height check: the image on disk is what gets built.
    check_vertical_height(height, orientation)

    dimensions = compute_region_dimensions(width, height, orientation)
    print(f"\n Building litematic ({orientation}): "
          f"{dimensions[0]} x {dimensions[1]} x {dimensions[2]} blocks (X x Y x Z)...")
    print(f"   This may take a few minutes for large images, please wait...")

    # Convert to numpy array for fast processing
    arr = np.array(img)  # shape: (height, width, 4) -> RGBA
    rgb = arr[:, :, :3].astype(np.int32)

    # Mask transparent and white (background) pixels
    is_empty = background_mask(arr)

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
    ys_img, xs_img = np.where(final_map >= 0)
    placed = len(xs_img)

    print(f"   Recognized pixels: {placed:,}")

    if placed == 0:
        raise NoRecognizedPixelsError(
            f"No pixel of '{image_path}' matched the block palette.\n"
            f"  Every pixel was either background (fully transparent, or all RGB "
            f"channels >= {WHITE_THRESHOLD})\n"
            f"  or further than {max_dist:.1f} in color distance from every palette color.\n"
            f"  Check COLOR_TO_BLOCK / COLOR_TOLERANCE, or the colors of the source image."
        )

    # Map image pixels to region coordinates and check them against the region
    # size before touching litemapy.
    xs, ys, zs = map_image_pixels_to_region(xs_img, ys_img, height, orientation)
    validate_region_coordinates(xs, ys, zs, dimensions, orientation)

    region = Region(0, 0, 0, dimensions[0], dimensions[1], dimensions[2])

    # Group the positions per block type so they can be written in bulk.
    block_ids = final_map[ys_img, xs_img]
    coords_by_block = {}
    for block_idx, block_name in enumerate(block_list):
        selection = block_ids == block_idx
        if not np.any(selection):
            continue
        coords_by_block[block_name] = (xs[selection], ys[selection], zs[selection])

    print(f"   Writing blocks...")
    placed = place_blocks(region, dimensions, coords_by_block)

    schem = Schematic(
        name=output_name,
        author="SkyLogoGenerator",
        description=f"Auto-generated by skylogo_generator.py ({orientation})",
        regions={output_name: region},
    )

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, f"{output_name}.litematic")
    print(f"   Saving .litematic file (may take a moment)...")
    schem.save(output_path)

    print(f"Litematic saved: {output_path}")
    print(f"   Blocks placed: {placed:,}")
    print(f"   Orientation: {orientation}")
    print(f"   Dimensions: {dimensions[0]} x {dimensions[1]} x {dimensions[2]} blocks (X x Y x Z)")
    if orientation == ORIENTATION_VERTICAL:
        print(f"   Needs {dimensions[1]:,} blocks of vertical space "
              f"(limit: {MC_MAX_BUILD_HEIGHT})")
    return True


# ──────────────────────────────────────────────
# FALLBACK: .schem OUTPUT
# ──────────────────────────────────────────────

def image_to_schem(image_path: str, output_name: str, orientation: str) -> bool:
    """Fallback .schem output through mcschematic, same orientation rules."""
    try:
        import mcschematic
    except ImportError:
        print("mcschematic is also unavailable.")
        print("   Install with: pip install litemapy  (recommended)")
        print("              or: pip install mcschematic")
        return False

    img_final = Image.open(image_path).convert("RGBA")
    width, height = img_final.size
    check_vertical_height(height, orientation)

    schem = mcschematic.MCSchematic()
    placed = 0

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
                if orientation == ORIENTATION_HORIZONTAL:
                    position = (x, 0, y)
                else:
                    position = (x, (height - 1) - y, 0)
                schem.setBlock(position, f"minecraft:{block_name}")
                placed += 1

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    schem.save(OUTPUT_DIR, output_name, mcschematic.Version.JE_1_20)
    print(f"Schematic saved: {OUTPUT_DIR}/{output_name}.schem")
    print(f"   Blocks placed: {placed:,}")
    return True


# ──────────────────────────────────────────────
# INTERACTIVE HELPERS
# ──────────────────────────────────────────────

def ask_image(files: list[str]) -> str:
    """Asks which image to use when several are available."""
    if len(files) == 1:
        return files[0]

    print(f"\nFound {len(files)} images:")
    for i, name in enumerate(files):
        print(f"  [{i}] {name}")
    answer = input("Which one do you want to use? (number): ").strip()
    if answer.isdigit() and int(answer) < len(files):
        return files[int(answer)]
    print("Invalid choice, using the first image.")
    return files[0]


def ask_orientation() -> str:
    """Asks for the logo orientation."""
    print("\nLogo orientation:")
    print("  [1] Horizontal - lies flat on the ground (X/Z plane)")
    print(f"  [2] Vertical   - stands up like a wall (X/Y plane), "
          f"max {MC_MAX_BUILD_HEIGHT} blocks tall")
    while True:
        answer = input("Choose the orientation (1 or 2): ").strip()
        if answer == "1":
            return ORIENTATION_HORIZONTAL
        if answer == "2":
            return ORIENTATION_VERTICAL
        print("Please type 1 or 2.")


def ask_target(current_nonwhite: int, original_height: int, orientation: str) -> int:
    """Asks for the desired number of non-white pixels."""
    if orientation == ORIENTATION_VERTICAL:
        max_target = max_target_for_vertical(current_nonwhite, original_height)
        print(f"\nVertical logos may not exceed {MC_MAX_BUILD_HEIGHT} blocks in height,")
        print(f"which for this image means at most ~{max_target:,} non-white pixels.")

    answer = input("\nEnter the desired number of non-white pixels (e.g. 3500000): ").strip()
    try:
        target = int(answer)
    except ValueError:
        print("Invalid value. Please enter an integer.")
        sys.exit(1)
    if target <= 0:
        print("Invalid value. The number of non-white pixels must be positive.")
        sys.exit(1)
    return target


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────

def main():
    print("=" * 50)
    print("  SkyLogo Generator - Unified Version")
    print("=" * 50)

    # -- Step 1: find images in current folder --
    files = sorted(f for f in os.listdir()
                   if f.lower().endswith(('.png', '.jpg', '.jpeg')))

    if not files:
        print("No images found in the current folder.")
        sys.exit(1)

    source_file = ask_image(files)
    print(f"\nSource image: {source_file}")

    img = Image.open(source_file)
    original_width, original_height = img.size
    current_nonwhite = count_non_white_pixels(img)
    print(f"  Size: {original_width} x {original_height}")
    print(f"  Non-white pixels: {current_nonwhite:,}")

    if current_nonwhite == 0:
        print("\nERROR: the image is completely white/transparent, "
              "there is nothing to build.")
        sys.exit(1)

    # -- Step 2: orientation and target --
    orientation = ask_orientation()
    target_nonwhite = ask_target(current_nonwhite, original_height, orientation)

    # -- Step 3: scale the image --
    print("\nScaling image...")
    new_w, new_h, scale = compute_scaled_size(
        original_width, original_height, current_nonwhite, target_nonwhite
    )

    # Refuse a logo that is too tall before spending minutes on processing.
    try:
        check_vertical_height(new_h, orientation, current_nonwhite, original_height)
    except LogoTooTallError as error:
        print(f"\nERROR: {error}")
        sys.exit(2)

    upscaled, current_nonwhite, density, scale, new_w, new_h = upscale_to_target_nonwhite(
        img, target_nonwhite, current_nonwhite
    )

    print(f"  Original non-white pixels: {current_nonwhite:,}")
    print(f"  Density: {density:.4f}")
    print(f"  Scale factor: {scale:.4f}")
    print(f"  New dimensions: {new_w} x {new_h}")

    # -- Step 4: fix gaps --
    print("\nFixing gaps...")
    filled_img = fill_gaps(upscaled, radius=GAP_FILL_RADIUS)

    # Saved inside the output folder so a source image in the current folder
    # (logo.png, for instance) is never overwritten by the processed version.
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    base_name = os.path.splitext(source_file)[0]
    logo_path = os.path.join(OUTPUT_DIR, f"{base_name}_processed.png")
    filled_img.save(logo_path)
    print(f"  Processed image saved as: {logo_path}")

    # -- Step 5: generate litematic --
    output_name = f"{base_name}_sky_{orientation}"
    try:
        success = image_to_litematic(logo_path, output_name, orientation)
    except LogoTooTallError as error:
        print(f"\nERROR: {error}")
        sys.exit(2)
    except NoRecognizedPixelsError as error:
        print(f"\nERROR: {error}")
        sys.exit(3)
    except CoordinateOutOfBoundsError as error:
        print(f"\nERROR: {error}")
        sys.exit(4)

    if not success:
        # Fallback: try .schem if litemapy is unavailable
        print("\nAttempting fallback with mcschematic (.schem)...")
        image_to_schem(logo_path, output_name, orientation)

    print("\nDone!")


if __name__ == "__main__":
    main()
