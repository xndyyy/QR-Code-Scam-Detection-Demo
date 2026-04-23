from PIL import Image
from pyzbar.pyzbar import decode
import numpy as np

def array_to_qr_image(arr, scale=12, border_modules=4, invert=False):
    """
    Convert a (H,W) QR matrix into a decodable PIL image.
    - scale: upscale factor
    - border_modules: white border thickness in QR modules
    """
    arr = np.array(arr)

    # Normalize -> uint8
    if arr.max() <= 1:
        arr = (arr * 255).astype(np.uint8)
    else:
        arr = arr.astype(np.uint8)

    if invert:
        arr = 255 - arr

    h, w = arr.shape

    # Upscale with NEAREST to preserve edges
    img = Image.fromarray(arr)
    img = img.resize((w * scale, h * scale), resample=Image.NEAREST)

    # Add quiet zone (white border)
    border = border_modules * scale
    new_w = img.width + 2 * border
    new_h = img.height + 2 * border
    bg = Image.new("L", (new_w, new_h), 255)
    bg.paste(img, (border, border))

    return bg

def decode_qr_from_array(arr):
    """
    Try decoding a QR code from a (H, W) array.
    Tries normal and inverted versions.
    Returns (decoded_text, pil_image_used) or (None, last_image_attempt)
    """
    for invert in (False, True):
        img = array_to_qr_image(arr, scale=10, invert=invert, border_modules=4)
        result = decode(img)
        if result:
            try:
                text = result[0].data.decode("utf-8")
            except Exception:
                text = result[0].data
            return text

    return None