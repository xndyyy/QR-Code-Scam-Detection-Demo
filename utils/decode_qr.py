from PIL import Image
import numpy as np
import cv2


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


def decode_qr_from_array(image: Image.Image) -> str | None:
    """
    Decode a QR code from a PIL image using OpenCV.
    Returns the decoded string, or None if decoding fails.
    """
    img_np = np.array(image.convert("RGB"))
    detector = cv2.QRCodeDetector()
    data, _, _ = detector.detectAndDecode(img_np)

    if data:
        return data
    return None