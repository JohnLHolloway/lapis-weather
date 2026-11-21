"""Blur a 1-bit frame into the grey impression a human gets at arm's length."""
import struct, sys, zlib
import numpy as np

def read_png(path):
    raw = open(path, "rb").read()
    pos, idat, width, height = 8, b"", None, None
    while pos < len(raw):
        length = struct.unpack(">I", raw[pos:pos+4])[0]
        tag = raw[pos+4:pos+8]
        body = raw[pos+8:pos+8+length]
        if tag == b"IHDR":
            width, height = struct.unpack(">II", body[:8])
        elif tag == b"IDAT":
            idat += body
        pos += 12 + length
    data = zlib.decompress(idat)
    arr = np.frombuffer(data, dtype=np.uint8).reshape(height, width + 1)[:, 1:]
    return arr.astype(np.float32)

def box_blur(a, k):
    out = a.copy()
    for axis in (0, 1):
        pad = [(0, 0), (0, 0)]
        pad[axis] = (k // 2, k - 1 - k // 2)
        p = np.pad(out, pad, mode="edge")
        c = np.cumsum(p, axis=axis)
        sl_hi = [slice(None), slice(None)]
        sl_lo = [slice(None), slice(None)]
        sl_hi[axis] = slice(k, None)
        sl_lo[axis] = slice(0, -k)
        out = (np.take(c, range(k, c.shape[axis]), axis=axis) - np.take(c, range(0, c.shape[axis]-k), axis=axis)) / k
    return out

def write_png(path, grey):
    h, w = grey.shape
    raw = b"".join(b"\x00" + grey[row].astype(np.uint8).tobytes() for row in range(h))
    def chunk(tag, body):
        return struct.pack(">I", len(body)) + tag + body + struct.pack(">I", zlib.crc32(tag+body) & 0xFFFFFFFF)
    png = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 0, 0, 0, 0)) + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b"")
    open(path, "wb").write(png)

src, dst, scale = sys.argv[1], sys.argv[2], float(sys.argv[3]) if len(sys.argv) > 3 else 2.0
img = read_png(src)
# downsample to the device's pixels, then blur to a perceived grey
h, w = img.shape
h, w = h // int(round(scale)), w // int(round(scale))
img = img[:h*int(round(scale)), :w*int(round(scale))].reshape(h, int(round(scale)), w, int(round(scale))).mean(axis=(1, 3))
blurred = box_blur(img, 5)
# upscale for viewing
blurred = np.repeat(np.repeat(blurred, 2, axis=0), 2, axis=1)
write_png(dst, np.clip(blurred, 0, 255))
print("wrote", dst)
