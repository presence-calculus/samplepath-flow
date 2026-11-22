const palette = [
  "#ffffff", "#14243b", "#52677e", "#e4eaf2", "#9eafc3", "#324b67", "#098478", "#264e80",
  "#185dcb", "#53647f", "#e9f1ff", "#184d9b", "#f5f8fc", "#eff3f8", "#d9e3ef", "#c8d7e9",
  "#f8ebe5", "#a0b8d6", "#899eb7", "#a0442c", "#617891", "#3e5370", "#243751", "#0f1c2e",
  "#e9f6f4", "#cae8e4", "#a0d7cf", "#70bfb5", "#3ca295", "#08695f", "#b7d3f6", "#5187cf",
].map(hex => Number.parseInt(hex.slice(1), 16))
  .map(color => [color >> 16, (color >> 8) & 255, color & 255]);

export class GifEncoder {
  private readonly bytes: number[] = [];
  private readonly cache = new Map<number, number>();
  private readonly width: number;
  private readonly height: number;

  constructor(width: number, height: number, loop = true) {
    this.width = width;
    this.height = height;
    this.bytes.push(...Array.from("GIF89a", character => character.charCodeAt(0)));
    this.word(width);
    this.word(height);
    this.bytes.push(0xf4, 0, 0); // Global palette with 32 colors and no pixel aspect ratio.
    for (const color of palette) this.bytes.push(...color);
    if (loop) {
      this.bytes.push(0x21, 0xff, 11, ...Array.from("NETSCAPE2.0", c => c.charCodeAt(0)), 3, 1, 0, 0, 0);
    }
  }

  private word(value: number) {
    this.bytes.push(value & 255, (value >> 8) & 255);
  }

  private paletteIndex(red: number, green: number, blue: number) {
    const color = (red << 16) | (green << 8) | blue;
    const cached = this.cache.get(color);
    if (cached !== undefined) return cached;

    let best = 0;
    let closest = Infinity;
    for (let index = 0; index < palette.length; index++) {
      const [r, g, b] = palette[index];
      const distance = (red - r) ** 2 + (green - g) ** 2 + (blue - b) ** 2;
      if (distance < closest) {
        best = index;
        closest = distance;
      }
    }
    this.cache.set(color, best);
    return best;
  }

  addFrame(pixels: Uint8ClampedArray, delay: number) {
    this.bytes.push(0x21, 0xf9, 4, 4);
    this.word(delay);
    this.bytes.push(0, 0, 0x2c);
    this.word(0);
    this.word(0);
    this.word(this.width);
    this.word(this.height);
    this.bytes.push(0, 5);

    const compressed = encodePixels(pixels, (r, g, b) => this.paletteIndex(r, g, b));
    for (let offset = 0; offset < compressed.length; offset += 255) {
      const block = compressed.slice(offset, offset + 255);
      this.bytes.push(block.length, ...block);
    }
    this.bytes.push(0);
  }

  finish() {
    return new Uint8Array([...this.bytes, 0x3b]);
  }
}

function encodePixels(pixels: Uint8ClampedArray, toIndex: (r: number, g: number, b: number) => number) {
  const clear = 32;
  const end = 33;
  const dictionary = new Map<number, number>();
  const result: number[] = [];
  let size = 6;
  let nextCode = 34;
  let buffer = 0;
  let used = 0;

  function write(code: number) {
    buffer |= code << used;
    used += size;
    while (used >= 8) {
      result.push(buffer & 255);
      buffer >>>= 8;
      used -= 8;
    }
    // The decoder adds a dictionary entry after reading this code.
    if (nextCode === (1 << size) && size < 12) size++;
  }

  write(clear);
  let prefix = toIndex(pixels[0], pixels[1], pixels[2]);
  for (let offset = 4; offset < pixels.length; offset += 4) {
    const value = toIndex(pixels[offset], pixels[offset + 1], pixels[offset + 2]);
    const key = prefix * 32 + value;
    const existing = dictionary.get(key);
    if (existing !== undefined) {
      prefix = existing;
      continue;
    }
    write(prefix);
    if (nextCode < 4096) {
      dictionary.set(key, nextCode++);
    } else {
      write(clear);
      dictionary.clear();
      size = 6;
      nextCode = 34;
    }
    prefix = value;
  }
  write(prefix);
  write(end);
  if (used) result.push(buffer & 255);
  return result;
}
