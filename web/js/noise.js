// The same world noise as terrain_gan.py. It has to match exactly, or the page would
// produce different terrain from the trained model. 64-bit values are (hi, lo) pairs.

let rh = 0, rl = 0;

function mul(ah, al, bh, bl) {
  const a0 = al & 0xffff, a1 = al >>> 16, b0 = bl & 0xffff, b1 = bl >>> 16;
  const p00 = a0 * b0, p01 = a0 * b1, p10 = a1 * b0, p11 = a1 * b1;
  const mid = Math.floor(p00 / 65536) + (p01 & 0xffff) + (p10 & 0xffff);
  const high = p11 + Math.floor(p01 / 65536) + Math.floor(p10 / 65536) + Math.floor(mid / 65536);
  rl = Math.imul(al, bl);
  rh = (high + Math.imul(ah, bl) + Math.imul(al, bh)) | 0;
}

// splitmix64 finaliser
function mix(h, l) {
  let xl = l ^ ((l >>> 30) | (h << 2)), xh = h ^ (h >>> 30);
  mul(xh, xl, 0xbf58476d | 0, 0x1ce4e5b9);
  h = rh; l = rl;
  xl = l ^ ((l >>> 27) | (h << 5)); xh = h ^ (h >>> 27);
  mul(xh, xl, 0x94d049bb | 0, 0x133111eb);
  h = rh; l = rl;
  rl = l ^ ((l >>> 31) | (h << 1));
  rh = h ^ (h >>> 31);
}

const TWO32 = 4294967296, TWO64 = 18446744073709551616;
const MASK = (1n << 64n) - 1n;

function toDouble(h, l) {
  return (h >>> 0) * TWO32 + (l >>> 0);
}

// Two's complement of a signed integer, as astype(uint64) gives.
function split(v) {
  const hi = Math.floor(v / TWO32);
  return [hi | 0, (v - hi * TWO32) | 0];
}

// Standard normal values of shape (channels, ny, nx), flattened, at the lattice points
// x0 .. x0 + nx - 1 and y0 .. y0 + ny - 1.
export function worldRandn(seed, tag, channels, x0, y0, nx, ny, out = new Float32Array(channels * ny * nx), offset = 0) {
  const key = (BigInt(seed) * 0x9e3779b97f4a7c15n + BigInt(tag) * 0xd1342543de82ef95n) & MASK;
  const keyH = Number(key >> 32n) | 0, keyL = Number(key & 0xffffffffn) | 0;
  const xParts = new Int32Array(2 * nx);
  for (let i = 0; i < nx; i++) {
    const [h, l] = split(x0 + i);
    mul(h, l, 0x9fb21c65 | 0, 0x1e98df25);
    xParts[2 * i] = rh;
    xParts[2 * i + 1] = rl;
  }
  let k = offset;
  for (let c = 0; c < channels; c++) {
    mul(0, c, 0xa24baed4 | 0, 0x963ee407 | 0);
    mix(keyH ^ rh, keyL ^ rl);
    const chH = rh, chL = rl;
    for (let y = 0; y < ny; y++) {
      const [yh0, yl0] = split(y0 + y);
      mul(yh0, yl0, 0xc13fa9a9 | 0, 0x02a6328f);
      const yh = rh, yl = rl;
      for (let x = 0; x < nx; x++) {
        mix(chH ^ xParts[2 * x], chL ^ xParts[2 * x + 1]);
        mix(rh ^ yh, rl ^ yl);
        const sh = rh, sl = rl;
        const l1 = (sl + 1) | 0;
        mix(l1 === 0 ? (sh + 1) | 0 : sh, l1);
        const u1 = (toDouble(rh, rl) + 1) / TWO64;
        const l2 = (sl + 2) | 0;
        mix((l2 >>> 0) < (sl >>> 0) ? (sh + 1) | 0 : sh, l2);
        const u2 = toDouble(rh, rl) / TWO64;
        out[k++] = Math.sqrt(-2 * Math.log(Math.min(u1, 1))) * Math.cos(2 * Math.PI * u2);
      }
    }
  }
  return out;
}

const f = Math.fround;

// Per-cell detail vectors plus a smooth large-scale field; shape (fine + coarse, ny, nx).
export function worldLatents(seed, ix0, iy0, nx, ny, fineCh, coarseCh, cells) {
  const out = new Float32Array((fineCh + coarseCh) * ny * nx);
  worldRandn(seed, 1, fineCh, ix0, iy0, nx, ny, out, 0);

  const cx0 = Math.floor(ix0 / cells), cy0 = Math.floor(iy0 / cells);
  const lx = Math.floor((ix0 + nx - 1) / cells) - cx0 + 2;
  const ly = Math.floor((iy0 + ny - 1) / cells) - cy0 + 2;
  const lattice = worldRandn(seed, 2, coarseCh, cx0, cy0, lx, ly);

  const smooth = (t) => {
    t = f(t);
    return f(f(t * t) * f(3 - f(2 * t)));
  };
  let k = fineCh * ny * nx;
  for (let c = 0; c < coarseCh; c++) {
    const base = c * ly * lx;
    for (let y = 0; y < ny; y++) {
      const wy = iy0 + y, cy = Math.floor(wy / cells), j = cy - cy0;
      const ty = smooth((wy - cy * cells) / cells);
      for (let x = 0; x < nx; x++) {
        const wx = ix0 + x, cx = Math.floor(wx / cells), i = cx - cx0;
        const tx = smooth((wx - cx * cells) / cells);
        const w00 = f(f(1 - ty) * f(1 - tx)), w01 = f(f(1 - ty) * tx), w10 = f(ty * f(1 - tx)), w11 = f(ty * tx);
        const row0 = base + j * lx + i, row1 = row0 + lx;
        const sum = w00 * lattice[row0] + w01 * lattice[row0 + 1] + w10 * lattice[row1] + w11 * lattice[row1 + 1];
        // Dividing by the weight norm keeps unit variance everywhere.
        out[k++] = sum / Math.sqrt(w00 * w00 + w01 * w01 + w10 * w10 + w11 * w11);
      }
    }
  }
  return out;
}
