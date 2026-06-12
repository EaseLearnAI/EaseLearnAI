// 通用数学工具（纯函数，便于单元测试）
export function clamp(v, min, max) {
  return v < min ? min : v > max ? max : v;
}

export function dist(ax, ay, bx, by) {
  const dx = bx - ax, dy = by - ay;
  return Math.hypot(dx, dy);
}

export function lerp(a, b, t) {
  return a + (b - a) * t;
}

// 返回单位向量 {x, y}；零向量返回 {x:0, y:0}
export function normalize(x, y) {
  const len = Math.hypot(x, y);
  if (len < 1e-9) return { x: 0, y: 0 };
  return { x: x / len, y: y / len };
}

export function angleTo(ax, ay, bx, by) {
  return Math.atan2(by - ay, bx - ax);
}

// 简单可复现的伪随机数生成器（LCG），保证测试可确定
export function makeRng(seed = 42) {
  let s = seed >>> 0;
  return function rng() {
    s = (s * 1664525 + 1013904223) >>> 0;
    return s / 4294967296;
  };
}

let _uid = 0;
export function uid() {
  return ++_uid;
}
