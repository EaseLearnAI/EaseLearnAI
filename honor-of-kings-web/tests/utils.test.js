import { test } from 'node:test';
import assert from 'node:assert/strict';
import { clamp, dist, lerp, normalize, makeRng } from '../js/core/utils.js';

test('clamp 限制取值范围', () => {
  assert.equal(clamp(5, 0, 10), 5);
  assert.equal(clamp(-3, 0, 10), 0);
  assert.equal(clamp(99, 0, 10), 10);
});

test('dist 计算欧氏距离', () => {
  assert.equal(dist(0, 0, 3, 4), 5);
  assert.equal(dist(1, 1, 1, 1), 0);
});

test('lerp 线性插值', () => {
  assert.equal(lerp(0, 10, 0.5), 5);
  assert.equal(lerp(2, 4, 0), 2);
  assert.equal(lerp(2, 4, 1), 4);
});

test('normalize 返回单位向量，零向量安全', () => {
  const v = normalize(3, 4);
  assert.ok(Math.abs(v.x - 0.6) < 1e-9);
  assert.ok(Math.abs(v.y - 0.8) < 1e-9);
  assert.deepEqual(normalize(0, 0), { x: 0, y: 0 });
});

test('makeRng 同种子可复现，输出在 [0,1)', () => {
  const a = makeRng(7), b = makeRng(7);
  for (let i = 0; i < 100; i++) {
    const v = a();
    assert.equal(v, b());
    assert.ok(v >= 0 && v < 1);
  }
});
