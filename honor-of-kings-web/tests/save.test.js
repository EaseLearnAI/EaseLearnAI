import { test } from 'node:test';
import assert from 'node:assert/strict';
import {
  defaultProfile, loadProfile, saveProfile, recordMatch,
  exportCode, importCode, mergeProfiles
} from '../js/core/save.js';

// 内存版 localStorage，用于在 Node 中测试三端同步逻辑
function memStorage() {
  const map = new Map();
  return {
    getItem: k => (map.has(k) ? map.get(k) : null),
    setItem: (k, v) => map.set(k, String(v))
  };
}

test('空存储返回默认档案', () => {
  const p = loadProfile(memStorage());
  assert.deepEqual(p, defaultProfile());
});

test('保存后能完整读回', () => {
  const st = memStorage();
  const p = defaultProfile();
  p.wins = 3; p.totalKills = 27;
  saveProfile(st, p);
  const back = loadProfile(st);
  assert.equal(back.wins, 3);
  assert.equal(back.totalKills, 27);
  assert.ok(back.updatedAt > 0);
});

test('损坏的存档安全回退到默认档案', () => {
  const st = memStorage();
  st.setItem('honor-arena-profile-v1', '{{{not json');
  assert.deepEqual(loadProfile(st), defaultProfile());
});

test('recordMatch 正确累计战绩', () => {
  const p = defaultProfile();
  recordMatch(p, { won: true, kills: 8, deaths: 2, heroId: 'fengyu' });
  recordMatch(p, { won: false, kills: 3, deaths: 5, heroId: 'fengyu' });
  assert.equal(p.wins, 1);
  assert.equal(p.losses, 1);
  assert.equal(p.totalKills, 11);
  assert.equal(p.totalDeaths, 7);
  assert.equal(p.favoriteHero, 'fengyu');
});

test('同步码导出导入往返一致（含中文昵称）', () => {
  const p = defaultProfile();
  p.name = '王者召唤师·风';
  p.wins = 12; p.totalKills = 150;
  const code = exportCode(p);
  const back = importCode(code);
  assert.ok(back);
  assert.equal(back.name, '王者召唤师·风');
  assert.equal(back.wins, 12);
  assert.equal(back.totalKills, 150);
});

test('非法同步码返回 null 而非抛错', () => {
  assert.equal(importCode('随便打的字'), null);
  assert.equal(importCode(''), null);
  assert.equal(importCode(Buffer.from('{"v":99}').toString('base64')), null);
});

test('mergeProfiles 跨设备合并不丢战绩', () => {
  const phone = { ...defaultProfile(), wins: 10, losses: 2, totalKills: 80, updatedAt: 2000, name: '手机端' };
  const pc = { ...defaultProfile(), wins: 7, losses: 5, totalKills: 95, updatedAt: 1000, name: '电脑端' };
  const merged = mergeProfiles(phone, pc);
  assert.equal(merged.name, '手机端');       // 取更新者的基础信息
  assert.equal(merged.wins, 10);             // 战绩取较大值
  assert.equal(merged.losses, 5);
  assert.equal(merged.totalKills, 95);
});
