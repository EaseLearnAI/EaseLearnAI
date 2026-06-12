import { test } from 'node:test';
import assert from 'node:assert/strict';
import {
  physicalDamage, expNeeded, equipTier, equipBonus, heroStats, gainExp
} from '../js/core/combat.js';
import { HERO_DEFS } from '../js/core/heroes.js';
import { MAX_LEVEL, EQUIP_MAX_TIER } from '../js/core/constants.js';

test('护甲减伤公式正确且至少造成 1 点伤害', () => {
  assert.equal(physicalDamage(100, 0), 100);
  assert.equal(physicalDamage(100, 100), 50);
  assert.equal(physicalDamage(200, 100), 100);
  assert.equal(physicalDamage(1, 9999), 1);
  assert.equal(physicalDamage(100, -50), 100); // 负防御不增伤
});

test('升级经验需求随等级增长', () => {
  assert.equal(expNeeded(1), 100);
  assert.ok(expNeeded(5) > expNeeded(2));
});

test('装备阶数按金币计算且有上限', () => {
  assert.equal(equipTier(0), 0);
  assert.equal(equipTier(799), 0);
  assert.equal(equipTier(800), 1);
  assert.equal(equipTier(999999), EQUIP_MAX_TIER);
  const b = equipBonus(2);
  assert.ok(b.atk > 0 && b.def > 0 && b.hp > 0);
});

test('heroStats 随等级和金币提升', () => {
  const def = HERO_DEFS[0];
  const lv1 = heroStats(def, 1, 0);
  const lv5 = heroStats(def, 5, 0);
  const rich = heroStats(def, 1, 1600);
  assert.ok(lv5.maxHp > lv1.maxHp);
  assert.ok(lv5.atk > lv1.atk);
  assert.ok(rich.atk > lv1.atk);
  assert.ok(rich.maxHp > lv1.maxHp);
});

test('gainExp 支持一次性连升多级并封顶', () => {
  const hero = { level: 1, exp: 0 };
  const ups = gainExp(hero, 1000);
  assert.ok(ups >= 3, `应连升多级，实际 ${ups}`);
  assert.equal(hero.level, 1 + ups);

  gainExp(hero, 1e9);
  assert.equal(hero.level, MAX_LEVEL);
  const before = hero.level;
  gainExp(hero, 1e9);
  assert.equal(hero.level, before); // 满级后不再升
});
