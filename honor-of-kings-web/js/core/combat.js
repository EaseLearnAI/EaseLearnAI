// 战斗数值计算（纯函数）
import {
  MAX_LEVEL, EQUIP_GOLD_PER_TIER, EQUIP_MAX_TIER,
  EQUIP_ATK_PER_TIER, EQUIP_DEF_PER_TIER, EQUIP_HP_PER_TIER
} from './constants.js';
import { clamp } from './utils.js';

// 护甲减伤公式：实际伤害 = 攻击 * 100 / (100 + 防御)，至少 1 点
export function physicalDamage(rawAtk, def) {
  return Math.max(1, Math.round(rawAtk * 100 / (100 + Math.max(0, def))));
}

// 升到 level+1 级所需经验
export function expNeeded(level) {
  return 100 + (level - 1) * 90;
}

// 装备阶数：金币每满 EQUIP_GOLD_PER_TIER 自动升一阶
export function equipTier(gold) {
  return clamp(Math.floor(gold / EQUIP_GOLD_PER_TIER), 0, EQUIP_MAX_TIER);
}

export function equipBonus(tier) {
  return {
    atk: tier * EQUIP_ATK_PER_TIER,
    def: tier * EQUIP_DEF_PER_TIER,
    hp: tier * EQUIP_HP_PER_TIER
  };
}

// 根据英雄定义、等级与金币计算总属性
export function heroStats(def, level, gold) {
  const lv = clamp(level, 1, MAX_LEVEL) - 1;
  const eq = equipBonus(equipTier(gold));
  const b = def.base;
  return {
    maxHp: b.hp + b.hpG * lv + eq.hp,
    maxMp: b.mp + b.mpG * lv,
    atk: b.atk + b.atkG * lv + eq.atk,
    def: b.def + b.defG * lv + eq.def,
    range: b.range,
    speed: b.speed,
    atkInterval: b.atkInterval
  };
}

// 给英雄增加经验，处理连续升级；返回升级次数
export function gainExp(hero, amount) {
  let ups = 0;
  hero.exp += amount;
  while (hero.level < MAX_LEVEL && hero.exp >= expNeeded(hero.level)) {
    hero.exp -= expNeeded(hero.level);
    hero.level += 1;
    ups += 1;
  }
  if (hero.level >= MAX_LEVEL) hero.exp = Math.min(hero.exp, expNeeded(MAX_LEVEL) - 1);
  return ups;
}
