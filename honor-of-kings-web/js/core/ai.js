// 机器人英雄 AI：每帧为非玩家英雄填写 input（与玩家走完全相同的处理管线）。
// 行为：低血量撤退回泉水 → 满状态再战；否则追击射程内目标并释放技能；无目标时沿兵线推进。
import { TEAM } from './constants.js';
import { dist, normalize } from './utils.js';
import { basePos, nearestEnemy, canCast } from './game.js';

const AGGRO_RANGE = 560;
const RETREAT_HP = 0.28;   // 低于 28% 血量撤退
const RESUME_HP = 0.85;    // 恢复到 85% 以上重新出击

export function botThink(game, hero) {
  const inp = hero.input;
  inp.mx = 0; inp.my = 0; inp.attack = false; inp.skill = -1; inp.recall = false;
  if (hero.dead) return;

  const base = basePos(hero.team);
  const hpRatio = hero.hp / hero.maxHp;

  // 撤退状态机
  if (hero.aiRetreating && hpRatio >= RESUME_HP) hero.aiRetreating = false;
  if (hpRatio < RETREAT_HP) hero.aiRetreating = true;

  if (hero.aiRetreating) {
    const d = dist(hero.x, hero.y, base.x, base.y);
    if (d > 120) {
      const dir = normalize(base.x - hero.x, base.y - hero.y);
      inp.mx = dir.x; inp.my = dir.y;
    }
    return;
  }

  // 索敌：优先残血敌方英雄，其次最近的任意敌人
  const lowHero = nearestEnemy(game, hero, AGGRO_RANGE,
    e => e.kind === 'hero' && e.hp / e.maxHp < 0.35);
  const target = lowHero || nearestEnemy(game, hero, AGGRO_RANGE);

  if (target) {
    const d = dist(hero.x, hero.y, target.x, target.y);
    // 技能：从大招往普通技能依次尝试（目标在射程内时）
    for (let i = hero.skills.length - 1; i >= 0; i--) {
      const sk = hero.skills[i];
      const rangeOk = sk.def.range === 0 || d <= sk.def.range;
      if (canCast(hero, i) && rangeOk) { inp.skill = i; break; }
    }
    if (d <= hero.range + 15) {
      inp.attack = true;
      // 远程英雄保持距离微调（风筝）
      if (hero.range > 200 && d < hero.range * 0.5) {
        const away = normalize(hero.x - target.x, hero.y - target.y);
        inp.mx = away.x * 0.6; inp.my = away.y * 0.6;
      }
    } else {
      const dir = normalize(target.x - hero.x, target.y - hero.y);
      inp.mx = dir.x; inp.my = dir.y;
      inp.attack = true; // 接近过程中进入射程即自动出手
    }
    return;
  }

  // 无目标：朝敌方水晶方向推进，回到兵线高度
  const enemyBase = basePos(1 - hero.team);
  const dir = normalize(enemyBase.x - hero.x, enemyBase.y - hero.y);
  inp.mx = dir.x; inp.my = dir.y;
}

// 为所有机器人填写输入（在 game.tick 之前调用）
export function runBots(game) {
  for (const h of game.heroes) {
    if (!h.isPlayer) botThink(game, h);
  }
}
