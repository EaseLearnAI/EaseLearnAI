// 游戏核心状态机：创建对局、每帧推进、技能释放、伤害与击杀结算。
// 不依赖任何浏览器 API，可在 Node 中直接模拟整局游戏（用于自动化测试）。
import {
  WORLD, LANE_Y, TEAM, FIRST_WAVE_DELAY, WAVE_INTERVAL,
  MELEE_PER_WAVE, RANGED_PER_WAVE, MINION_STATS, MINION_HP_GROWTH, MINION_AGGRO,
  TOWER_HP, TOWER_ATK, TOWER_DEF, TOWER_RANGE, CRYSTAL_HP, CRYSTAL_DEF,
  LAYOUT, RESPAWN_BASE, RECALL_TIME, GOLD_PASSIVE, GOLD_MINION, GOLD_HERO_KILL,
  GOLD_TOWER, EXP_MINION, EXP_HERO_KILL, EXP_SHARE_RANGE
} from './constants.js';
import { clamp, dist, normalize, makeRng, uid } from './utils.js';
import { physicalDamage, heroStats, gainExp } from './combat.js';
import { HERO_DEFS, heroDefById } from './heroes.js';

// x 坐标镜像：蓝方用原值，红方用地图宽度减原值
function mirrorX(x, team) {
  return team === TEAM.BLUE ? x : WORLD.w - x;
}

export function basePos(team) {
  return { x: mirrorX(LAYOUT.spawnX, team), y: LANE_Y };
}

function freshInput() {
  return { mx: 0, my: 0, attack: false, skill: -1, recall: false };
}

export function createHero(defId, team, isPlayer, slot = 0) {
  const def = heroDefById(defId);
  const stats = heroStats(def, 1, 0);
  const pos = basePos(team);
  return {
    id: uid(), kind: 'hero', def, heroId: defId, name: def.name,
    team, isPlayer, slot,
    x: pos.x, y: pos.y + (slot - 1) * 90,
    homeY: pos.y + (slot - 1) * 90,
    facing: team === TEAM.BLUE ? 0 : Math.PI,
    level: 1, exp: 0, gold: 0,
    hp: stats.maxHp, maxHp: stats.maxHp,
    mp: stats.maxMp, maxMp: stats.maxMp,
    atk: stats.atk, def_: stats.def, range: stats.range,
    baseSpeed: stats.speed, atkInterval: stats.atkInterval, atkTimer: 0,
    skills: def.skills.map(s => ({ def: s, cd: 0 })),
    buffs: [],
    dead: false, respawnTimer: 0, recallTimer: -1,
    kills: 0, deaths: 0,
    input: freshInput()
  };
}

function createTower(team, x) {
  return {
    id: uid(), kind: 'tower', team, x: mirrorX(x, team), y: LANE_Y,
    hp: TOWER_HP, maxHp: TOWER_HP, atk: TOWER_ATK, def_: TOWER_DEF,
    range: TOWER_RANGE, atkInterval: 1.0, atkTimer: 0
  };
}

function createCrystal(team) {
  return {
    id: uid(), kind: 'crystal', team, x: mirrorX(LAYOUT.crystalX, team), y: LANE_Y,
    hp: CRYSTAL_HP, maxHp: CRYSTAL_HP, def_: CRYSTAL_DEF
  };
}

export function createMinion(team, type, wave, offsetIdx, rng) {
  const s = MINION_STATS[type];
  const pos = basePos(team);
  const jitter = (rng() - 0.5) * 120;
  return {
    id: uid(), kind: 'minion', team, type,
    x: pos.x + (team === TEAM.BLUE ? 60 : -60) - offsetIdx * (team === TEAM.BLUE ? 35 : -35),
    y: LANE_Y + jitter,
    hp: s.hp + wave * MINION_HP_GROWTH, maxHp: s.hp + wave * MINION_HP_GROWTH,
    atk: s.atk, def_: s.def, range: s.range, speed: s.speed,
    atkInterval: s.atkInterval, atkTimer: 0
  };
}

// playerHeroId: 玩家选择的英雄；双方各 3 名英雄（玩家 + 2 个 AI 队友 vs 3 个 AI）
export function createGame(playerHeroId = 'jianhun', seed = 42) {
  const rng = makeRng(seed);
  const ids = HERO_DEFS.map(h => h.id);
  const allyIds = [playerHeroId, ...ids.filter(i => i !== playerHeroId)].slice(0, 3);
  const enemyIds = [...ids];

  const heroes = [];
  allyIds.forEach((hid, i) => heroes.push(createHero(hid, TEAM.BLUE, i === 0, i)));
  enemyIds.forEach((hid, i) => heroes.push(createHero(hid, TEAM.RED, false, i)));

  return {
    time: 0, state: 'playing', winner: -1,
    rng, wave: 0, waveTimer: FIRST_WAVE_DELAY,
    heroes,
    minions: [],
    towers: [createTower(TEAM.BLUE, LAYOUT.towerXs[0]), createTower(TEAM.BLUE, LAYOUT.towerXs[1]),
              createTower(TEAM.RED, LAYOUT.towerXs[0]), createTower(TEAM.RED, LAYOUT.towerXs[1])],
    crystals: [createCrystal(TEAM.BLUE), createCrystal(TEAM.RED)],
    effects: [],   // 渲染特效 {type,x,y,r,ttl,maxTtl,color}
    killFeed: []   // 击杀播报 {text,ttl}
  };
}

export function playerHero(game) {
  return game.heroes.find(h => h.isPlayer);
}

// ---------- 查询 ----------
export function aliveEnemies(game, team) {
  const out = [];
  for (const h of game.heroes) if (h.team !== team && !h.dead) out.push(h);
  for (const m of game.minions) if (m.team !== team && m.hp > 0) out.push(m);
  for (const t of game.towers) if (t.team !== team && t.hp > 0) out.push(t);
  for (const c of game.crystals) if (c.team !== team && c.hp > 0) out.push(c);
  return out;
}

export function nearestEnemy(game, ent, range, filter = null) {
  let best = null, bestD = range;
  for (const e of aliveEnemies(game, ent.team)) {
    if (filter && !filter(e)) continue;
    const d = dist(ent.x, ent.y, e.x, e.y);
    if (d <= bestD) { bestD = d; best = e; }
  }
  return best;
}

export function moveSpeed(hero) {
  let mult = 1;
  for (const b of hero.buffs) {
    if (b.type === 'haste') mult += b.value;
    if (b.type === 'slow') mult -= b.value;
  }
  return Math.max(40, hero.baseSpeed * mult);
}

export function addBuff(hero, type, value, duration) {
  const exist = hero.buffs.find(b => b.type === type);
  if (exist) { exist.value = value; exist.t = duration; }
  else hero.buffs.push({ type, value, t: duration });
}

function addEffect(game, type, x, y, r, ttl, color) {
  game.effects.push({ type, x, y, r, ttl, maxTtl: ttl, color });
}

function announce(game, text) {
  game.killFeed.push({ text, ttl: 5 });
  if (game.killFeed.length > 4) game.killFeed.shift();
}

// ---------- 伤害与击杀结算 ----------
export function damageEntity(game, target, rawAtk, source) {
  if (target.hp <= 0 || (target.kind === 'hero' && target.dead)) return 0;
  const dmg = physicalDamage(rawAtk, target.def_ ?? 0);
  target.hp -= dmg;
  if (target.kind === 'hero') target.recallTimer = -1; // 受击打断回城
  addEffect(game, 'hit', target.x, target.y, 18, 0.25, '#fff');
  if (target.hp <= 0) onKill(game, target, source);
  return dmg;
}

function rewardTeamExp(game, team, x, y, amount) {
  const near = game.heroes.filter(h => h.team === team && !h.dead &&
    dist(h.x, h.y, x, y) <= EXP_SHARE_RANGE);
  for (const h of near) {
    if (gainExp(h, Math.round(amount / Math.max(1, near.length))) > 0) applyLevelStats(h);
  }
}

function applyLevelStats(hero) {
  const s = heroStats(hero.def, hero.level, hero.gold);
  const hpRatio = hero.maxHp > 0 ? hero.hp / hero.maxHp : 1;
  const mpRatio = hero.maxMp > 0 ? hero.mp / hero.maxMp : 1;
  hero.maxHp = s.maxHp; hero.maxMp = s.maxMp;
  hero.atk = s.atk; hero.def_ = s.def;
  // 升级 / 出装时按比例保留当前血蓝
  hero.hp = Math.min(hero.maxHp, Math.max(hero.hp, hero.maxHp * hpRatio));
  hero.mp = Math.min(hero.maxMp, hero.maxMp * mpRatio);
}

export function addGold(hero, amount) {
  hero.gold += amount;
  applyLevelStats(hero);
}

function onKill(game, target, source) {
  target.hp = 0;
  const killerHero = source && source.kind === 'hero' ? source : null;
  if (target.kind === 'minion') {
    if (killerHero) addGold(killerHero, GOLD_MINION);
    rewardTeamExp(game, 1 - target.team, target.x, target.y, EXP_MINION);
  } else if (target.kind === 'hero') {
    target.dead = true;
    target.deaths += 1;
    target.respawnTimer = RESPAWN_BASE + target.level * 2;
    target.buffs = [];
    if (killerHero) {
      killerHero.kills += 1;
      addGold(killerHero, GOLD_HERO_KILL);
      announce(game, `${killerHero.name} 击杀了 ${target.name}！`);
    } else {
      announce(game, `${target.name} 阵亡了`);
    }
    rewardTeamExp(game, 1 - target.team, target.x, target.y, EXP_HERO_KILL);
    addEffect(game, 'aoe', target.x, target.y, 60, 0.6, '#f87171');
  } else if (target.kind === 'tower') {
    for (const h of game.heroes) if (h.team !== target.team) addGold(h, GOLD_TOWER);
    announce(game, `${target.team === TEAM.BLUE ? '蓝方' : '红方'}防御塔被摧毁！`);
    addEffect(game, 'aoe', target.x, target.y, 120, 1, '#fbbf24');
  } else if (target.kind === 'crystal') {
    game.state = 'ended';
    game.winner = 1 - target.team;
    addEffect(game, 'aoe', target.x, target.y, 200, 2, '#fbbf24');
  }
}

// ---------- 技能 ----------
export function canCast(hero, idx) {
  const sk = hero.skills[idx];
  return !!sk && sk.cd <= 0 && hero.mp >= sk.def.mana && !hero.dead;
}

// 释放技能；成功返回 true。所有技能效果在此集中实现。
export function castSkill(game, hero, idx) {
  if (!canCast(hero, idx)) return false;
  const sk = hero.skills[idx];
  const key = sk.def.key;
  const color = hero.def.color;

  // 自我增益类技能不需要目标
  if (key === 'haste') {
    addBuff(hero, 'haste', 0.6, 3);
    addEffect(game, 'ring', hero.x, hero.y, 70, 0.5, color);
  } else if (key === 'spin') {
    const hit = aliveEnemies(game, hero.team)
      .filter(e => dist(hero.x, hero.y, e.x, e.y) <= sk.def.range);
    if (!hit.length) return false;
    for (const e of hit) damageEntity(game, e, hero.atk * 0.85 + 45, hero);
    addEffect(game, 'ring', hero.x, hero.y, sk.def.range, 0.4, color);
  } else if (key === 'dash') {
    const target = nearestEnemy(game, hero, sk.def.range);
    if (!target) return false;
    const dir = normalize(target.x - hero.x, target.y - hero.y);
    hero.x = clamp(hero.x + dir.x * 240, 30, WORLD.w - 30);
    hero.y = clamp(hero.y + dir.y * 240, 30, WORLD.h - 30);
    damageEntity(game, target, hero.atk * 1.3 + 60, hero);
    addEffect(game, 'slash', target.x, target.y, 70, 0.35, color);
  } else if (key === 'execute') {
    const hit = aliveEnemies(game, hero.team)
      .filter(e => dist(hero.x, hero.y, e.x, e.y) <= sk.def.range);
    if (!hit.length) return false;
    for (const e of hit) damageEntity(game, e, hero.atk * 2.0 + 150, hero);
    hero.hp = Math.min(hero.maxHp, hero.hp + hero.maxHp * 0.25);
    addEffect(game, 'ring', hero.x, hero.y, sk.def.range, 0.6, color);
  } else if (key === 'bolt' || key === 'pierce') {
    const target = nearestEnemy(game, hero, sk.def.range);
    if (!target) return false;
    const mult = key === 'bolt' ? 1.5 : 1.8;
    damageEntity(game, target, hero.atk * mult + 55, hero);
    addEffect(game, 'beam', hero.x, hero.y, 0, 0.3, color);
    game.effects[game.effects.length - 1].tx = target.x;
    game.effects[game.effects.length - 1].ty = target.y;
  } else if (key === 'field') {
    const target = nearestEnemy(game, hero, sk.def.range);
    if (!target) return false;
    const hit = aliveEnemies(game, hero.team)
      .filter(e => dist(target.x, target.y, e.x, e.y) <= 200);
    for (const e of hit) {
      damageEntity(game, e, hero.atk * 1.0 + 60, hero);
      if (e.kind === 'hero') addBuff(e, 'slow', 0.4, 2);
    }
    addEffect(game, 'aoe', target.x, target.y, 200, 0.7, color);
  } else if (key === 'meteor' || key === 'volley') {
    const target = nearestEnemy(game, hero, sk.def.range);
    if (!target) return false;
    const radius = key === 'meteor' ? 260 : 240;
    const mult = key === 'meteor' ? 2.2 : 1.6;
    const bonus = key === 'meteor' ? 200 : 120;
    const hit = aliveEnemies(game, hero.team)
      .filter(e => dist(target.x, target.y, e.x, e.y) <= radius);
    for (const e of hit) damageEntity(game, e, hero.atk * mult + bonus, hero);
    addEffect(game, 'aoe', target.x, target.y, radius, 0.8, color);
  } else {
    return false;
  }

  sk.cd = sk.def.cd;
  hero.mp -= sk.def.mana;
  hero.recallTimer = -1;
  return true;
}

// ---------- 兵线 ----------
export function spawnWave(game) {
  game.wave += 1;
  for (const team of [TEAM.BLUE, TEAM.RED]) {
    for (let i = 0; i < MELEE_PER_WAVE; i++) {
      game.minions.push(createMinion(team, 'melee', game.wave, i, game.rng));
    }
    for (let i = 0; i < RANGED_PER_WAVE; i++) {
      game.minions.push(createMinion(team, 'ranged', game.wave, MELEE_PER_WAVE + i, game.rng));
    }
  }
}

// ---------- 单位行为 ----------
function tickHero(game, h, dt) {
  if (h.dead) {
    h.respawnTimer -= dt;
    if (h.respawnTimer <= 0) {
      h.dead = false;
      const pos = basePos(h.team);
      h.x = pos.x; h.y = h.homeY;
      h.hp = h.maxHp; h.mp = h.maxMp;
      h.input = freshInput();
    }
    return;
  }

  // 冷却 / 增益衰减
  for (const sk of h.skills) sk.cd = Math.max(0, sk.cd - dt);
  h.buffs = h.buffs.filter(b => (b.t -= dt) > 0);
  h.atkTimer = Math.max(0, h.atkTimer - dt);

  // 回血回蓝（泉水内大幅强化）
  const base = basePos(h.team);
  const atBase = dist(h.x, h.y, base.x, base.y) < 260;
  h.hp = Math.min(h.maxHp, h.hp + h.maxHp * (atBase ? 0.08 : 0.005) * dt);
  h.mp = Math.min(h.maxMp, h.mp + (atBase ? 60 : 7) * dt);

  // 被动金币
  h.goldAcc = (h.goldAcc || 0) + GOLD_PASSIVE * dt;
  if (h.goldAcc >= 1) {
    const g = Math.floor(h.goldAcc);
    h.goldAcc -= g;
    addGold(h, g);
  }

  const inp = h.input;

  // 回城
  if (inp.recall && h.recallTimer < 0) h.recallTimer = RECALL_TIME;
  if (h.recallTimer >= 0) {
    if (inp.mx !== 0 || inp.my !== 0) {
      h.recallTimer = -1; // 移动打断回城
    } else {
      h.recallTimer -= dt;
      if (h.recallTimer <= 0) {
        h.x = base.x; h.y = h.homeY;
        h.recallTimer = -1;
      }
      inp.recall = false;
      return; // 读条期间不做其他动作
    }
  }
  inp.recall = false;

  // 移动
  if (inp.mx !== 0 || inp.my !== 0) {
    const dir = normalize(inp.mx, inp.my);
    const sp = moveSpeed(h);
    h.x = clamp(h.x + dir.x * sp * dt, 30, WORLD.w - 30);
    h.y = clamp(h.y + dir.y * sp * dt, 30, WORLD.h - 30);
    h.facing = Math.atan2(dir.y, dir.x);
  }

  // 技能
  if (inp.skill >= 0) {
    castSkill(game, h, inp.skill);
    inp.skill = -1;
  }

  // 普攻（自动索敌：范围内最近的敌人）
  if (inp.attack && h.atkTimer <= 0) {
    const target = nearestEnemy(game, h, h.range + 25);
    if (target) {
      damageEntity(game, target, h.atk, h);
      h.atkTimer = h.atkInterval;
      h.facing = Math.atan2(target.y - h.y, target.x - h.x);
      addEffect(game, 'beam', h.x, h.y, 0, 0.15, '#e5e7eb');
      game.effects[game.effects.length - 1].tx = target.x;
      game.effects[game.effects.length - 1].ty = target.y;
    }
  }
}

function tickMinion(game, m, dt) {
  m.atkTimer = Math.max(0, m.atkTimer - dt);
  const target = nearestEnemy(game, m, MINION_AGGRO);
  if (target) {
    const d = dist(m.x, m.y, target.x, target.y);
    if (d <= m.range + 20) {
      if (m.atkTimer <= 0) {
        damageEntity(game, target, m.atk, m);
        m.atkTimer = m.atkInterval;
      }
      return;
    }
    const dir = normalize(target.x - m.x, target.y - m.y);
    m.x += dir.x * m.speed * dt;
    m.y += dir.y * m.speed * dt;
    return;
  }
  // 无目标：沿兵线推进
  const dirX = m.team === TEAM.BLUE ? 1 : -1;
  m.x += dirX * m.speed * dt;
  m.y += (LANE_Y - m.y) * 0.5 * dt;
}

function tickTower(game, t, dt) {
  t.atkTimer = Math.max(0, t.atkTimer - dt);
  if (t.atkTimer > 0) return;
  // 优先攻击小兵，其次英雄
  const target = nearestEnemy(game, t, t.range, e => e.kind === 'minion')
    || nearestEnemy(game, t, t.range, e => e.kind === 'hero');
  if (target) {
    damageEntity(game, target, t.atk, t);
    t.atkTimer = t.atkInterval;
    addEffect(game, 'beam', t.x, t.y - 40, 0, 0.2, '#fbbf24');
    game.effects[game.effects.length - 1].tx = target.x;
    game.effects[game.effects.length - 1].ty = target.y;
  }
}

// ---------- 主循环 ----------
export function tick(game, dt) {
  if (game.state !== 'playing') return;
  game.time += dt;

  game.waveTimer -= dt;
  if (game.waveTimer <= 0) {
    spawnWave(game);
    game.waveTimer = WAVE_INTERVAL;
  }

  for (const h of game.heroes) {
    tickHero(game, h, dt);
    if (game.state !== 'playing') break;
  }
  if (game.state === 'playing') {
    for (const m of game.minions) {
      if (m.hp > 0) tickMinion(game, m, dt);
      if (game.state !== 'playing') break;
    }
  }
  if (game.state === 'playing') {
    for (const t of game.towers) {
      if (t.hp > 0) tickTower(game, t, dt);
      if (game.state !== 'playing') break;
    }
  }

  game.minions = game.minions.filter(m => m.hp > 0);
  for (const e of game.effects) e.ttl -= dt;
  game.effects = game.effects.filter(e => e.ttl > 0);
  for (const k of game.killFeed) k.ttl -= dt;
  game.killFeed = game.killFeed.filter(k => k.ttl > 0);
}
