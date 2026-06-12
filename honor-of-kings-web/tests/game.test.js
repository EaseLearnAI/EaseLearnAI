import { test } from 'node:test';
import assert from 'node:assert/strict';
import {
  createGame, tick, playerHero, damageEntity, castSkill, canCast,
  nearestEnemy, addBuff, moveSpeed, basePos, spawnWave
} from '../js/core/game.js';
import { runBots } from '../js/core/ai.js';
import {
  TEAM, FIRST_WAVE_DELAY, MELEE_PER_WAVE, RANGED_PER_WAVE,
  GOLD_MINION, GOLD_HERO_KILL, RECALL_TIME
} from '../js/core/constants.js';

function step(game, seconds, dt = 0.1) {
  for (let t = 0; t < seconds; t += dt) tick(game, dt);
}

test('createGame 初始布局正确', () => {
  const g = createGame('xinglan', 1);
  assert.equal(g.heroes.length, 6);
  assert.equal(g.heroes.filter(h => h.isPlayer).length, 1);
  assert.equal(playerHero(g).heroId, 'xinglan');
  assert.equal(playerHero(g).team, TEAM.BLUE);
  assert.equal(g.towers.length, 4);
  assert.equal(g.crystals.length, 2);
  assert.equal(g.state, 'playing');
  assert.equal(g.minions.length, 0);
});

test('首波兵按时双方对称出兵', () => {
  const g = createGame('jianhun', 1);
  step(g, FIRST_WAVE_DELAY - 0.5);
  assert.equal(g.minions.length, 0);
  step(g, 1);
  const perWave = MELEE_PER_WAVE + RANGED_PER_WAVE;
  assert.equal(g.minions.filter(m => m.team === TEAM.BLUE).length, perWave);
  assert.equal(g.minions.filter(m => m.team === TEAM.RED).length, perWave);
});

test('击杀小兵获得金币和经验', () => {
  const g = createGame('jianhun', 1);
  spawnWave(g);
  const me = playerHero(g);
  const target = g.minions.find(m => m.team === TEAM.RED);
  // 把目标拉到玩家身边以便共享经验
  target.x = me.x + 50; target.y = me.y;
  const goldBefore = me.gold, expBefore = me.exp + me.level * 1000;
  damageEntity(g, target, 1e6, me);
  assert.equal(target.hp, 0);
  assert.ok(me.gold >= goldBefore + GOLD_MINION);
  assert.ok(me.exp + me.level * 1000 > expBefore, '应获得经验');
});

test('击杀英雄：计入战绩、死亡复活流程完整', () => {
  const g = createGame('jianhun', 1);
  const me = playerHero(g);
  const enemy = g.heroes.find(h => h.team === TEAM.RED);
  damageEntity(g, enemy, 1e7, me);

  assert.equal(me.kills, 1);
  assert.equal(enemy.deaths, 1);
  assert.ok(enemy.dead);
  assert.ok(enemy.respawnTimer > 0);
  assert.ok(me.gold >= GOLD_HERO_KILL);
  assert.ok(g.killFeed.length > 0);

  // 等待复活
  step(g, enemy.respawnTimer + 1);
  assert.ok(!enemy.dead, '复活计时结束后应复活');
  assert.equal(enemy.hp, enemy.maxHp);
  const base = basePos(TEAM.RED);
  assert.ok(Math.abs(enemy.x - base.x) < 5, '应在泉水复活');
});

test('摧毁水晶立即结束对局并判定胜方', () => {
  const g = createGame('jianhun', 1);
  const redCrystal = g.crystals.find(c => c.team === TEAM.RED);
  damageEntity(g, redCrystal, 1e7, playerHero(g));
  assert.equal(g.state, 'ended');
  assert.equal(g.winner, TEAM.BLUE);
  // 结束后 tick 不再推进
  const t = g.time;
  tick(g, 1);
  assert.equal(g.time, t);
});

test('技能：消耗蓝量、进入冷却、冷却随时间恢复', () => {
  const g = createGame('fengyu', 1);
  const me = playerHero(g);
  const enemy = g.heroes.find(h => h.team === TEAM.RED);
  enemy.x = me.x + 200; enemy.y = me.y;

  assert.ok(canCast(me, 0));
  const mpBefore = me.mp;
  const hpBefore = enemy.hp;
  assert.ok(castSkill(g, me, 0), '技能应释放成功');
  assert.ok(enemy.hp < hpBefore, '目标应受到伤害');
  assert.ok(me.mp < mpBefore, '应消耗蓝量');
  assert.ok(me.skills[0].cd > 0);
  assert.ok(!canCast(me, 0), '冷却中不可再次释放');

  // 原地等待冷却结束（不移动避免打断任何状态）
  step(g, me.skills[0].def.cd + 0.5);
  assert.ok(canCast(me, 0), '冷却结束后可再次释放');
});

test('蓝量不足时无法释放技能', () => {
  const g = createGame('xinglan', 1);
  const me = playerHero(g);
  me.mp = 0;
  assert.ok(!canCast(me, 0));
  assert.ok(!castSkill(g, me, 0));
});

test('加速/减速 buff 影响移速并随时间消失', () => {
  const g = createGame('fengyu', 1);
  const me = playerHero(g);
  const base = moveSpeed(me);
  addBuff(me, 'haste', 0.6, 2);
  assert.ok(moveSpeed(me) > base);
  addBuff(me, 'slow', 0.9, 2);
  assert.ok(moveSpeed(me) >= 40, '移速有下限');
  step(g, 3);
  assert.equal(me.buffs.length, 0, 'buff 到期应清除');
  assert.equal(moveSpeed(me), base);
});

test('回城：读条完成传送回泉水，移动会打断', () => {
  const g = createGame('jianhun', 1);
  const me = playerHero(g);
  // 站在安全位置（敌方塔射程之外），避免受击打断干扰本测试
  me.x = 800; me.y = 700;

  // 移动打断回城
  me.input.recall = true;
  tick(g, 0.1);
  me.input.mx = 1;
  tick(g, 0.1);
  assert.equal(me.recallTimer, -1, '移动应打断回城');
  assert.ok(me.x > 790, '未被传送');

  // 静止读条完成
  me.input.mx = 0; me.input.my = 0;
  me.input.recall = true;
  step(g, RECALL_TIME + 0.5);
  const base = basePos(TEAM.BLUE);
  assert.ok(Math.abs(me.x - base.x) < 5, '读条完成应回到泉水');
});

test('防御塔自动攻击进入射程的敌人', () => {
  const g = createGame('jianhun', 1);
  const tower = g.towers.find(t => t.team === TEAM.RED);
  const me = playerHero(g);
  me.x = tower.x - 200; me.y = tower.y;
  const hpBefore = me.hp;
  step(g, 1.5);
  assert.ok(me.hp < hpBefore, '塔下站立应被攻击');
});

test('泉水内回血明显快于野外', () => {
  const g = createGame('jianhun', 1);
  const me = playerHero(g);
  me.hp = me.maxHp * 0.3;
  const atBaseStart = me.hp;
  step(g, 2);
  const baseHealed = me.hp - atBaseStart;

  const g2 = createGame('jianhun', 1);
  const me2 = playerHero(g2);
  me2.x = 1200; me2.hp = me2.maxHp * 0.3;
  const fieldStart = me2.hp;
  step(g2, 2);
  const fieldHealed = me2.hp - fieldStart;
  assert.ok(baseHealed > fieldHealed * 3, '泉水回复应远快于野外');
});

test('nearestEnemy 选中最近目标且尊重过滤器', () => {
  const g = createGame('jianhun', 1);
  const me = playerHero(g);
  const e1 = g.heroes.find(h => h.team === TEAM.RED && h.slot === 0);
  const e2 = g.heroes.find(h => h.team === TEAM.RED && h.slot === 1);
  e1.x = me.x + 100; e1.y = me.y;
  e2.x = me.x + 300; e2.y = me.y;
  assert.equal(nearestEnemy(g, me, 500), e1);
  assert.equal(nearestEnemy(g, me, 500, t => t === e2), e2);
  assert.equal(nearestEnemy(g, me, 50), null);
});

test('整局模拟：六个 AI 自动对战最终分出胜负', () => {
  const g = createGame('jianhun', 99);
  const me = playerHero(g);
  me.isPlayer = false; // 让玩家英雄也交给 AI，全自动打完整局
  const dt = 0.1;
  const maxSeconds = 60 * 25;
  for (let t = 0; t < maxSeconds && g.state === 'playing'; t += dt) {
    runBots(g);
    tick(g, dt);
  }
  assert.equal(g.state, 'ended', '25 分钟内应分出胜负');
  assert.ok(g.winner === TEAM.BLUE || g.winner === TEAM.RED);
  const loserCrystal = g.crystals.find(c => c.team !== g.winner);
  assert.equal(loserCrystal.hp, 0, '败方水晶应被摧毁');
});
