import { test } from 'node:test';
import assert from 'node:assert/strict';
import { createGame, playerHero } from '../js/core/game.js';
import { botThink, runBots } from '../js/core/ai.js';
import { TEAM } from '../js/core/constants.js';

function isolateBot(game) {
  // 取一个蓝方机器人，把所有敌人移到远处避免干扰
  const bot = game.heroes.find(h => !h.isPlayer && h.team === TEAM.BLUE);
  for (const h of game.heroes) {
    if (h.team === TEAM.RED) { h.x = 99999; h.y = 99999; }
  }
  return bot;
}

test('残血机器人撤退回泉水方向', () => {
  const g = createGame('jianhun', 1);
  const bot = isolateBot(g);
  bot.x = 1500; bot.y = 700;
  bot.hp = bot.maxHp * 0.1;
  botThink(g, bot);
  assert.ok(bot.input.mx < 0, '蓝方残血应向左（己方泉水）撤退');
  assert.ok(!bot.input.attack);
});

test('恢复健康后停止撤退继续进攻', () => {
  const g = createGame('jianhun', 1);
  const bot = isolateBot(g);
  bot.x = 1500; bot.y = 700;
  bot.hp = bot.maxHp * 0.1;
  botThink(g, bot); // 进入撤退状态
  assert.ok(bot.aiRetreating);
  bot.hp = bot.maxHp;
  botThink(g, bot);
  assert.ok(!bot.aiRetreating);
  assert.ok(bot.input.mx > 0, '满血后应向敌方推进');
});

test('附近有敌人时锁定并进攻', () => {
  const g = createGame('jianhun', 1);
  const bot = g.heroes.find(h => !h.isPlayer && h.team === TEAM.BLUE);
  bot.x = 1200; bot.y = 700;
  const enemy = g.heroes.find(h => h.team === TEAM.RED);
  enemy.x = 1250; enemy.y = 700; // 进入普攻射程
  botThink(g, bot);
  assert.ok(bot.input.attack, '射程内应发起攻击');
});

test('目标在技能射程内时尝试放技能', () => {
  const g = createGame('jianhun', 1);
  const bot = g.heroes.find(h => !h.isPlayer && h.team === TEAM.BLUE);
  bot.x = 1200; bot.y = 700;
  const enemy = g.heroes.find(h => h.team === TEAM.RED);
  enemy.x = 1300; enemy.y = 700;
  botThink(g, bot);
  assert.ok(bot.input.skill >= 0, '冷却完毕且在射程内应释放技能');
});

test('runBots 不会修改玩家英雄的输入', () => {
  const g = createGame('jianhun', 1);
  const me = playerHero(g);
  me.input.mx = 0.5; me.input.attack = true;
  runBots(g);
  assert.equal(me.input.mx, 0.5);
  assert.equal(me.input.attack, true);
  for (const h of g.heroes) {
    if (!h.isPlayer) assert.ok(h.input !== me.input);
  }
});

test('死亡机器人不产生任何动作', () => {
  const g = createGame('jianhun', 1);
  const bot = g.heroes.find(h => !h.isPlayer);
  bot.dead = true;
  bot.input.mx = 1;
  botThink(g, bot);
  assert.equal(bot.input.mx, 0);
  assert.equal(bot.input.attack, false);
  assert.equal(bot.input.skill, -1);
});
