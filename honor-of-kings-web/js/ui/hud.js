// HUD：血蓝条、技能冷却、比分、计时、击杀播报、复活倒计时
import { playerHero, canCast } from '../core/game.js';
import { TEAM } from '../core/constants.js';

export function createHud() {
  const $ = id => document.getElementById(id);
  const els = {
    hud: $('hud'),
    hp: $('hp-bar').firstElementChild,
    mp: $('mp-bar').firstElementChild,
    meta: $('hero-meta'),
    scoreBlue: $('score-blue'),
    scoreRed: $('score-red'),
    timer: $('game-timer'),
    feed: $('kill-feed'),
    respawn: $('respawn-overlay'),
    respawnCount: $('respawn-count'),
    skillBtns: [...document.querySelectorAll('.skill-btn[data-skill]')]
  };

  function show() { els.hud.classList.remove('hidden'); }
  function hide() { els.hud.classList.add('hidden'); }

  function setupSkills(hero) {
    els.skillBtns.forEach((btn, i) => {
      btn.querySelector('b').textContent = hero.skills[i].def.name;
      btn.title = `${hero.skills[i].def.name}：${hero.skills[i].def.desc}（耗蓝 ${hero.skills[i].def.mana}）`;
    });
  }

  function update(game) {
    const me = playerHero(game);

    els.hp.style.width = `${Math.max(0, me.hp / me.maxHp) * 100}%`;
    els.mp.style.width = `${Math.max(0, me.mp / me.maxMp) * 100}%`;
    els.meta.innerHTML =
      `${me.name} <b>Lv.${me.level}</b> · 攻击 ${Math.round(me.atk)} · ` +
      `金币 <b>${Math.floor(me.gold)}</b> · ${me.kills}/${me.deaths}`;

    let blue = 0, red = 0;
    for (const h of game.heroes) {
      if (h.team === TEAM.BLUE) blue += h.kills; else red += h.kills;
    }
    els.scoreBlue.textContent = blue;
    els.scoreRed.textContent = red;
    const t = Math.floor(game.time);
    els.timer.textContent =
      `${String(Math.floor(t / 60)).padStart(2, '0')}:${String(t % 60).padStart(2, '0')}`;

    // 技能冷却遮罩
    els.skillBtns.forEach((btn, i) => {
      const sk = me.skills[i];
      const cdEl = btn.querySelector('.cd');
      if (sk.cd > 0) {
        cdEl.style.transform = `scaleY(${sk.cd / sk.def.cd})`;
        cdEl.textContent = sk.cd > 1 ? Math.ceil(sk.cd) : '';
      } else {
        cdEl.style.transform = 'scaleY(0)';
        cdEl.textContent = '';
      }
      btn.classList.toggle('ready', canCast(me, i));
      btn.classList.toggle('noMana', sk.cd <= 0 && me.mp < sk.def.mana);
    });

    // 击杀播报
    els.feed.innerHTML = game.killFeed.map(k => `<div>${k.text}</div>`).join('');

    // 复活倒计时
    if (me.dead) {
      els.respawn.classList.remove('hidden');
      els.respawnCount.textContent = Math.ceil(me.respawnTimer);
    } else {
      els.respawn.classList.add('hidden');
    }
  }

  return { show, hide, update, setupSkills };
}
