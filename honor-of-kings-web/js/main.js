// 入口：英雄选择 → 对局循环（AI + 逻辑 tick + 渲染 + HUD）→ 结算与存档同步
import { HERO_DEFS } from './core/heroes.js';
import { createGame, tick, playerHero } from './core/game.js';
import { runBots } from './core/ai.js';
import {
  loadProfile, saveProfile, recordMatch,
  exportCode, importCode, mergeProfiles
} from './core/save.js';
import { createInput, applyInput } from './ui/input.js';
import { createRenderer } from './ui/renderer.js';
import { createHud } from './ui/hud.js';
import { TEAM } from './core/constants.js';

const $ = id => document.getElementById(id);

let profile = loadProfile(localStorage);
let selectedHero = profile.favoriteHero || 'jianhun';
let game = null;
let rafId = 0;
let lastT = 0;

const input = createInput();
const renderer = createRenderer($('game'));
const hud = createHud();

// ---------- 英雄选择界面 ----------
function renderProfileBar() {
  const total = profile.wins + profile.losses;
  $('profile-bar').innerHTML = total
    ? `${profile.name} · 战绩 <b>${profile.wins}胜 ${profile.losses}负</b> · 总击杀 <b>${profile.totalKills}</b>`
    : `${profile.name} · 欢迎来到王者竞技场，开始你的第一场对局吧！`;
}

function renderHeroList() {
  $('hero-list').innerHTML = HERO_DEFS.map(h => `
    <div class="hero-card ${h.id === selectedHero ? 'selected' : ''}" data-id="${h.id}" style="color:${h.color}">
      <div class="avatar">${h.name[0]}</div>
      <div><b>${h.name}</b></div>
      <div class="role">${h.role}</div>
      <div class="desc">${h.desc}</div>
    </div>`).join('');
  for (const card of document.querySelectorAll('.hero-card')) {
    card.addEventListener('click', () => {
      selectedHero = card.dataset.id;
      renderHeroList();
    });
  }
}

// ---------- 同步码 ----------
$('btn-export').addEventListener('click', () => {
  const code = exportCode(profile);
  $('sync-code').value = code;
  if (navigator.clipboard) navigator.clipboard.writeText(code).catch(() => {});
});
$('btn-import').addEventListener('click', () => {
  const incoming = importCode($('sync-code').value);
  if (!incoming) {
    $('sync-code').value = '';
    $('sync-code').placeholder = '同步码无效，请检查后重试';
    return;
  }
  profile = mergeProfiles(profile, incoming);
  saveProfile(localStorage, profile);
  selectedHero = profile.favoriteHero || selectedHero;
  renderProfileBar();
  renderHeroList();
  $('sync-code').value = '';
  $('sync-code').placeholder = '同步成功！战绩已合并';
});

// ---------- 对局流程 ----------
function startGame() {
  game = createGame(selectedHero, Date.now() & 0xffff);
  $('select-screen').classList.add('hidden');
  $('end-screen').classList.add('hidden');
  hud.setupSkills(playerHero(game));
  hud.show();
  lastT = performance.now();
  cancelAnimationFrame(rafId);
  rafId = requestAnimationFrame(loop);
}

function loop(now) {
  const dt = Math.min((now - lastT) / 1000, 0.05); // 防止切后台后的大步长
  lastT = now;

  if (game.state === 'playing') {
    applyInput(input, playerHero(game), dt);
    runBots(game);
    tick(game, dt);
  }

  renderer.render(game);
  hud.update(game);

  if (game.state === 'ended') {
    endGame();
    return;
  }
  rafId = requestAnimationFrame(loop);
}

function endGame() {
  const me = playerHero(game);
  const won = game.winner === TEAM.BLUE;
  recordMatch(profile, { won, kills: me.kills, deaths: me.deaths, heroId: me.heroId });
  saveProfile(localStorage, profile);

  hud.hide();
  const title = $('end-title');
  title.textContent = won ? '胜 利' : '失 败';
  title.className = won ? 'win' : 'lose';
  const mins = Math.floor(game.time / 60), secs = Math.floor(game.time % 60);
  $('end-stats').innerHTML =
    `对局时长 ${mins} 分 ${secs} 秒<br>` +
    `${me.name} · ${me.kills} 击杀 / ${me.deaths} 阵亡 · 等级 ${me.level} · 金币 ${Math.floor(me.gold)}<br>` +
    `生涯战绩：${profile.wins} 胜 ${profile.losses} 负`;
  $('end-screen').classList.remove('hidden');
}

$('btn-start').addEventListener('click', startGame);
$('btn-again').addEventListener('click', () => {
  $('end-screen').classList.add('hidden');
  $('select-screen').classList.remove('hidden');
  renderProfileBar();
  renderHeroList();
});

renderProfileBar();
renderHeroList();
