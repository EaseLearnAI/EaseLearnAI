// Canvas 渲染器：地形预渲染 + 摄像机跟随 + 实体/特效/小地图绘制。
// 所有美术素材均为程序化绘制的原创内容，无外部资源依赖。
import { WORLD, LANE_Y, TEAM_COLOR, TEAM } from '../core/constants.js';
import { playerHero } from '../core/game.js';
import { clamp } from '../core/utils.js';

const VIEW_W = 1150; // 摄像机取景宽度（世界坐标），按屏幕等比缩放

export function createRenderer(canvas) {
  const ctx = canvas.getContext('2d');
  let dpr = 1;

  function resize() {
    dpr = Math.min(window.devicePixelRatio || 1, 2);
    canvas.width = Math.floor(innerWidth * dpr);
    canvas.height = Math.floor(innerHeight * dpr);
  }
  window.addEventListener('resize', resize);
  resize();

  const terrain = buildTerrain();

  function render(game) {
    const me = playerHero(game);
    const scale = canvas.width / VIEW_W;
    const viewH = canvas.height / scale;
    const camX = clamp(me.x - VIEW_W / 2, 0, WORLD.w - VIEW_W);
    const camY = clamp(me.y - viewH / 2, 0, Math.max(0, WORLD.h - viewH));

    ctx.setTransform(scale, 0, 0, scale, -camX * scale, -camY * scale);
    ctx.clearRect(camX, camY, VIEW_W, viewH);
    ctx.drawImage(terrain, 0, 0);

    drawBuildings(ctx, game);
    for (const m of game.minions) drawMinion(ctx, m);
    for (const h of game.heroes) if (!h.dead) drawHero(ctx, h, game.time);
    for (const e of game.effects) drawEffect(ctx, e);

    ctx.setTransform(1, 0, 0, 1, 0, 0);
    drawMinimap(ctx, game, canvas);
  }

  return { render, resize };
}

// ---------- 地形（仅生成一次） ----------
function buildTerrain() {
  const c = document.createElement('canvas');
  c.width = WORLD.w; c.height = WORLD.h;
  const g = c.getContext('2d');

  // 草地底色
  const bg = g.createLinearGradient(0, 0, 0, WORLD.h);
  bg.addColorStop(0, '#14331f');
  bg.addColorStop(1, '#0d2417');
  g.fillStyle = bg;
  g.fillRect(0, 0, WORLD.w, WORLD.h);

  // 草丛装饰（固定随机种子保证两次生成一致）
  let s = 7;
  const rnd = () => (s = (s * 16807) % 2147483647) / 2147483647;
  g.fillStyle = 'rgba(34,84,52,.55)';
  for (let i = 0; i < 90; i++) {
    const x = rnd() * WORLD.w, y = rnd() * WORLD.h;
    if (Math.abs(y - LANE_Y) < 160) continue;
    g.beginPath();
    g.ellipse(x, y, 30 + rnd() * 50, 18 + rnd() * 26, rnd() * Math.PI, 0, Math.PI * 2);
    g.fill();
  }

  // 中央河道
  g.fillStyle = 'rgba(37,99,168,.35)';
  g.beginPath();
  g.ellipse(WORLD.w / 2, LANE_Y, 130, 480, 0, 0, Math.PI * 2);
  g.fill();

  // 兵线道路
  g.fillStyle = 'rgba(120,113,90,.4)';
  g.fillRect(0, LANE_Y - 90, WORLD.w, 180);
  g.strokeStyle = 'rgba(229,231,235,.18)';
  g.setLineDash([26, 22]);
  g.lineWidth = 4;
  g.beginPath();
  g.moveTo(0, LANE_Y); g.lineTo(WORLD.w, LANE_Y);
  g.stroke();
  g.setLineDash([]);

  // 双方泉水区域
  for (const team of [0, 1]) {
    const x = team === 0 ? 260 : WORLD.w - 260;
    const grad = g.createRadialGradient(x, LANE_Y, 30, x, LANE_Y, 260);
    grad.addColorStop(0, team === 0 ? 'rgba(59,130,246,.35)' : 'rgba(239,68,68,.35)');
    grad.addColorStop(1, 'rgba(0,0,0,0)');
    g.fillStyle = grad;
    g.beginPath(); g.arc(x, LANE_Y, 260, 0, Math.PI * 2); g.fill();
  }

  // 边界
  g.strokeStyle = 'rgba(251,191,36,.25)';
  g.lineWidth = 10;
  g.strokeRect(5, 5, WORLD.w - 10, WORLD.h - 10);
  return c;
}

// ---------- 实体 ----------
function hpBar(g, x, y, w, ratio, color) {
  g.fillStyle = 'rgba(0,0,0,.6)';
  g.fillRect(x - w / 2, y, w, 7);
  g.fillStyle = color;
  g.fillRect(x - w / 2, y, w * clamp(ratio, 0, 1), 7);
}

function drawBuildings(g, game) {
  for (const t of game.towers) {
    if (t.hp <= 0) {
      g.fillStyle = 'rgba(75,85,99,.5)';
      g.beginPath(); g.arc(t.x, t.y, 30, 0, Math.PI * 2); g.fill();
      continue;
    }
    const color = TEAM_COLOR[t.team];
    g.fillStyle = '#374151';
    g.beginPath(); g.arc(t.x, t.y, 38, 0, Math.PI * 2); g.fill();
    g.fillStyle = color;
    g.beginPath(); g.arc(t.x, t.y, 24, 0, Math.PI * 2); g.fill();
    g.fillStyle = '#f3f4f6';
    g.beginPath(); g.arc(t.x, t.y - 18, 10, 0, Math.PI * 2); g.fill();
    hpBar(g, t.x, t.y - 64, 90, t.hp / t.maxHp, color);
  }
  for (const c of game.crystals) {
    if (c.hp <= 0) continue;
    const color = TEAM_COLOR[c.team];
    g.save();
    g.translate(c.x, c.y);
    g.rotate(Math.PI / 4);
    g.fillStyle = color;
    g.shadowColor = color; g.shadowBlur = 30;
    g.fillRect(-34, -34, 68, 68);
    g.shadowBlur = 0;
    g.fillStyle = 'rgba(255,255,255,.75)';
    g.fillRect(-16, -16, 32, 32);
    g.restore();
    hpBar(g, c.x, c.y - 78, 110, c.hp / c.maxHp, color);
  }
}

function drawMinion(g, m) {
  const color = TEAM_COLOR[m.team];
  g.fillStyle = color;
  g.beginPath();
  if (m.type === 'melee') {
    g.arc(m.x, m.y, 14, 0, Math.PI * 2);
  } else {
    g.moveTo(m.x, m.y - 16);
    g.lineTo(m.x + 14, m.y + 11);
    g.lineTo(m.x - 14, m.y + 11);
    g.closePath();
  }
  g.fill();
  g.strokeStyle = 'rgba(255,255,255,.5)';
  g.lineWidth = 2;
  g.stroke();
  hpBar(g, m.x, m.y - 28, 34, m.hp / m.maxHp, color);
}

function drawHero(g, h, time) {
  const teamColor = TEAM_COLOR[h.team];

  // 脚下光圈（玩家金色高亮）
  g.strokeStyle = h.isPlayer ? '#fbbf24' : teamColor;
  g.lineWidth = h.isPlayer ? 4 : 2.5;
  g.beginPath();
  g.ellipse(h.x, h.y + 16, 26, 12, 0, 0, Math.PI * 2);
  g.stroke();

  // 身体
  g.fillStyle = h.def.color;
  g.beginPath(); g.arc(h.x, h.y, 22, 0, Math.PI * 2); g.fill();
  g.strokeStyle = teamColor; g.lineWidth = 3.5;
  g.stroke();

  // 朝向指示
  g.strokeStyle = 'rgba(255,255,255,.85)';
  g.lineWidth = 4;
  g.beginPath();
  g.moveTo(h.x + Math.cos(h.facing) * 12, h.y + Math.sin(h.facing) * 12);
  g.lineTo(h.x + Math.cos(h.facing) * 26, h.y + Math.sin(h.facing) * 26);
  g.stroke();

  // 回城读条
  if (h.recallTimer >= 0) {
    g.strokeStyle = '#60a5fa';
    g.lineWidth = 5;
    g.beginPath();
    g.arc(h.x, h.y, 34, -Math.PI / 2, -Math.PI / 2 + (1 - h.recallTimer / 3) * Math.PI * 2);
    g.stroke();
  }

  // 名字 / 等级 / 血条蓝条
  g.font = 'bold 14px sans-serif';
  g.textAlign = 'center';
  g.fillStyle = '#f9fafb';
  g.fillText(`${h.name} Lv.${h.level}`, h.x, h.y - 44);
  hpBar(g, h.x, h.y - 38, 60, h.hp / h.maxHp, h.team === TEAM.BLUE ? '#4ade80' : '#f87171');
  g.fillStyle = '#3b82f6';
  g.fillRect(h.x - 30, h.y - 30, 60 * clamp(h.mp / h.maxMp, 0, 1), 3);
}

function drawEffect(g, e) {
  const p = e.ttl / e.maxTtl; // 1 → 0
  g.globalAlpha = p;
  if (e.type === 'hit') {
    g.fillStyle = e.color;
    g.beginPath(); g.arc(e.x, e.y, e.r * (1.4 - p * 0.4), 0, Math.PI * 2); g.fill();
  } else if (e.type === 'ring' || e.type === 'aoe') {
    g.strokeStyle = e.color;
    g.lineWidth = e.type === 'aoe' ? 8 : 5;
    g.beginPath(); g.arc(e.x, e.y, e.r * (1.15 - p * 0.15), 0, Math.PI * 2); g.stroke();
    if (e.type === 'aoe') {
      g.globalAlpha = p * 0.25;
      g.fillStyle = e.color;
      g.fill();
    }
  } else if (e.type === 'slash') {
    g.strokeStyle = e.color;
    g.lineWidth = 7;
    g.beginPath(); g.arc(e.x, e.y, e.r, -0.8, 0.8); g.stroke();
  } else if (e.type === 'beam' && e.tx !== undefined) {
    g.strokeStyle = e.color;
    g.lineWidth = 4;
    g.beginPath(); g.moveTo(e.x, e.y); g.lineTo(e.tx, e.ty); g.stroke();
  }
  g.globalAlpha = 1;
}

// ---------- 小地图 ----------
function drawMinimap(g, game, canvas) {
  const W = Math.min(190, canvas.width * 0.22), H = W * WORLD.h / WORLD.w;
  const x0 = canvas.width - W - 10, y0 = 10;
  const sx = W / WORLD.w, sy = H / WORLD.h;

  g.fillStyle = 'rgba(11,18,32,.78)';
  g.fillRect(x0, y0, W, H);
  g.strokeStyle = 'rgba(148,163,184,.5)';
  g.strokeRect(x0, y0, W, H);

  const dot = (ex, ey, r, color) => {
    g.fillStyle = color;
    g.beginPath(); g.arc(x0 + ex * sx, y0 + ey * sy, r, 0, Math.PI * 2); g.fill();
  };
  for (const t of game.towers) if (t.hp > 0) dot(t.x, t.y, 3.5, TEAM_COLOR[t.team]);
  for (const c of game.crystals) if (c.hp > 0) dot(c.x, c.y, 5, TEAM_COLOR[c.team]);
  for (const m of game.minions) dot(m.x, m.y, 1.5, TEAM_COLOR[m.team]);
  for (const h of game.heroes) {
    if (h.dead) continue;
    dot(h.x, h.y, h.isPlayer ? 5 : 3.5, h.isPlayer ? '#fbbf24' : TEAM_COLOR[h.team]);
  }
}
