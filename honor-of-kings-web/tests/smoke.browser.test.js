// 浏览器端冒烟测试：用最小 DOM/Canvas 仿真加载完整入口 main.js，
// 模拟点击"开始对战"并推进数百帧，验证 UI 接线 / 渲染 / HUD 无运行时错误。
import { test } from 'node:test';
import assert from 'node:assert/strict';

// ---------- 最小 DOM 仿真 ----------
function fakeCtx() {
  const gradient = { addColorStop() {} };
  return new Proxy({ canvas: null }, {
    get(t, p) {
      if (p === 'createLinearGradient' || p === 'createRadialGradient') return () => gradient;
      if (p in t) return t[p];
      t[p] = () => undefined;
      return t[p];
    },
    set(t, p, v) { t[p] = v; return true; }
  });
}

function fakeElement(id = '') {
  const el = {
    id,
    listeners: {},
    dataset: {},
    style: {},
    classList: {
      _set: new Set(),
      add(c) { this._set.add(c); },
      remove(c) { this._set.delete(c); },
      toggle(c, on) { on ? this._set.add(c) : this._set.delete(c); },
      contains(c) { return this._set.has(c); }
    },
    textContent: '',
    innerHTML: '',
    title: '',
    value: '',
    placeholder: '',
    width: 0, height: 0,
    addEventListener(type, fn) { (el.listeners[type] ||= []).push(fn); },
    dispatch(type, ev = {}) {
      for (const fn of el.listeners[type] || []) fn({ preventDefault() {}, ...ev });
    },
    getContext: () => fakeCtx(),
    getBoundingClientRect: () => ({ left: 0, top: 0, width: 120, height: 120 }),
    querySelector(sel) {
      el._children ||= {};
      return (el._children[sel] ||= fakeElement(sel));
    },
    querySelectorAll: () => []
  };
  Object.defineProperty(el, 'firstElementChild', {
    get() { return el.querySelector(':first-child'); }
  });
  return el;
}

const elements = new Map();
const byId = id => {
  if (!elements.has(id)) elements.set(id, fakeElement(id));
  return elements.get(id);
};

const skillBtns = [0, 1, 2].map(i => {
  const el = fakeElement(`skill-${i}`);
  el.dataset.skill = String(i);
  return el;
});

const windowListeners = {};
const rafQueue = [];

function setupGlobals() {
  globalThis.window = globalThis;
  globalThis.innerWidth = 1280;
  globalThis.innerHeight = 720;
  globalThis.devicePixelRatio = 1;
  globalThis.addEventListener = (type, fn) => { (windowListeners[type] ||= []).push(fn); };
  globalThis.requestAnimationFrame = fn => { rafQueue.push(fn); return rafQueue.length; };
  globalThis.cancelAnimationFrame = () => {};
  // Node 自带只读 navigator，需用 defineProperty 覆盖
  Object.defineProperty(globalThis, 'navigator', {
    value: { clipboard: null }, configurable: true
  });
  const store = new Map();
  globalThis.localStorage = {
    getItem: k => (store.has(k) ? store.get(k) : null),
    setItem: (k, v) => store.set(k, String(v))
  };
  globalThis.document = {
    getElementById: byId,
    createElement: () => fakeElement('created-canvas'),
    querySelectorAll(sel) {
      if (sel.includes('skill-btn')) return skillBtns;
      return [];
    }
  };
}

function dispatchWindow(type, ev) {
  for (const fn of windowListeners[type] || []) fn({ preventDefault() {}, ...ev });
}

test('浏览器入口冒烟：开局并运行 600 帧无异常', async () => {
  setupGlobals();
  await import('../js/main.js');

  // 选人界面已渲染
  assert.ok(byId('hero-list').innerHTML.includes('hero-card'), '英雄列表应已渲染');
  assert.ok(byId('profile-bar').innerHTML.length > 0, '档案栏应已渲染');

  // 点击开始对战
  byId('btn-start').dispatch('click');
  assert.ok(byId('select-screen').classList.contains('hidden'), '选人界面应隐藏');
  assert.ok(!byId('hud').classList.contains('hidden'), 'HUD 应显示');
  assert.equal(rafQueue.length, 1, '游戏循环应已启动');

  // 模拟玩家键盘操作：向右移动 + 普攻 + 放技能
  dispatchWindow('keydown', { code: 'KeyD', repeat: false });
  dispatchWindow('keydown', { code: 'Space', repeat: false });
  dispatchWindow('keydown', { code: 'KeyK', repeat: false });

  // 推进 600 帧（约 10 秒游戏时间，覆盖首波出兵与 AI 交战）
  let now = 1000;
  for (let i = 0; i < 600; i++) {
    const frame = rafQueue.shift();
    assert.ok(frame, `第 ${i} 帧循环不应中断`);
    now += 16.7;
    frame(now);
  }

  // HUD 可观察状态被正确更新
  assert.match(byId('game-timer').textContent, /^\d{2}:\d{2}$/, '计时器应更新');
  assert.ok(byId('hero-meta').innerHTML.includes('Lv.'), '英雄信息应更新');
  assert.ok(byId('hp-bar').firstElementChild.style.width.endsWith('%'), '血条应更新');

  // 松开按键 + 触发回城按钮，再跑 100 帧确认依旧稳定
  dispatchWindow('keyup', { code: 'KeyD' });
  dispatchWindow('keyup', { code: 'Space' });
  byId('btn-recall').dispatch('mousedown');
  for (let i = 0; i < 100; i++) {
    const frame = rafQueue.shift();
    if (!frame) break; // 对局提前结束也算正常
    now += 16.7;
    frame(now);
  }
});
