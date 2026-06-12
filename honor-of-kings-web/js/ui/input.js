// 三端统一输入层：
//  - 电脑：WASD/方向键移动，空格或 J 普攻，K/L/U 技能 1/2/3，B 回城
//  - 手机/平板：左半屏虚拟摇杆移动，右下按钮普攻 / 技能 / 回城
// 输出统一写入 state，由 main.js 每帧拷贝到玩家英雄的 input。
export function createInput() {
  const state = {
    mx: 0, my: 0,          // 摇杆 / 键盘合成的移动方向
    attackHeld: false,     // 攻击键按住
    attackPulse: 0,        // 点按攻击的短暂保持（秒）
    skillQueue: [],        // 待释放技能序号
    recall: false
  };

  const keys = new Set();
  const KEY_SKILL = { KeyK: 0, KeyL: 1, KeyU: 2 };

  function syncKeyboardMove() {
    let x = 0, y = 0;
    if (keys.has('KeyA') || keys.has('ArrowLeft')) x -= 1;
    if (keys.has('KeyD') || keys.has('ArrowRight')) x += 1;
    if (keys.has('KeyW') || keys.has('ArrowUp')) y -= 1;
    if (keys.has('KeyS') || keys.has('ArrowDown')) y += 1;
    // 摇杆优先：仅当摇杆未激活时键盘生效
    if (!joyActive) { state.mx = x; state.my = y; }
  }

  window.addEventListener('keydown', e => {
    if (e.repeat) return;
    keys.add(e.code);
    if (e.code === 'Space' || e.code === 'KeyJ') { state.attackHeld = true; e.preventDefault(); }
    if (e.code in KEY_SKILL) state.skillQueue.push(KEY_SKILL[e.code]);
    if (e.code === 'KeyB') state.recall = true;
    syncKeyboardMove();
  });
  window.addEventListener('keyup', e => {
    keys.delete(e.code);
    if (e.code === 'Space' || e.code === 'KeyJ') state.attackHeld = false;
    syncKeyboardMove();
  });

  // ---------- 虚拟摇杆 ----------
  const zone = document.getElementById('joystick-zone');
  const baseEl = document.getElementById('joystick-base');
  const knobEl = document.getElementById('joystick-knob');
  let joyActive = false, joyId = -1, cx = 0, cy = 0;
  const RADIUS = 52;

  function joyStart(e) {
    const t = e.changedTouches ? e.changedTouches[0] : e;
    joyActive = true;
    joyId = t.identifier ?? -1;
    const r = baseEl.getBoundingClientRect();
    cx = r.left + r.width / 2; cy = r.top + r.height / 2;
    baseEl.classList.add('active');
    joyMove(e);
  }
  function joyMove(e) {
    if (!joyActive) return;
    const touches = e.changedTouches ? [...e.changedTouches] : [e];
    const t = touches.find(t => (t.identifier ?? -1) === joyId);
    if (!t) return;
    let dx = t.clientX - cx, dy = t.clientY - cy;
    const len = Math.hypot(dx, dy);
    if (len > RADIUS) { dx = dx / len * RADIUS; dy = dy / len * RADIUS; }
    knobEl.style.transform = `translate(calc(-50% + ${dx}px), calc(-50% + ${dy}px))`;
    const dead = 8;
    state.mx = len > dead ? dx / RADIUS : 0;
    state.my = len > dead ? dy / RADIUS : 0;
    e.preventDefault();
  }
  function joyEnd(e) {
    if (!joyActive) return;
    const touches = e.changedTouches ? [...e.changedTouches] : [e];
    if (!touches.some(t => (t.identifier ?? -1) === joyId)) return;
    joyActive = false; joyId = -1;
    state.mx = 0; state.my = 0;
    knobEl.style.transform = 'translate(-50%, -50%)';
    baseEl.classList.remove('active');
    syncKeyboardMove();
  }

  zone.addEventListener('touchstart', joyStart, { passive: false });
  zone.addEventListener('touchmove', joyMove, { passive: false });
  zone.addEventListener('touchend', joyEnd);
  zone.addEventListener('touchcancel', joyEnd);
  // 鼠标也可拖摇杆（桌面调试）
  zone.addEventListener('mousedown', joyStart);
  window.addEventListener('mousemove', joyMove);
  window.addEventListener('mouseup', joyEnd);

  // ---------- 按钮 ----------
  const bindPress = (el, down, up) => {
    el.addEventListener('touchstart', e => { e.preventDefault(); down(); }, { passive: false });
    el.addEventListener('touchend', e => { e.preventDefault(); up && up(); }, { passive: false });
    el.addEventListener('mousedown', down);
    el.addEventListener('mouseup', () => up && up());
    el.addEventListener('mouseleave', () => up && up());
  };

  bindPress(document.getElementById('btn-attack'),
    () => { state.attackHeld = true; state.attackPulse = 0.3; },
    () => { state.attackHeld = false; });
  for (const btn of document.querySelectorAll('.skill-btn[data-skill]')) {
    bindPress(btn, () => state.skillQueue.push(Number(btn.dataset.skill)));
  }
  bindPress(document.getElementById('btn-recall'), () => { state.recall = true; });

  return state;
}

// 把输入状态写入玩家英雄（每帧调用一次）
export function applyInput(state, hero, dt) {
  hero.input.mx = state.mx;
  hero.input.my = state.my;
  state.attackPulse = Math.max(0, state.attackPulse - dt);
  hero.input.attack = state.attackHeld || state.attackPulse > 0;
  hero.input.skill = state.skillQueue.length ? state.skillQueue.shift() : -1;
  if (state.recall) { hero.input.recall = true; state.recall = false; }
}
