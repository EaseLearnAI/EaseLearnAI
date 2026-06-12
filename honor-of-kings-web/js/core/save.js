// 玩家档案存取与三端同步。
// 档案保存在 localStorage（每台设备本地持久化），并支持导出 / 导入同步码：
// 把同步码复制到另一台设备（手机 / 平板 / 电脑）导入，即可同步战绩与设置。
const SAVE_KEY = 'honor-arena-profile-v1';

export function defaultProfile() {
  return {
    v: 1,
    name: '召唤师',
    wins: 0,
    losses: 0,
    totalKills: 0,
    totalDeaths: 0,
    favoriteHero: 'jianhun',
    updatedAt: 0
  };
}

// storage 参数允许在 Node 测试中注入内存实现
export function loadProfile(storage) {
  try {
    const raw = storage.getItem(SAVE_KEY);
    if (!raw) return defaultProfile();
    const data = JSON.parse(raw);
    if (!data || data.v !== 1) return defaultProfile();
    return { ...defaultProfile(), ...data };
  } catch {
    return defaultProfile();
  }
}

export function saveProfile(storage, profile) {
  profile.updatedAt = Date.now();
  storage.setItem(SAVE_KEY, JSON.stringify(profile));
}

// 对局结束后记录战绩
export function recordMatch(profile, { won, kills, deaths, heroId }) {
  if (won) profile.wins += 1; else profile.losses += 1;
  profile.totalKills += kills;
  profile.totalDeaths += deaths;
  profile.favoriteHero = heroId;
  return profile;
}

// ---- 同步码（base64 编码的 JSON，跨设备粘贴即可同步） ----
function toB64(str) {
  if (typeof btoa === 'function') return btoa(unescape(encodeURIComponent(str)));
  return Buffer.from(str, 'utf8').toString('base64');
}

function fromB64(b64) {
  if (typeof atob === 'function') return decodeURIComponent(escape(atob(b64)));
  return Buffer.from(b64, 'base64').toString('utf8');
}

export function exportCode(profile) {
  return toB64(JSON.stringify(profile));
}

// 非法同步码返回 null
export function importCode(code) {
  try {
    const data = JSON.parse(fromB64(code.trim()));
    if (!data || data.v !== 1 || typeof data.wins !== 'number') return null;
    return { ...defaultProfile(), ...data };
  } catch {
    return null;
  }
}

// 冲突合并：取两份档案中“更新时间更晚”者，战绩字段取较大值，保证不丢进度
export function mergeProfiles(a, b) {
  const newer = (a.updatedAt || 0) >= (b.updatedAt || 0) ? a : b;
  return {
    ...newer,
    wins: Math.max(a.wins, b.wins),
    losses: Math.max(a.losses, b.losses),
    totalKills: Math.max(a.totalKills, b.totalKills),
    totalDeaths: Math.max(a.totalDeaths, b.totalDeaths)
  };
}
