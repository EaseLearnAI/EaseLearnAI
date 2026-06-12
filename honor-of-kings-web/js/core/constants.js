// 全局常量（纯数据，浏览器 / Node 测试通用）
export const WORLD = { w: 2400, h: 1400 };
export const LANE_Y = 700;

export const TEAM = { BLUE: 0, RED: 1 };
export const TEAM_NAME = ['蓝方', '红方'];
export const TEAM_COLOR = ['#3b82f6', '#ef4444'];

export const FIRST_WAVE_DELAY = 5;       // 开局首波兵延迟（秒）
export const WAVE_INTERVAL = 25;         // 兵线间隔（秒）
export const MELEE_PER_WAVE = 3;
export const RANGED_PER_WAVE = 1;

export const MAX_LEVEL = 15;
export const RESPAWN_BASE = 5;           // 复活基础时间 + 等级 * 2
export const RECALL_TIME = 3;            // 回城读条（秒）

export const GOLD_PASSIVE = 5;           // 每秒被动金币
export const GOLD_MINION = 60;
export const GOLD_HERO_KILL = 300;
export const GOLD_TOWER = 150;
export const EXP_MINION = 60;
export const EXP_HERO_KILL = 200;
export const EXP_SHARE_RANGE = 700;      // 经验共享半径

export const EQUIP_GOLD_PER_TIER = 800;  // 每 800 金币自动升一阶装备
export const EQUIP_MAX_TIER = 6;
export const EQUIP_ATK_PER_TIER = 14;
export const EQUIP_DEF_PER_TIER = 9;
export const EQUIP_HP_PER_TIER = 160;

// 防御塔 / 水晶
export const TOWER_HP = 3200;
export const TOWER_ATK = 190;
export const TOWER_DEF = 110;
export const TOWER_RANGE = 340;
export const CRYSTAL_HP = 5200;
export const CRYSTAL_DEF = 90;

// 小兵
export const MINION_STATS = {
  melee: { hp: 480, atk: 42, def: 20, range: 50, speed: 95, atkInterval: 1.1 },
  ranged: { hp: 330, atk: 58, def: 10, range: 270, speed: 95, atkInterval: 1.4 }
};
export const MINION_HP_GROWTH = 22;      // 每波小兵成长
export const MINION_AGGRO = 320;

// 出生点与建筑布局（蓝方，红方按地图宽度镜像）
export const LAYOUT = {
  crystalX: 170,
  towerXs: [620, 1050],
  spawnX: 260
};
