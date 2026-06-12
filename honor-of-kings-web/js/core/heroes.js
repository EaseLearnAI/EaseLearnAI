// 英雄定义（全部为原创角色，数值为基础值 + 每级成长）
// skill.type 决定 game.castSkill 中的效果分支
export const HERO_DEFS = [
  {
    id: 'jianhun',
    name: '剑魂',
    role: '战士',
    desc: '近战突进型战士，擅长切入收割',
    color: '#f59e0b',
    base: { hp: 1450, hpG: 185, mp: 420, mpG: 40, atk: 112, atkG: 15, def: 62, defG: 8, range: 90, speed: 205, atkInterval: 1.0 },
    skills: [
      { key: 'dash', name: '破阵突斩', cd: 6, mana: 60, range: 320, desc: '向前突进并斩击路径终点的敌人' },
      { key: 'spin', name: '剑气旋风', cd: 9, mana: 80, range: 170, desc: '以自身为中心旋转剑刃造成范围伤害' },
      { key: 'execute', name: '裂地终结', cd: 28, mana: 140, range: 230, desc: '大范围重斩并回复自身生命' }
    ]
  },
  {
    id: 'xinglan',
    name: '星澜',
    role: '法师',
    desc: '远程爆发法师，团战核心输出',
    color: '#a78bfa',
    base: { hp: 1080, hpG: 125, mp: 560, mpG: 60, atk: 132, atkG: 19, def: 42, defG: 5, range: 390, speed: 188, atkInterval: 1.05 },
    skills: [
      { key: 'bolt', name: '星辉弹', cd: 4.5, mana: 55, range: 480, desc: '射出星辉能量打击最近的敌人' },
      { key: 'field', name: '减速星域', cd: 11, mana: 90, range: 420, desc: '在敌人脚下展开星域，伤害并减速' },
      { key: 'meteor', name: '坠星术', cd: 30, mana: 160, range: 460, desc: '召唤巨大陨星轰炸目标区域' }
    ]
  },
  {
    id: 'fengyu',
    name: '风羽',
    role: '射手',
    desc: '远程物理射手，持续输出与拆塔强势',
    color: '#34d399',
    base: { hp: 1180, hpG: 135, mp: 460, mpG: 45, atk: 122, atkG: 17, def: 46, defG: 6, range: 430, speed: 198, atkInterval: 0.78 },
    skills: [
      { key: 'pierce', name: '穿云箭', cd: 5, mana: 50, range: 540, desc: '射出强力穿透箭，重创最近的敌人' },
      { key: 'haste', name: '风之疾行', cd: 12, mana: 70, range: 0, desc: '借助风力大幅提升移动速度' },
      { key: 'volley', name: '万箭齐发', cd: 26, mana: 150, range: 500, desc: '箭雨覆盖目标区域内所有敌人' }
    ]
  }
];

export function heroDefById(id) {
  return HERO_DEFS.find(h => h.id === id) || null;
}
