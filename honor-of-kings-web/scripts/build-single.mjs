// 把游戏打包成单文件 HTML（内联全部 CSS/JS，无模块依赖）。
// 产物 dist/index.html 可直接双击打开、部署到任意静态托管或发布为在线页面。
import { readFile, writeFile, mkdir } from 'node:fs/promises';

const ROOT = new URL('..', import.meta.url).pathname;
const read = p => readFile(ROOT + p, 'utf8');

// 依赖顺序：常量 → 工具 → 数值 → 英雄 → 核心 → AI → 存档 → 输入 → 渲染 → HUD → 入口
const MODULES = [
  'js/core/constants.js', 'js/core/utils.js', 'js/core/combat.js',
  'js/core/heroes.js', 'js/core/game.js', 'js/core/ai.js', 'js/core/save.js',
  'js/ui/input.js', 'js/ui/renderer.js', 'js/ui/hud.js', 'js/main.js'
];

function stripModuleSyntax(src) {
  return src
    .replace(/^import[\s\S]*?from\s+['"][^'"]+['"];\s*$/gm, '')
    .replace(/^export\s+/gm, '');
}

const js = [];
// 沙箱环境（如嵌入 iframe）访问 localStorage 可能抛异常，降级为内存存储
js.push(`const safeStorage = (() => {
  try {
    const s = window.localStorage;
    s.setItem('__t', '1'); s.removeItem('__t');
    return s;
  } catch {
    const m = new Map();
    return { getItem: k => (m.has(k) ? m.get(k) : null), setItem: (k, v) => m.set(k, String(v)) };
  }
})();`);

for (const path of MODULES) {
  let src = stripModuleSyntax(await read(path));
  if (path === 'js/main.js') src = src.replaceAll('localStorage', 'safeStorage');
  js.push(`// ===== ${path} =====\n${src.trim()}`);
}

const css = await read('css/style.css');
const html = await read('index.html');
const body = html
  .match(/<body>([\s\S]*)<\/body>/)[1]
  .replace(/<script type="module"[^>]*><\/script>/, '')
  .trim();

const out = `<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no, viewport-fit=cover">
<meta name="apple-mobile-web-app-capable" content="yes">
<title>王者竞技场 · 网页版 MOBA</title>
<style>
${css}
</style>
</head>
<body>
${body}
<script type="module">
${js.join('\n\n')}
</script>
</body>
</html>
`;

await mkdir(ROOT + 'dist', { recursive: true });
await writeFile(ROOT + 'dist/index.html', out);
console.log(`已生成 dist/index.html（${(out.length / 1024).toFixed(1)} KB）`);
