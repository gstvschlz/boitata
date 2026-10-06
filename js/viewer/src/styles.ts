export const CSS = /* css */ `
:host { all: initial; display: block; }
.btv-root {
  position: relative; width: 100%; overflow: hidden; outline: none;
  background: var(--btv-bg); color: var(--btv-text);
  font: 12px/1.35 system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
  -webkit-font-smoothing: antialiased; user-select: none;
}
.btv-root canvas { display: block; }
.btv-root *, .btv-root *::before, .btv-root *::after { box-sizing: border-box; }
.btv-labels { position: absolute; inset: 0; pointer-events: none; overflow: hidden; }
.btv-label { position: absolute; left: 0; top: 0; white-space: nowrap; color: var(--btv-muted); font-size: 11px;
  font-variant-numeric: tabular-nums; }
.btv-label.btv-title { color: var(--btv-text); font-weight: 600; }
.btv-gizmo { position: absolute; right: 8px; bottom: 8px; cursor: pointer; }
.btv-gizmo .btv-gizmo-axis:hover circle { stroke: var(--btv-text); stroke-width: 1.5; }

.btv-panel {
  position: absolute; top: 10px; right: 10px; width: 264px; max-height: calc(100% - 20px);
  display: flex; flex-direction: column; background: var(--btv-panel); border: 1px solid var(--btv-border);
  border-radius: 6px; box-shadow: 0 4px 16px rgba(0, 0, 0, .10); overflow: hidden;
}
.btv-panel[hidden], .btv-open[hidden] { display: none; }
.btv-head { display: flex; align-items: center; gap: 2px; padding: 6px 6px 6px 12px; border-bottom: 1px solid var(--btv-border); }
.btv-head .btv-brand { flex: 1; font-weight: 600; letter-spacing: .02em; }
.btv-body { overflow-y: auto; padding: 4px 0 8px; }
.btv-icon {
  display: inline-flex; align-items: center; justify-content: center; width: 26px; height: 26px; padding: 0;
  border: 0; border-radius: 4px; background: transparent; color: var(--btv-muted); cursor: pointer; flex: none;
}
.btv-icon:hover { background: var(--btv-raised); color: var(--btv-text); }
.btv-icon.btv-on { color: var(--btv-accent); }
.btv-icon svg { width: 16px; height: 16px; }
.btv-open { position: absolute; top: 10px; right: 10px; width: 34px; height: 34px; background: var(--btv-panel);
  border: 1px solid var(--btv-border); box-shadow: 0 4px 16px rgba(0, 0, 0, .10); }
.btv-section { display: flex; align-items: center; gap: 4px; width: 100%; padding: 8px 12px 4px; border: 0;
  background: none; color: var(--btv-muted); font: inherit; font-size: 10.5px; font-weight: 600;
  letter-spacing: .08em; text-transform: uppercase; cursor: pointer; text-align: left; }
.btv-section svg { width: 13px; height: 13px; }
.btv-section .btv-count { margin-left: auto; font-weight: 500; letter-spacing: 0; }
.btv-group { padding: 0 6px; }
.btv-row { display: flex; align-items: center; gap: 4px; padding: 1px 4px 1px 2px; border-radius: 4px; min-height: 28px; }
.btv-row:hover { background: var(--btv-raised); }
.btv-row.btv-off .btv-name, .btv-row.btv-off .btv-key { opacity: .45; }
.btv-name { flex: 1; min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.btv-key { flex: none; width: 34px; height: 10px; border-radius: 2px; border: 1px solid var(--btv-border); }
.btv-key.btv-swatch { width: 12px; height: 12px; border-radius: 3px; }
.btv-settings { margin: 2px 4px 8px 32px; display: grid; grid-template-columns: 62px 1fr; gap: 6px 8px;
  align-items: center; }
.btv-settings label { color: var(--btv-muted); }
.btv-settings select, .btv-settings input[type=number] {
  width: 100%; min-width: 0; height: 24px; padding: 0 6px; font: inherit; color: var(--btv-text);
  background: var(--btv-raised); border: 1px solid var(--btv-border); border-radius: 4px; }
.btv-settings input[type=color] { width: 40px; height: 22px; padding: 0; border: 1px solid var(--btv-border);
  border-radius: 4px; background: none; }
.btv-range { display: flex; align-items: center; gap: 4px; }
.btv-range .btv-icon { width: 22px; height: 22px; }
.btv-root input[type=range] { width: 100%; accent-color: var(--btv-accent); }
.btv-toggle { display: flex; align-items: center; gap: 8px; padding: 4px 12px; cursor: pointer; }
.btv-toggle input { margin: 0; accent-color: var(--btv-accent); }
.btv-shot { display: flex; align-items: center; gap: 6px; padding: 8px 12px; border-bottom: 1px solid var(--btv-border); }
.btv-shot[hidden] { display: none; }
.btv-seg { display: inline-flex; border: 1px solid var(--btv-border); border-radius: 4px; overflow: hidden; }
.btv-seg button { border: 0; padding: 3px 7px; font: inherit; background: none; color: var(--btv-muted); cursor: pointer; }
.btv-seg button.btv-on { background: var(--btv-accent); color: var(--btv-bg); }
.btv-shot label { display: flex; align-items: center; gap: 4px; color: var(--btv-muted); flex: 1; }
.btv-shot label input { accent-color: var(--btv-accent); margin: 0; }
.btv-bars { position: absolute; left: 10px; bottom: 10px; display: flex; flex-direction: column-reverse; gap: 8px;
  pointer-events: none; }
.btv-bar { pointer-events: auto; cursor: grab; touch-action: none; }
.btv-bar.btv-moved { position: absolute; }
.btv-bar svg { display: block; }
`;
