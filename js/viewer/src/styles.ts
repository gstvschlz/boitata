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
  position: absolute; top: 10px; right: 10px; width: 288px; max-height: calc(100% - 20px);
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
.btv-settings { margin: 4px 6px 10px 14px; display: grid; grid-template-columns: 58px minmax(0, 1fr); gap: 6px 8px;
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
.btv-shot { display: flex; align-items: center; gap: 6px; padding: 8px 12px; border-bottom: 1px solid var(--btv-border); }
.btv-shot[hidden] { display: none; }
.btv-seg { display: inline-flex; border: 1px solid var(--btv-border); border-radius: 4px; overflow: hidden; }
.btv-seg button { flex: 1; border: 0; padding: 3px 6px; white-space: nowrap; font: inherit; background: none; color: var(--btv-muted); cursor: pointer; }
.btv-seg button.btv-on { background: var(--btv-accent); color: var(--btv-bg); }
.btv-seg button:not(.btv-on):hover { background: var(--btv-raised); color: var(--btv-text); }
.btv-seg button + button { border-left: 1px solid var(--btv-border); }
.btv-settings .btv-seg { display: flex; }
.btv-root button:focus-visible { outline: 2px solid var(--btv-accent); outline-offset: 1px; }
.btv-view { margin: 4px 12px 6px; }
.btv-tools { display: flex; gap: 4px; }
.btv-icon.btv-tool { width: 30px; height: 26px; border: 1px solid var(--btv-border); }
.btv-icon.btv-tool[aria-pressed=true] { background: var(--btv-accent); border-color: var(--btv-accent); color: var(--btv-bg); }
.btv-icon.btv-tool[aria-pressed=true]:hover { filter: brightness(1.08); }
.btv-shot label { display: flex; align-items: center; gap: 4px; color: var(--btv-muted); flex: 1; }
.btv-shot label input { accent-color: var(--btv-accent); margin: 0; }
.btv-mark { flex: none; width: 12px; height: 12px; color: var(--btv-accent); }
.btv-readout { margin: -2px 6px 2px 32px; color: var(--btv-muted); font-size: 11px; font-variant-numeric: tabular-nums; }
.btv-readout[hidden] { display: none; }
.btv-filter { grid-column: 1 / -1; display: flex; flex-direction: column; gap: 6px; padding-top: 6px;
  border-top: 1px solid var(--btv-border); }
.btv-filter-head { display: flex; align-items: center; justify-content: space-between; }
.btv-icon.btv-add { width: auto; height: 22px; gap: 3px; padding: 0 7px 0 4px; font: inherit; border: 1px solid var(--btv-border); }
.btv-icon.btv-add svg { width: 14px; height: 14px; }
.btv-note { color: var(--btv-muted); font-size: 11px; }
.btv-cond { border: 1px solid var(--btv-border); border-radius: 5px; padding: 4px 8px 8px; background: var(--btv-panel); }
.btv-cond-head { display: flex; align-items: center; gap: 5px; min-height: 24px; }
.btv-cond-head > svg { flex: none; width: 13px; height: 13px; color: var(--btv-accent); }
.btv-cond-head .btv-name { font-weight: 600; }
.btv-cond-head .btv-icon { width: 22px; height: 22px; }
.btv-hist { height: 34px; margin-top: 2px; }
.btv-hist svg { display: block; width: 100%; height: 100%; }
.btv-bars-all { fill: var(--btv-muted); opacity: .35; }
.btv-bars-on { fill: var(--btv-accent); }
.btv-track { position: relative; height: 16px; margin: 0 6px 6px; cursor: pointer; touch-action: none; }
.btv-track::before { content: ""; position: absolute; left: -6px; right: -6px; top: 7px; height: 2px; border-radius: 1px;
  background: var(--btv-border); }
.btv-fill { position: absolute; top: 7px; height: 2px; background: var(--btv-accent); }
.btv-handle { position: absolute; top: 2px; width: 12px; height: 12px; margin-left: -6px; border-radius: 50%;
  background: var(--btv-bg); border: 2px solid var(--btv-accent); }
.btv-handle:focus-visible { outline: 2px solid var(--btv-accent); outline-offset: 1px; }
.btv-quick { display: flex; gap: 4px; margin: 2px 0 4px; }
.btv-quick button { border: 1px solid var(--btv-border); border-radius: 4px; padding: 1px 8px; font: inherit;
  background: none; color: var(--btv-muted); cursor: pointer; }
.btv-quick button:hover { background: var(--btv-raised); color: var(--btv-text); }
.btv-list { max-height: 150px; overflow-y: auto; }
.btv-list label { display: flex; align-items: center; gap: 6px; min-height: 22px; color: var(--btv-text); cursor: pointer; }
.btv-list input { margin: 0; accent-color: var(--btv-accent); }
.btv-list .btv-count { color: var(--btv-muted); font-variant-numeric: tabular-nums; }
.btv-filter > select { width: 100%; height: 24px; padding: 0 6px; font: inherit; color: var(--btv-text);
  background: var(--btv-raised); border: 1px solid var(--btv-border); border-radius: 4px; }
.btv-bars { position: absolute; left: 10px; bottom: 10px; display: flex; flex-direction: column-reverse; gap: 8px;
  pointer-events: none; }
.btv-bar { pointer-events: auto; cursor: grab; touch-action: none; }
.btv-bar.btv-moved { position: absolute; }
.btv-bar svg { display: block; }
`;
