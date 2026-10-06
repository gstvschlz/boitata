import {
  Axis3d,
  Box,
  Camera,
  ChevronDown,
  ChevronRight,
  createElement,
  Download,
  Ellipsis,
  Eraser,
  Eye,
  EyeOff,
  FoldHorizontal,
  Focus,
  Funnel,
  Grid3x3,
  Keyboard,
  Layers,
  Maximize,
  Minimize,
  Moon,
  PencilRuler,
  Plus,
  PanelRightClose,
  RotateCcw,
  Ruler,
  Sun,
  UnfoldHorizontal,
  X,
  type IconNode,
} from "lucide";

export const ICONS = {
  axes: Axis3d,
  box: Box,
  camera: Camera,
  chevronDown: ChevronDown,
  chevronRight: ChevronRight,
  download: Download,
  draw: PencilRuler,
  ellipsis: Ellipsis,
  clear: Eraser,
  fold: FoldHorizontal,
  unfold: UnfoldHorizontal,
  eye: Eye,
  eyeOff: EyeOff,
  filter: Funnel,
  grid: Grid3x3,
  layers: Layers,
  moon: Moon,
  plus: Plus,
  hide: PanelRightClose,
  reset: RotateCcw,
  ruler: Ruler,
  fit: Focus,
  keyboard: Keyboard,
  maximize: Maximize,
  minimize: Minimize,
  sun: Sun,
  x: X,
} satisfies Record<string, IconNode>;

export type IconName = keyof typeof ICONS;

export function icon(name: IconName): SVGElement {
  return createElement(ICONS[name], { "aria-hidden": "true" });
}

export function iconButton(name: IconName, title: string, onClick: (e: MouseEvent) => void): HTMLButtonElement {
  const button = document.createElement("button");
  button.type = "button";
  button.className = "btv-icon";
  button.title = title;
  button.setAttribute("aria-label", title);
  button.appendChild(icon(name));
  button.addEventListener("click", (e) => {
    e.stopPropagation();
    onClick(e);
  });
  return button;
}

export function setIcon(button: HTMLElement, name: IconName): void {
  button.replaceChildren(icon(name));
}
