import {
  Camera,
  ChevronDown,
  ChevronRight,
  createElement,
  Download,
  Ellipsis,
  Eye,
  EyeOff,
  Layers,
  Moon,
  PanelRightClose,
  RotateCcw,
  Scan,
  Sun,
  type IconNode,
} from "lucide";

export const ICONS = {
  camera: Camera,
  chevronDown: ChevronDown,
  chevronRight: ChevronRight,
  download: Download,
  ellipsis: Ellipsis,
  eye: Eye,
  eyeOff: EyeOff,
  layers: Layers,
  moon: Moon,
  hide: PanelRightClose,
  reset: RotateCcw,
  fit: Scan,
  sun: Sun,
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
