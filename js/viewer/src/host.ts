/** Whether the page hosting the viewer is dark: JupyterLab, VS Code, the docs site, else the OS preference. */
export function hostIsDark(): boolean {
  for (const doc of documents()) {
    const body = doc.body;
    if (!body) continue;
    const jupyter = body.getAttribute("data-jp-theme-light");
    if (jupyter) return jupyter === "false";
    if (body.classList.contains("vscode-dark") || body.classList.contains("vscode-high-contrast")) return true;
    if (body.classList.contains("vscode-light")) return false;
    const docs = body.getAttribute("data-md-color-scheme");
    if (docs) return docs === "slate";
    const colab = doc.documentElement.getAttribute("theme");
    if (colab) return colab === "dark";
  }
  return typeof matchMedia === "function" && matchMedia("(prefers-color-scheme: dark)").matches;
}

/** Calls `change` when the host's theme may have switched; returns the unsubscribe. */
export function watchHost(change: () => void): () => void {
  const observers = documents().map((doc) => {
    const observer = new MutationObserver(change);
    const options = { attributes: true, attributeFilter: ["class", "data-jp-theme-light", "data-md-color-scheme", "theme"] };
    observer.observe(doc.documentElement, options);
    if (doc.body) observer.observe(doc.body, options);
    return observer;
  });
  const media = typeof matchMedia === "function" ? matchMedia("(prefers-color-scheme: dark)") : null;
  media?.addEventListener("change", change);
  return () => {
    observers.forEach((o) => o.disconnect());
    media?.removeEventListener("change", change);
  };
}

/** This document and, when same-origin, the page embedding it (the docs site embeds scenes in iframes). */
function documents(): Document[] {
  const out = [document];
  try {
    if (window.parent !== window && window.parent.document) out.push(window.parent.document);
  } catch {
    // cross-origin parent
  }
  return out;
}
