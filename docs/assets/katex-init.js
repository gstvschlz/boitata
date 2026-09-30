const renderMath = () => {
  for (const el of document.querySelectorAll(".arithmatex")) {
    renderMathInElement(el, {
      delimiters: [{ left: "\\(", right: "\\)", display: false }, { left: "\\[", right: "\\]", display: true }],
    });
  }
};
if (window.document$) window.document$.subscribe(renderMath);
else document.addEventListener("DOMContentLoaded", renderMath);
