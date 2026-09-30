// Prototype only: a theme picker in the header, remembered per browser. Removed once a theme is chosen.
const BT_THEMES = [
  ["baseline", "Refined baseline"],
  ["editorial", "Editorial serif"],
  ["swiss", "Swiss grid"],
  ["tufte", "Tufte"],
  ["textbook", "Textbook"],
  ["notebook", "Field notebook"],
  ["blueprint", "Blueprint"],
  ["terminal", "Lab terminal"],
  ["magazine", "Magazine"],
  ["paper", "Minimal paper"],
  ["strata", "Strata"],
  ["contrast", "High contrast"],
];

(() => {
  let saved = "baseline";
  try { saved = localStorage.getItem("bt-theme") || saved; } catch {}
  document.documentElement.dataset.btTheme = saved;

  const mount = () => {
    const header = document.querySelector(".md-header__inner");
    if (!header || header.querySelector(".bt-theme-picker")) return;
    const select = document.createElement("select");
    select.className = "bt-theme-picker";
    select.title = "Documentation theme (prototype)";
    for (const [value, label] of BT_THEMES) select.add(new Option(label, value, false, value === saved));
    select.addEventListener("change", () => {
      document.documentElement.dataset.btTheme = select.value;
      try { localStorage.setItem("bt-theme", select.value); } catch {}
    });
    header.insertBefore(select, header.querySelector(".md-header__source"));
  };

  const watch = () => {
    const seen = new IntersectionObserver((entries) => {
      for (const e of entries) if (e.isIntersecting) { e.target.classList.add("in-view"); seen.unobserve(e.target); }
    }, { threshold: 0.35 });
    for (const fig of document.querySelectorAll("figure.bt-figure")) {
      seen.observe(fig);
      if (fig.querySelector("[class*='a-']") && !fig.querySelector(".bt-replay")) {
        const replay = document.createElement("button");
        replay.className = "bt-replay";
        replay.textContent = "↻ replay";
        replay.addEventListener("click", () => { fig.classList.remove("in-view"); void fig.offsetWidth; fig.classList.add("in-view"); });
        fig.prepend(replay);
      }
    }
  };

  const ready = () => { mount(); watch(); };
  if (window.document$) window.document$.subscribe(ready);
  else document.addEventListener("DOMContentLoaded", ready);
})();
