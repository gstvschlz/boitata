// Starts a figure's animations when it scrolls into view, and adds a replay button to animated figures.
(() => {
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
  if (window.document$) window.document$.subscribe(watch);
  else document.addEventListener("DOMContentLoaded", watch);
})();
