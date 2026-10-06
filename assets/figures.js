// Starts a figure's animations when it scrolls into view, and adds a replay button to animated figures.
// A 3D scene poster swaps itself for the live scene's page when clicked, so a page with several stays light.
(() => {
  const scenes = () => {
    for (const poster of document.querySelectorAll("button.bt-scene[data-scene]")) {
      poster.addEventListener("click", () => {
        const frame = document.createElement("iframe");
        frame.src = poster.dataset.scene;
        frame.title = poster.querySelector("img")?.alt || "3D scene";
        frame.allow = "fullscreen";
        const box = document.createElement("div");
        box.className = "bt-scene";
        box.append(frame);
        poster.replaceWith(box);
      }, { once: true });
    }
  };
  if (window.document$) window.document$.subscribe(scenes);
  else document.addEventListener("DOMContentLoaded", scenes);
})();
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
