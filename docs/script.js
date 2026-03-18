document.addEventListener("DOMContentLoaded", () => {
  setupTabs(".install-tab", "data-install", ".install-panel", "data-install-panel");
  setupTabs(".code-tab", "data-lang", ".code-panel", "data-lang-panel");
  setupTabs("[data-decay]", "data-decay", "[data-decay-panel]", "data-decay-panel");
  setupTabs("[data-context]", "data-context", "[data-context-panel]", "data-context-panel");
  setupCopyButtons();
  setupThemeToggle();
  renderKatex();
});

function renderKatex() {
  if (typeof katex === "undefined") {
    setTimeout(renderKatex, 100);
    return;
  }
  document.querySelectorAll("[data-katex]").forEach((el) => {
    katex.render(el.getAttribute("data-katex"), el, {
      displayMode: true,
      throwOnError: false,
    });
  });
}

function setupTabs(tabSelector, tabAttr, panelSelector, panelAttr) {
  const tabs = document.querySelectorAll(tabSelector);
  const panels = document.querySelectorAll(panelSelector);

  tabs.forEach((tab) => {
    tab.addEventListener("click", () => {
      const value = tab.getAttribute(tabAttr);

      tabs.forEach((t) => t.classList.remove("active"));
      panels.forEach((p) => p.classList.remove("active"));

      tab.classList.add("active");
      const target = document.querySelector(`[${panelAttr}="${value}"]`);
      if (target) target.classList.add("active");
    });
  });
}

function setupThemeToggle() {
  const btn = document.querySelector(".theme-toggle");
  if (!btn) return;

  const saved = localStorage.getItem("mem7-theme");
  if (saved) {
    document.documentElement.setAttribute("data-theme", saved);
  } else if (window.matchMedia("(prefers-color-scheme: light)").matches) {
    document.documentElement.setAttribute("data-theme", "light");
  }

  btn.addEventListener("click", () => {
    const current = document.documentElement.getAttribute("data-theme");
    const next = current === "light" ? "dark" : "light";
    document.documentElement.setAttribute("data-theme", next);
    localStorage.setItem("mem7-theme", next);
  });

  window.matchMedia("(prefers-color-scheme: light)").addEventListener("change", (e) => {
    if (!localStorage.getItem("mem7-theme")) {
      document.documentElement.setAttribute("data-theme", e.matches ? "light" : "dark");
    }
  });
}

function setupCopyButtons() {
  document.querySelectorAll(".copy-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      let text = btn.getAttribute("data-copy");

      if (!text) {
        const codeId = btn.getAttribute("data-copy-code");
        if (codeId) {
          const codeEl = document.getElementById(`code-${codeId}`);
          if (codeEl) text = codeEl.textContent;
        }
      }

      if (!text) return;

      navigator.clipboard.writeText(text).then(() => {
        btn.classList.add("copied");
        const svg = btn.querySelector("svg");
        const original = svg.innerHTML;
        svg.innerHTML =
          '<polyline points="20 6 9 17 4 12" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>';
        setTimeout(() => {
          btn.classList.remove("copied");
          svg.innerHTML = original;
        }, 1500);
      });
    });
  });
}
