document.addEventListener("DOMContentLoaded", () => {
  setupTabs(".install-tab", "data-install", ".install-panel", "data-install-panel");
  setupTabs(".code-tab", "data-lang", ".code-panel", "data-lang-panel");
  setupTabs("[data-decay]", "data-decay", "[data-decay-panel]", "data-decay-panel");
  setupCopyButtons();
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
