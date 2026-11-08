// Picks the colour theme before the first paint. A manual choice is remembered per browser;
// without one, the theme follows the system setting.
(function () {
  var root = document.documentElement;
  var media = window.matchMedia ? window.matchMedia("(prefers-color-scheme: dark)") : null;

  function stored() {
    try {
      var value = window.localStorage.getItem("theme");
      return value === "light" || value === "dark" ? value : null;
    } catch (error) {
      return null;
    }
  }

  root.dataset.theme = stored() ? stored() : media && media.matches ? "dark" : "light";

  function sync() {
    var dark = root.dataset.theme === "dark";
    var meta = document.getElementById("theme-color");
    if (meta) meta.setAttribute("content", dark ? "#0b0c0f" : "#f5f5f7");
    var button = document.getElementById("theme-toggle");
    if (!button) return;
    var label = dark ? "Açık temaya geç" : "Koyu temaya geç";
    button.setAttribute("aria-label", label);
    button.setAttribute("title", dark ? "Açık tema" : "Koyu tema");
    var use = button.querySelector("use");
    if (use) use.setAttribute("href", dark ? "#i-sun" : "#i-moon");
  }

  function toggle() {
    var dark = root.dataset.theme !== "dark";
    root.classList.add("theme-switching");
    root.dataset.theme = dark ? "dark" : "light";
    try {
      window.localStorage.setItem("theme", root.dataset.theme);
    } catch (error) {
      // Private windows refuse storage: the choice then lasts for this visit only.
    }
    sync();
    window.setTimeout(function () {
      root.classList.remove("theme-switching");
    }, 320);
  }

  document.addEventListener("DOMContentLoaded", function () {
    sync();
    var button = document.getElementById("theme-toggle");
    if (button) button.addEventListener("click", toggle);
  });

  if (media && media.addEventListener) {
    media.addEventListener("change", function () {
      if (stored()) return;
      root.dataset.theme = media.matches ? "dark" : "light";
      sync();
    });
  }
})();
