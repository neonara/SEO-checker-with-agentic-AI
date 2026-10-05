// Before first paint: a saved light/dark choice wins over the system setting.
try {
  var saved = localStorage.getItem("theme");
  if (saved === "dark" || saved === "light")
    document.documentElement.dataset.theme = saved;
} catch (e) {
  /* storage unavailable: follow the system */
}
