// One dropdown interaction shared by the reader, graph, API docs and SearchSpider.
document.addEventListener("click", (event) => {
  if (!(event.target instanceof Element)) return;
  document.querySelectorAll(".bnr-navigation details[open]").forEach((menu) => {
    if (!menu.contains(event.target) || event.target.closest("a")) menu.open = false;
  });
});
document.addEventListener("keydown", (event) => {
  if (event.key !== "Escape") return;
  document.querySelectorAll(".bnr-navigation details[open]").forEach((menu) => {
    menu.open = false;
    menu.querySelector("summary").focus();
  });
});
