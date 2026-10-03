// When a page is opened at a #target (e.g. a cross-reference from another page),
// the browser jumps there before MathJax has typeset the math above it, which
// then changes the page height and leaves the target off screen. Re-scroll
// once MathJax is done.
function scrollToHash() {
  if (!window.location.hash) return;
  const target = document.getElementById(
    decodeURIComponent(window.location.hash.slice(1))
  );
  if (target) target.scrollIntoView();
}

window.addEventListener("load", () => {
  if (window.MathJax && window.MathJax.startup && window.MathJax.startup.promise) {
    window.MathJax.startup.promise.then(scrollToHash);
  } else {
    scrollToHash();
  }
});
