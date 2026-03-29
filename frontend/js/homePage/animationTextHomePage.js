export function initAnimationText(){
  document.querySelectorAll('[data-typewriter]').forEach(el => {
    const text = el.textContent.trim();
    el.textContent = '';

    function type(i = 0) {
      if (i < text.length) {
        el.textContent += text[i];
        setTimeout(() => type(i + 1), 38 + Math.random() * 25);
      } else {
        setTimeout(() => erase(), 2000);
      }
    }

    function erase(i) {
      i = i ?? text.length;
      if (i > 0) {
        el.textContent = text.slice(0, i - 1);
        setTimeout(() => erase(i - 1), 20);
      } else {
        setTimeout(() => type(), 500);
      }
    }

    type();
  });
}