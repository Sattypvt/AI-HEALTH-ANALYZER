// Age slider live display
document.addEventListener("DOMContentLoaded", () => {
  const ageSlider = document.getElementById("age");
  const ageVal    = document.getElementById("age-val");
  if (ageSlider && ageVal) {
    ageSlider.addEventListener("input", () => {
      ageVal.textContent = ageSlider.value;
    });
  }

  // Animate progress bars on result page
  document.querySelectorAll(".progress-bar-fill").forEach(bar => {
    const target = bar.style.width;
    bar.style.width = "0%";
    setTimeout(() => { bar.style.width = target; }, 200);
  });

  // Animate health score counter
  const scoreEl = document.querySelector(".score-number");
  if (scoreEl) {
    const target = parseInt(scoreEl.textContent);
    let current  = 0;
    const step   = Math.ceil(target / 40);
    const timer  = setInterval(() => {
      current = Math.min(current + step, target);
      scoreEl.textContent = current;
      if (current >= target) clearInterval(timer);
    }, 25);
  }
});