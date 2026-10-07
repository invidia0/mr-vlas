// Interactive quizzes rendered by book/hooks/quiz.py.
// Single choice: clicking an option grades immediately.
// Multi choice: toggle options, then press "Check".
function setupQuiz(q) {
  if (q.dataset.ready) return;
  q.dataset.ready = "1";
  const opts = [...q.querySelectorAll(".quiz-opt")];
  const multi = q.dataset.type === "multi";

  const grade = () => {
    let allRight = true;
    for (const o of opts) {
      const correct = o.dataset.correct === "1";
      const chosen = o.classList.contains("chosen");
      o.classList.add("graded", correct ? "is-correct" : "is-wrong");
      if (correct !== chosen) allRight = false;
    }
    q.classList.add("answered", allRight ? "right" : "wrong");
    status.textContent = allRight ? "Correct." : "Not quite — read the explanations below each option.";
  };

  for (const o of opts) {
    o.tabIndex = 0;
    o.setAttribute("role", multi ? "checkbox" : "radio");
    const pick = () => {
      if (q.classList.contains("answered")) return;
      if (multi) {
        o.classList.toggle("chosen");
        o.setAttribute("aria-checked", o.classList.contains("chosen"));
      } else {
        o.classList.add("chosen");
        grade();
      }
    };
    o.addEventListener("click", pick);
    o.addEventListener("keydown", (e) => {
      if (e.key === "Enter" || e.key === " ") { e.preventDefault(); pick(); }
    });
  }

  const bar = document.createElement("div");
  bar.className = "quiz-bar";
  const status = document.createElement("span");
  status.className = "quiz-status";
  status.setAttribute("aria-live", "polite");
  if (multi) {
    const check = document.createElement("button");
    check.className = "md-button md-button--primary";
    check.textContent = "Check";
    check.onclick = () => { if (!q.classList.contains("answered")) grade(); };
    bar.append(check);
  }
  const reset = document.createElement("button");
  reset.className = "md-button";
  reset.textContent = "Reset";
  reset.onclick = () => {
    q.classList.remove("answered", "right", "wrong");
    for (const o of opts) o.classList.remove("chosen", "graded", "is-correct", "is-wrong");
    status.textContent = "";
  };
  bar.append(reset, status);
  q.querySelector(".quiz-explain").before(bar);
}

document$.subscribe(() => document.querySelectorAll(".quiz-q").forEach(setupQuiz));
