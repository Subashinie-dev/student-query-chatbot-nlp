// static/chat.js
const form = document.getElementById("chat-form");
const input = document.getElementById("chat-input");
const win = document.getElementById("chat-window");

function addMessage(text, who="bot") {
  const wrapper = document.createElement("div");
  wrapper.className = "msg " + (who === "user" ? "user" : "bot");

  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.innerHTML = text.replace(/\n/g, "<br>");

  wrapper.appendChild(bubble);
  win.appendChild(wrapper);
  win.scrollTop = win.scrollHeight;
}

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  const text = input.value.trim();
  if (!text) return;

  addMessage(text, "user");
  input.value = "";

  // bot placeholder
  addMessage("Thinking...", "bot");
  const botBubble = win.lastChild.querySelector(".bubble");

  try {
    const r = await fetch("http://127.0.0.1:5000/api/query", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({ text })
    });

    const j = await r.json();

    if (j.error) {
      botBubble.textContent = "Error: " + j.error;
      return;
    }

    // Show main answer
    botBubble.innerHTML = j.answer.replace(/\n/g, "<br>");

    // Add metadata (intents + multi-intent answers)
    const meta = document.createElement("div");
    meta.className = "small";
    meta.innerHTML = `
      <br><b>Detected intents:</b> 
      ${j.intents.map(i => `${i.intent} (${(i.conf*100).toFixed(1)}%)`).join(", ")}
    `;
    botBubble.parentElement.appendChild(meta);

    // Add additional multi answers (if any)
    if (j.multi && j.multi.length > 1) {
      j.multi.slice(1).forEach(m => {
        const extra = document.createElement("div");
        extra.className = "bubble small";
        extra.style.opacity = "0.8";
        extra.innerHTML = `➤ ${m.answer}`;
        botBubble.parentElement.appendChild(extra);
      });
    }

  } catch (err) {
    botBubble.textContent = "Network error.";
    console.error(err);
  }
});
