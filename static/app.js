const fileInput = document.getElementById("fileInput");
const uploadBtn = document.getElementById("uploadBtn");
const uploadStatus = document.getElementById("uploadStatus");
const chatWindow = document.getElementById("chatWindow");
const chatForm = document.getElementById("chatForm");
const questionInput = document.getElementById("questionInput");
const askBtn = document.getElementById("askBtn");

let polling = null;

uploadBtn.addEventListener("click", async () => {
  const file = fileInput.files[0];
  if (!file) {
    uploadStatus.textContent = "Choose a PDF first.";
    return;
  }

  const formData = new FormData();
  formData.append("file", file);

  uploadBtn.disabled = true;
  questionInput.disabled = true;
  askBtn.disabled = true;
  uploadStatus.textContent = "Uploading...";
  chatWindow.innerHTML = "";

  // Poll /status for progress while the (blocking) /upload request runs.
  polling = setInterval(async () => {
    const res = await fetch("/status");
    const data = await res.json();
    if (data.status === "processing") {
      uploadStatus.textContent = data.message;
    }
  }, 800);

  try {
    const res = await fetch("/upload", { method: "POST", body: formData });
    const data = await res.json();

    clearInterval(polling);

    if (!res.ok) {
      uploadStatus.textContent = "Error: " + (data.error || "upload failed");
    } else {
      uploadStatus.textContent = `Indexed: ${data.file_name}\n${data.num_documents} documents indexed.`;
      questionInput.disabled = false;
      askBtn.disabled = false;
    }
  } catch (err) {
    clearInterval(polling);
    uploadStatus.textContent = "Error: " + err.message;
  } finally {
    uploadBtn.disabled = false;
  }
});

chatForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  const question = questionInput.value.trim();
  if (!question) return;

  appendMessage("user", question);
  questionInput.value = "";
  askBtn.disabled = true;

  const thinkingEl = appendMessage("assistant", "Retrieving and reasoning...");

  try {
    const res = await fetch("/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    });
    const data = await res.json();

    if (!res.ok) {
      thinkingEl.querySelector(".bubble").textContent = "Error: " + (data.error || "request failed");
    } else {
      thinkingEl.querySelector(".bubble").textContent = data.answer;
      renderSources(thinkingEl, data.sources);
    }
  } catch (err) {
    thinkingEl.querySelector(".bubble").textContent = "Error: " + err.message;
  } finally {
    askBtn.disabled = false;
  }
});

function appendMessage(role, text) {
  const msg = document.createElement("div");
  msg.className = "msg " + role;
  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.textContent = text;
  msg.appendChild(bubble);
  chatWindow.appendChild(msg);
  chatWindow.scrollTop = chatWindow.scrollHeight;
  return msg;
}

function renderSources(msgEl, sources) {
  if (!sources || sources.length === 0) return;
  const wrap = document.createElement("div");
  wrap.className = "sources";

  const summary = document.createElement("div");
  summary.textContent = `📚 Sources (${sources.length})`;
  wrap.appendChild(summary);

  sources.forEach((s, i) => {
    const details = document.createElement("details");
    const sum = document.createElement("summary");
    sum.textContent = `Source ${i + 1} — ${s.type} · section: ${s.section} · page: ${s.page}`;
    details.appendChild(sum);

    const content = document.createElement("div");
    content.textContent = s.content;
    details.appendChild(content);

    if (s.image_base64) {
      const img = document.createElement("img");
      img.src = "data:image/png;base64," + s.image_base64;
      details.appendChild(img);
    }

    wrap.appendChild(details);
  });

  msgEl.appendChild(wrap);
  chatWindow.scrollTop = chatWindow.scrollHeight;
}
