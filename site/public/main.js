(() => {
  const $ = (id) => document.getElementById(id);
  const reduce = matchMedia("(prefers-reduced-motion: reduce)").matches;
  const wait = (ms) => new Promise((r) => setTimeout(r, reduce ? 0 : ms));
  const html = (s) => s.replace(/[&<>"]/g, (c) => (
    { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]
  ));

  // Each memo has a matching reply written the way a model would see it:
  // {value} marks a detail the model only knows as a placeholder.
  const MEMOS = {
    "Support reply": {
      re: "Customer reply, draft needed",
      text: "Please draft a reply to Sarah Johnson at Acme Corp. She wrote in from sarah.j@acmecorp.com about a double charge on her card 4111 1111 1111 1111 and would like a call back on +44 20 7946 0958 before Friday.",
      reply: "Here's a draft:\n\nHi {Sarah Johnson}, thanks for flagging the double charge on {4111 1111 1111 1111}. We've refunded the duplicate payment, and someone from {Acme Corp} will call you on {+44 20 7946 0958} before Friday. A copy of this note is on its way to {sarah.j@acmecorp.com}.",
    },
    "Patient referral": {
      re: "Referral summary",
      text: "Summarise this referral for Dr. Priya Shah. The patient, Tom Hughes, was seen at Northwind Health Clinic in Leeds and can be reached on 0113 496 0782 or tom.hughes@mail.com.",
      reply: "Summary for {Dr. Priya Shah}: {Tom Hughes} attended {Northwind Health Clinic} in {Leeds} and has been referred for review. Contact the patient on {0113 496 0782} or at {tom.hughes@mail.com} to book an appointment.",
    },
    "Card dispute": {
      re: "Dispute, holding reply",
      text: "Maria Lopez disputes a payment on card 5555 5555 5555 4444. Write a short holding reply and point her to https://help.acmebank.com/disputes.",
      reply: "Hi {Maria Lopez}, we've received your dispute for the payment on {5555 5555 5555 4444} and paused it while we investigate. You can follow progress at {https://help.acmebank.com/disputes}.",
    },
    "Security alert": {
      re: "Locked account",
      text: "Explain to James Okafor why his Monzo account was locked after a sign-in from 185.220.101.4 in Berlin, and ask him to confirm on +44 7700 900123.",
      reply: "Hi {James Okafor}, we locked your {Monzo} account after a sign-in from {185.220.101.4} in {Berlin} that didn't match your usual activity. If that wasn't you, please confirm on {+44 7700 900123} and we'll secure the account.",
    },
  };
  const FIRST = Object.keys(MEMOS)[0];

  const memo = $("memo");
  let state = Shield.cloak("");
  let current = FIRST;
  let view = "user";
  let streaming = false;

  function bars(text, map, stamp) {
    let i = 0;
    return Shield.segments(text, map).map((s) => {
      if (!s.token) return html(s.text);
      const delay = stamp ? `style="animation-delay:${180 + 150 * i++}ms"` : "";
      return `<span class="redact${stamp ? " stamp" : ""}" tabindex="0" data-real="${html(s.value)}" ${delay}>${html(s.token)}</span>`;
    }).join("");
  }

  function fit() {
    memo.style.height = "auto";
    memo.style.height = memo.scrollHeight + "px";
  }

  function render(stamp) {
    state = Shield.cloak(memo.value);
    $("copy").innerHTML = bars(state.cloaked, state.map, stamp);
    const n = state.map.byToken.size;
    $("tally").textContent = n
      ? `${n} ${n === 1 ? "detail" : "details"} redacted. The originals never left your server.`
      : "Nothing to redact. This copy goes out as written.";
    fit();
    if (!streaming) paintReply(modelReply(), "");
  }

  function modelReply() {
    const m = MEMOS[current];
    return m && memo.value === m.text ? Shield.fill(m.reply, state.map) : Shield.reply(state.map);
  }

  // ---------- memo tabs ----------
  const tabs = $("memo-tabs");
  function selectMemo(name, opts = {}) {
    current = name;
    tabs.querySelectorAll("button").forEach((b) => b.setAttribute("aria-selected", String(b.textContent === name)));
    document.querySelectorAll(".memo-re").forEach((el) => (el.textContent = MEMOS[name].re));
    if (opts.type) return typeIn(MEMOS[name].text);
    memo.value = MEMOS[name].text;
    render(true);
  }
  Object.keys(MEMOS).forEach((name) => {
    const b = document.createElement("button");
    b.type = "button";
    b.setAttribute("role", "tab");
    b.setAttribute("aria-controls", "memo");
    b.textContent = name;
    b.addEventListener("click", () => selectMemo(name));
    tabs.append(b);
  });

  memo.addEventListener("input", () => {
    if (MEMOS[current] && memo.value !== MEMOS[current].text) {
      tabs.querySelectorAll("button").forEach((b) => b.setAttribute("aria-selected", "false"));
      document.querySelectorAll(".memo-re").forEach((el) => (el.textContent = "Your own memo"));
      current = null;
    }
    render(false);
  });

  let typing = 0;
  async function typeIn(text) {
    const run = ++typing;
    if (reduce) { memo.value = text; render(true); return; }
    for (let i = 0; i <= text.length; i += 3) {
      if (run !== typing) return;
      memo.value = text.slice(0, i);
      $("copy").textContent = "";
      fit();
      await wait(12);
    }
    memo.value = text;
    render(true);
  }

  // ---------- the reply ----------

  function paintReply(raw, note) {
    $("reply-sheet").classList.toggle("restored", view === "user");
    $("reply").innerHTML = view === "user"
      ? html(Shield.uncloak(raw, state.map))
      : bars(raw, state.map, false);
    $("stream-note").innerHTML = note;
  }

  async function streamReply() {
    if (streaming) return;
    streaming = true;
    $("replay").disabled = true;
    const raw = modelReply();
    const un = Shield.streamUncloaker(state.map);
    let sent = "", shown = "";
    const caret = '<span class="caret" aria-hidden="true"></span>';
    for (const chunk of Shield.chunks(raw)) {
      sent += chunk;
      const step = un.push(chunk);
      shown += step.text;
      $("reply-sheet").classList.toggle("restored", view === "user");
      $("reply").innerHTML = (view === "user" ? html(shown) : html(sent)) + caret;
      $("stream-note").innerHTML = step.held && view === "user"
        ? `Holding <code>${html(step.held)}</code> until the placeholder is complete`
        : "";
      await wait(40);
    }
    streaming = false;
    $("replay").disabled = false;
    paintReply(raw, "");
  }

  document.querySelectorAll("[data-view]").forEach((b) => b.addEventListener("click", () => {
    view = b.dataset.view;
    document.querySelectorAll("[data-view]").forEach((x) => x.setAttribute("aria-pressed", String(x === b)));
    if (!streaming) paintReply(modelReply(), "");
  }));
  $("replay").addEventListener("click", streamReply);
  new IntersectionObserver((entries, obs) => {
    if (entries[0].isIntersecting) { streamReply(); obs.disconnect(); }
  }, { threshold: 0.6 }).observe($("reply-sheet"));

  // ---------- conversation pages ----------
  const TURNS = [
    ["Turn 1", "You", "I'm Grace Chen from Contoso Ltd. Please plan a two-day visit to our Leeds office."],
    ["Turn 2", "Model", "Of course, {Grace Chen}. Day one is with the {Leeds} team and day two is with clients. Shall I send it on?"],
    ["Turn 3", "You", "Yes, send it to Grace Chen at grace.chen@contoso.com and copy Raj Patel."],
  ];
  function renderPages(copy) {
    const map = Shield.cloak("").map;
    $("pages").innerHTML = TURNS.map(([turn, who, text]) => {
      const raw = who === "You" ? Shield.cloak(text, { map }).cloaked : Shield.fill(text, map);
      const body = copy
        ? bars(raw, map, false)
        : Shield.segments(raw, map).map((s) => s.token
          ? `<span class="restored-mark">${html(s.value)}</span>`
          : html(s.text)).join("");
      return `<div class="page"><div class="page-head"><span>${turn}</span><span>${who}</span></div><p class="typed">${body}</p></div>`;
    }).join("");
    $("pages").classList.toggle("copy", copy);
    const grace = map.byValue.get("Grace Chen");
    $("pages-note").innerHTML = copy
      ? `Grace Chen is <code>${html(grace)}</code> on every page.`
      : "Highlighted details are swapped out before each turn is sent.";
  }
  document.querySelectorAll("[data-turns]").forEach((b) => b.addEventListener("click", () => {
    document.querySelectorAll("[data-turns]").forEach((x) => x.setAttribute("aria-pressed", String(x === b)));
    renderPages(b.dataset.turns === "copy");
  }));
  renderPages(false);

  // ---------- code folders ----------
  const CODE = {
    OpenAI: ["from openai import OpenAI", "+from llmshield import LLMShield", "", "client = OpenAI()", "+shield = LLMShield(llm_func=client.chat.completions.create)", "", "+reply = shield.ask(", '    model="gpt-5",', "    messages=messages,", ")"],
    Anthropic: ["from anthropic import Anthropic", "+from llmshield import LLMShield", "", "client = Anthropic()", "+shield = LLMShield(llm_func=client.messages.create)", "", "+reply = shield.ask(", '    model="claude-sonnet-5-5",', "    max_tokens=1024,", "    messages=messages,", ")"],
    Gemini: ["from google import genai", "+from llmshield import LLMShield", "", "client = genai.Client()", "+shield = LLMShield(llm_func=client.models.generate_content)", "", "+reply = shield.ask(", '    model="gemini-2.5-pro",', "    contents=prompt,", ")"],
    Cohere: ["import cohere", "+from llmshield import LLMShield", "", "client = cohere.ClientV2()", "+shield = LLMShield(llm_func=client.chat)", "", "+reply = shield.ask(", '    model="command-a",', "    messages=messages,", ")"],
    xAI: ["from xai_sdk import Client", "+from llmshield import LLMShield", "", "client = Client()", "+shield = LLMShield(llm_func=client.chat.create)", "", "+reply = shield.ask(", '    model="grok-4",', "    messages=messages,", ")"],
  };
  const folders = $("folders");
  function showCode(name) {
    folders.querySelectorAll("button").forEach((b) => b.setAttribute("aria-selected", String(b.textContent === name)));
    $("code-sheet").innerHTML = CODE[name]
      .map((l) => l.startsWith("+") ? `<span class="changed">${html(l.slice(1))}</span>` : html(l))
      .join("\n");
  }
  Object.keys(CODE).forEach((name) => {
    const b = document.createElement("button");
    b.type = "button";
    b.setAttribute("role", "tab");
    b.setAttribute("aria-controls", "code-panel");
    b.textContent = name;
    b.addEventListener("click", () => showCode(name));
    folders.append(b);
  });
  showCode("OpenAI");

  // arrow keys move between tabs in a tablist
  document.querySelectorAll('[role="tablist"]').forEach((list) => list.addEventListener("keydown", (e) => {
    if (e.key !== "ArrowRight" && e.key !== "ArrowLeft") return;
    const items = [...list.querySelectorAll('[role="tab"]')];
    const i = items.indexOf(document.activeElement);
    if (i === -1) return;
    const next = items[(i + (e.key === "ArrowRight" ? 1 : items.length - 1)) % items.length];
    next.focus();
    next.click();
  }));

  // ---------- copy buttons ----------
  document.querySelectorAll("[data-copy]").forEach((b) => b.addEventListener("click", async () => {
    const label = b.querySelector(".copy");
    try {
      await navigator.clipboard.writeText(b.dataset.copy);
      label.textContent = "Copied";
    } catch {
      getSelection().selectAllChildren(b.firstElementChild);
      label.textContent = "Press Ctrl+C";
    }
    setTimeout(() => (label.textContent = "Copy"), 1600);
  }));

  document.fonts.ready.then(() => selectMemo(FIRST, { type: true }));
})();
