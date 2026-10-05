/*
 * A small in-browser approximation of llmshield's cloak/uncloak round trip,
 * used only to power the interactive demos on the landing page.
 */
(function (global) {
  const GROUP = {
    PERSON: "noun", ORGANISATION: "noun", PLACE: "noun",
    PHONE: "number", CREDIT_CARD: "number",
    EMAIL: "locator", URL: "locator", IP_ADDRESS: "locator",
  };

  const FIRST_NAMES = new Set((
    "Aisha Alice Amelia Ana Anna Arjun Ben Carlos Charlotte Chen Chloe " +
    "Daniel David Diego Elena Emily Emma Fatima Grace Hannah Harry Isabel " +
    "Jack James Jane Jasmine John Jonas Jose Julia Kenji Laura Leo Liam " +
    "Lucas Lucy Maria Mark Maya Mei Michael Mohammed Nadia Noah Olivia Omar " +
    "Oscar Priya Rahul Raj Rosa Sam Sarah Sofia Sophie Tom Wei Yuki Zara " +
    "Margaret Robert Thomas Peter Helen Ruth Ines Aditya Sebastian Felix"
  ).split(" "));
  const TITLES = "Dr|Mr|Mrs|Ms|Mx|Prof|Sir|Dame";
  const ORG_SUFFIX =
    "Corp|Inc|Ltd|LLC|PLC|GmbH|Bank|Hospital|Clinic|Partners|Group|Labs|" +
    "Health|Capital|Insurance|University|Trust|Pharma|Logistics|Legal";
  const KNOWN_ORGS = ["NHS", "HMRC", "Barclays", "Monzo", "Deloitte",
    "Stripe", "Acme", "Northwind", "Contoso"];
  const PLACES = ["London", "Manchester", "Leeds", "Bristol", "Edinburgh",
    "Glasgow", "Cardiff", "Belfast", "Dublin", "Paris", "Berlin", "Madrid",
    "Lisbon", "Mumbai", "Bangalore", "Tokyo", "Toronto", "Boston",
    "Chicago", "Austin", "Seattle", "New York", "San Francisco",
    "Camden", "Shoreditch", "Baker Street", "Oxford Street", "Harley Street"];

  const esc = (s) => s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");

  const RULES = [
    ["EMAIL", /\b[a-zA-Z0-9](?:[a-zA-Z0-9+]|[._-](?![._-]))*[a-zA-Z0-9]?@[a-zA-Z0-9-]+(?:\.[a-zA-Z0-9-]+)*\.[a-zA-Z]{2,}\b/g],
    ["URL", /https?:\/\/(?:[\w-]+\.)*[\w-]+\.[\w-]+(?:\/[^\s,)]*[^\s.,)])?/g],
    ["IP_ADDRESS", /\b(?:(?:25[0-5]|2[0-4]\d|[01]?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|[01]?\d?\d)\b/g],
    ["CREDIT_CARD", /\b(?:\d[ -]?){12,18}\d\b/g, luhn],
    ["PHONE", /(?<![\d\w])\+?\d[\d\s().-]{8,16}\d(?!\d)/g],
    ["PERSON", new RegExp(
      `\\b(?:(?:${TITLES})\\.?\\s)?(?:${[...FIRST_NAMES].join("|")})` +
      `(?:\\s[A-Z][a-z]+(?:-[A-Z][a-z]+)?)?\\b`, "g")],
    ["PERSON", new RegExp(`\\b(?:${TITLES})\\.?\\s[A-Z][a-z]+\\b`, "g")],
    ["ORGANISATION", new RegExp(
      `\\b(?:[A-Z][\\w&]*\\s){0,2}[A-Z][\\w&]*\\s(?:${ORG_SUFFIX})\\b`, "g")],
    ["ORGANISATION", new RegExp(`\\b(?:${KNOWN_ORGS.join("|")})\\b`, "g")],
    ["PLACE", new RegExp(`\\b(?:${PLACES.map(esc).join("|")})\\b`, "g")],
  ];

  function luhn(raw) {
    const digits = raw.replace(/\D/g, "");
    if (digits.length < 13) return false;
    let sum = 0;
    for (let i = 0; i < digits.length; i++) {
      let d = +digits[digits.length - 1 - i];
      if (i % 2) { d *= 2; if (d > 9) d -= 9; }
      sum += d;
    }
    return sum % 10 === 0;
  }

  function detect(text, allow = []) {
    const allowed = new Set(allow.map((a) => a.toLowerCase()));
    const found = [];
    const taken = (s, e) => found.some((f) => s < f.end && e > f.start);
    for (const [type, re, check] of RULES) {
      re.lastIndex = 0;
      let m;
      while ((m = re.exec(text))) {
        const value = m[0].trim();
        const start = m.index + m[0].indexOf(value);
        const end = start + value.length;
        if (check && !check(value)) continue;
        if (allowed.has(value.toLowerCase())) continue;
        if (taken(start, end)) continue;
        found.push({ type, value, start, end });
      }
    }
    return found.sort((a, b) => a.start - b.start);
  }

  /** Cloak text; pass an existing map to keep placeholders stable. */
  function cloak(text, opts = {}) {
    const map = opts.map || { byValue: new Map(), byToken: new Map(), n: 0 };
    const spans = detect(text, opts.allow).map((s) => {
      let token = map.byValue.get(s.value);
      if (!token) {
        token = `<${s.type}_${map.n++}>`;
        map.byValue.set(s.value, token);
        map.byToken.set(token, s);
      }
      return { ...s, token, group: GROUP[s.type] };
    });
    let out = "", last = 0;
    for (const s of spans) {
      out += text.slice(last, s.start) + s.token;
      last = s.end;
    }
    out += text.slice(last);
    return { cloaked: out, spans, map };
  }

  const TOKEN_RE = /<([A-Z_]+)_(\d+)>/g;

  function uncloak(text, map) {
    return text.replace(TOKEN_RE, (t) =>
      map.byToken.has(t) ? map.byToken.get(t).value : t);
  }

  /** Split text into plain and token segments for rendering. */
  function segments(text, map) {
    const out = [];
    let last = 0, m;
    TOKEN_RE.lastIndex = 0;
    while ((m = TOKEN_RE.exec(text))) {
      if (m.index > last) out.push({ text: text.slice(last, m.index) });
      const hit = map && map.byToken.get(m[0]);
      out.push({
        token: m[0], type: m[1], group: GROUP[m[1]] || "noun",
        value: hit ? hit.value : m[0],
      });
      last = m.index + m[0].length;
    }
    if (last < text.length) out.push({ text: text.slice(last) });
    return out;
  }

  /** Write a plausible model reply that only ever refers to placeholders. */
  function reply(map) {
    const first = (type) => {
      for (const [tok, s] of map.byToken) if (s.type === type) return tok;
      return null;
    };
    const p = first("PERSON"), o = first("ORGANISATION"),
      e = first("EMAIL"), ph = first("PHONE"), c = first("CREDIT_CARD"),
      pl = first("PLACE"), u = first("URL"), ip = first("IP_ADDRESS");
    if (!map.byToken.size) {
      return "Nothing personal in there, so this one went through as written.";
    }
    const parts = [];
    parts.push(p ? `Here's a reply for ${p}` : "Here's a draft reply");
    if (o) parts[0] += ` at ${o}`;
    parts[0] += ".";
    if (c) parts.push(`I've flagged the payment on ${c} for a refund.`);
    if (e && ph) parts.push(`Send it to ${e} and offer a call on ${ph} if they'd rather talk it through.`);
    else if (e) parts.push(`Send it to ${e}.`);
    else if (ph) parts.push(`Offer a call on ${ph}.`);
    if (pl) parts.push(`I've suggested meeting in ${pl} next week.`);
    if (u) parts.push(`Link them to ${u} for the details.`);
    if (ip) parts.push(`The sign-in from ${ip} is worth flagging to security.`);
    return parts.join(" ");
  }

  /**
   * Stream a string in small chunks, the way a provider would. Placeholders
   * are often split across chunks, which is what the stream uncloaker is
   * built to handle.
   */
  function* chunks(text) {
    let i = 0;
    while (i < text.length) {
      const n = 2 + Math.floor(Math.random() * 5);
      yield text.slice(i, i + n);
      i += n;
    }
  }

  /** Mirror of llmshield's stream uncloaker: hold back partial tokens. */
  function streamUncloaker(map) {
    let buffer = "";
    return {
      push(chunk) {
        buffer += chunk;
        const open = buffer.lastIndexOf("<");
        let ready = buffer, held = "";
        if (open !== -1 && buffer.indexOf(">", open) === -1) {
          ready = buffer.slice(0, open);
          held = buffer.slice(open);
        }
        buffer = held;
        return { text: uncloak(ready, map), held };
      },
      flush() {
        const out = uncloak(buffer, map);
        buffer = "";
        return out;
      },
    };
  }

  /** Turn "{Sarah Johnson}" references into that value's placeholder. */
  function fill(template, map) {
    return template.replace(/\{(.+?)\}/g, (_, v) => map.byValue.get(v) || v);
  }

  global.Shield = {
    cloak, uncloak, segments, reply, fill, chunks, streamUncloaker, detect,
    GROUP,
  };
})(window);
