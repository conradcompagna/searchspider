import type { OrderHit, PageHit, Results, SpiderSettings, Verdict } from "./types";

const verdict = (v: Verdict | undefined, judged?: boolean, status?: string) =>
  status ? status.replace("_", " ") :
  v === undefined ? "" : v === null ? (judged === false ? "not judged" : "not relevant") : { r: "relevant", p: "partial", n: "not relevant" }[v.v];

function annotation(v: Verdict | undefined, judged: boolean | undefined, found?: string[], status?: string, score?: number | null, deeper?: number): string {
  const parts = [];
  if (v !== undefined || status) parts.push(`**SearchSpider: ${verdict(v, judged, status)}**${deeper ? ` (judged in push deeper ${deeper})` : ""}${score != null ? ` (score ${score.toFixed(2)})` : ""}${v?.s ? ` — ${v.s}` : ""}`);
  if (found?.length) parts.push(`_Found by: ${found.join("; ")}_`);
  return parts.join("\n\n");
}

function scoreDetails(x: OrderHit | PageHit): string {
  return `_Ranking: hybrid relevance ${x.score.toFixed(4)}._`;
}

// Full order: every English passage (matched ones marked ►) and the full Burmese text where it exists.
function orderMd(o: OrderHit): string {
  const pages = o.pdf_pages.length > 1 ? `${o.pdf_pages[0]}–${o.pdf_pages[o.pdf_pages.length - 1]}` : `${o.pdf_pages[0]}`;
  const hits = new Set(o.hit_en);
  return [
    `### ${o.alias ? o.alias + " · " : ""}Royal Order ${o.date} — Than Tun, ROB Part ${o.part}, PDF p. ${pages}`,
    annotation(o.spider, o.judged, o.found, o.status, o.score, o.deeper),
    scoreDetails(o),
    o.topics.length ? `Than Tun subject index: ${o.topics.join("; ")}` : "",
    "**English**",
    o.en_passages.map((p, i) => (hits.has(i) ? `► ${p}` : p)).join("\n\n"),
    o.my_passages.length ? `**Burmese**${o.my_alignment === "same_date" ? " (all Burmese orders of this date)" : ""}\n\n${o.my_passages.join("\n\n")}` : "",
  ].filter(Boolean).join("\n\n");
}

// Full chronicle page: every sentence, Burmese then English, matched ones marked ►.
function pageMd(p: PageHit): string {
  const hits = new Set(p.hits);
  return [
    `### ${p.alias ? p.alias + " · " : ""}Konbaungset Yazawin vol. ${p.volume}, p. ${p.page}`,
    annotation(p.spider, p.judged, p.found, p.status, p.score, p.deeper),
    scoreDetails(p),
    p.sentences.map((s) => `${hits.has(s.id) ? "► " : ""}${s.my}\n${s.en}`).join("\n\n"),
  ].filter(Boolean).join("\n\n");
}

export type Shown = { orders: OrderHit[]; pages: PageHit[]; chrono: boolean; description: string };

// Writes only what is shown on the page (visibility filters, cutoff), in the order shown.
export function toMarkdown(q: string, r: Results, st: SpiderSettings | null, shown: Shown): string {
  const out = [`# ${q}`, "", `_SearchSpider, ${new Date().toISOString().slice(0, 10)}. ► marks the passages that matched._`, "",
    `_This file contains the ${shown.orders.length} orders and ${shown.pages.length} chronicle pages shown on screen (${shown.description})._`, ""];
  if (r.summary !== undefined) {
    out.push("## SearchSpider answer", "", r.summary || "(no answer)", "");
    for (const a of r.addenda ?? [])
      out.push(a.kind === "followup" ? `### Follow-up ${a.n}: ${a.q}` : `### Push deeper ${a.n}: ${a.assessed} more documents judged — ${a.tally}`, "",
        a.text, "", ...(a.in !== undefined ? [`_${a.in.toLocaleString()} input / ${(a.out ?? 0).toLocaleString()} output tokens._`, ""] : []));
    if (r.counts) out.push(`**Evidence:** ${r.orders.length} orders and ${r.pages.length} chronicle pages gathered; ${r.counts.relevant + r.counts.partial + r.counts.dropped + r.counts.missing} judged: ${r.counts.relevant} relevant, ${r.counts.partial} partly relevant, ${r.counts.dropped} dropped${r.counts.missing ? `, ${r.counts.missing} not returned` : ""}; ${r.counts.not_assessed} unjudged.`, "");
    if (st && r.usage) out.push(`_${r.usage.calls} Gemini calls, ${r.usage.in.toLocaleString()} input / ${r.usage.out.toLocaleString()} output tokens._`, "");
    if (r.rounds?.length) {
      out.push("## Research trail", "");
      for (const rd of r.rounds)
        out.push(`- **${rd.label ?? `Stage ${rd.step}`}**${rd.in > 0 ? ` (${rd.in.toLocaleString()} in / ${rd.out.toLocaleString()} out tokens)` : ""}${rd.text ? `: ${rd.text}` : ""}`);
      out.push("");
    }
  }
  if (r.aliases?.length) out.push(`Name variants also searched: ${r.aliases.join(", ")}`, "");
  if (r.counts && !shown.chrono) {
    // evidence dossier as displayed: grouped by verdict, relevant first; within a group, orders then chronicle pages
    const groups: [string, string][] = [["relevant", "Relevant"], ["partial", "Partial"], ["not_assessed", "Unjudged"],
      ["missing", "Verdict not returned"], ["dropped", "Dropped"]];
    for (const [st, title] of groups) {
      const o = shown.orders.filter((x) => x.status === st);
      const p = shown.pages.filter((x) => x.status === st);
      if (!o.length && !p.length) continue;
      out.push(`## ${title} — ${o.length} orders, ${p.length} chronicle pages`, "",
        ...o.map(orderMd).flatMap((s) => [s, "", "---", ""]), ...p.map(pageMd).flatMap((s) => [s, "", "---", ""]));
    }
    return out.join("\n");
  }
  const order = shown.chrono ? ", by date" : "";
  if (shown.orders.length) out.push(`## Royal Orders (${shown.orders.length}${order})`, "", ...shown.orders.map(orderMd).flatMap((s) => [s, "", "---", ""]));
  if (shown.pages.length) out.push(`## Konbaungset Yazawin (${shown.pages.length} pages${shown.chrono ? ", by volume and page" : ""})`, "", ...shown.pages.map(pageMd).flatMap((s) => [s, "", "---", ""]));
  return out.join("\n");
}

export function save(name: string, text: string, type: string) {
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([text], { type }));
  a.download = name;
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
}
