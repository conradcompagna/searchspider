import type { Access, Filters, OrderHit, PageHit, Results, SpiderSettings, Status, Status_, TrailStep } from "./types";

// Same origin when served by backend/app.py; direct to the local server when opened as a file.
const BASE = location.protocol === "file:" ? "http://127.0.0.1:8765" : location.pathname.startsWith("/searchspider") ? "/searchspider" : "";
function headers(): Record<string, string> {
  return { "Content-Type": "application/json" };
}

async function get<T>(path: string): Promise<T> {
  const r = await fetch(BASE + path);
  if (!r.ok) throw new Error((await r.json().catch(() => null))?.detail ?? r.statusText);
  return r.json();
}

export const status = () => get<Status>("/api/status");
export const access = () => get<Access>("/api/access");
export const logout = () => post<{ ok: boolean }>("/api/logout", {});

function sources(f: Filters) {
  return [f.orders && "order", f.chronicle && "chronicle"].filter(Boolean) as string[];
}

export function search(q: string, f: Filters): Promise<Results> {
  const p = new URLSearchParams({ q, sources: sources(f).join(",") });
  if (f.yearFrom) p.set("year_from", f.yearFrom);
  if (f.yearTo) p.set("year_to", f.yearTo);
  if (f.parts.length) p.set("parts", f.parts.join(","));
  if (f.volumes.length) p.set("volumes", f.volumes.join(","));
  return get<Results>("/api/search?" + p.toString());
}

export async function spider(q: string, f: Filters, st: SpiderSettings): Promise<Results> {
  const r = await fetch(BASE + "/api/spider", {
    method: "POST",
    headers: headers(),
    body: JSON.stringify({
      q,
      sources: sources(f),
      year_from: f.yearFrom ? Number(f.yearFrom) : null,
      year_to: f.yearTo ? Number(f.yearTo) : null,
      parts: f.parts.map(String),
      volumes: f.volumes.map(String),
      reading_rounds: st.reading_rounds,
    }),
  });
  if (!r.ok) throw new Error((await r.json().catch(() => null))?.detail ?? r.statusText);
  return r.json();
}

async function post<T>(path: string, body: unknown): Promise<T> {
  const r = await fetch(BASE + path, { method: "POST", headers: headers(), body: JSON.stringify(body) });
  if (!r.ok) throw new Error((await r.json().catch(() => null))?.detail ?? r.statusText);
  return r.json();
}

type RunUpdate = { trail: TrailStep[]; rounds: NonNullable<Results["rounds"]>; usage: NonNullable<Results["usage"]> };

export const followup = (run_id: string, q: string) =>
  post<RunUpdate & { followup: { n: number; q: string; text: string; in: number; out: number } }>("/api/spider/followup", { run_id, q });

export const deeper = (run_id: string) =>
  post<RunUpdate & { deeper: { n: number; text: string; tally: string; assessed: number; in: number; out: number }; updated: (OrderHit | PageHit)[];
    counts: Record<Status_, number>; remaining: number }>("/api/spider/deeper", { run_id });
