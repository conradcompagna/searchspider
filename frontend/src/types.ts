export type Status_ = "relevant" | "partial" | "dropped" | "missing" | "not_assessed";

export type Verdict = { v: "r" | "p" | "n"; s?: string } | null;

export interface OrderHit {
  id: string;
  part: number;
  date: string;
  date_supplied: boolean;
  date_uncertain: boolean;
  pdf: string;
  pdf_pages: number[];
  topics: string[];
  score: number;
  fusion_score?: number;
  rerank?: number | null;
  score_kind?: "hybrid_consensus";
  query_count?: number;
  rank?: number;
  en_passages: string[];
  my_passages: string[];
  my_alignment: "paired" | "same_date" | null;
  hit_en: number[];
  hit_my: number[];
  spider?: Verdict;
  alias?: string;
  judged?: boolean;
  found?: string[];
  read?: boolean;
  status?: Status_;
  n_routes?: number;
  deeper?: number;
  terms?: string[];
}

export interface Sentence {
  id: string;
  my: string;
  en: string;
}

export interface PageHit {
  page_id: string;
  volume: number;
  page: number;
  score: number;
  fusion_score?: number;
  rerank?: number | null;
  score_kind?: "hybrid_consensus";
  query_count?: number;
  rank?: number;
  hits: string[];
  sentences: Sentence[];
  spider?: Verdict;
  alias?: string;
  judged?: boolean;
  found?: string[];
  read?: boolean;
  status?: Status_;
  n_routes?: number;
  deeper?: number;
  terms?: string[];
}

export interface Plan {
  queries: string[];
  my: string[];
  from: number | null;
  to: number | null;
}

export interface Addendum {
  kind: "followup" | "deeper";
  n: number;
  q?: string;
  text: string;
  tally?: string;
  assessed?: number;
  in?: number;
  out?: number;
}

export interface Results {
  can_continue?: boolean;
  access_mode?: string;
  run_id?: string;
  addenda?: Addendum[];
  orders: OrderHit[];
  pages: PageHit[];
  ms: number;
  reranked?: boolean;
  rerank_ms?: number;
  plan?: Plan;
  summary?: string;
  refs?: Record<string, { kind: string; id: string; label: string }>;
  usage?: { in: number; out: number; think: number; calls: number };
  trail?: TrailStep[];
  rounds?: { step: number | "final"; in: number; out: number; docs?: number; text?: string; allowed?: string[] | null; label?: string }[];
  aliases?: string[];
  keywords?: { stem: string; word: string; n: number }[];
  highlight_terms?: string[];
  settings?: Record<string, string | number>;
  stop?: string;
  reading_rounds?: number;
  completed_reading_rounds?: number;
  response_tokens?: number;
  counts?: Record<Status_, number>;
}

export interface TrailStep {
  step: number;
  tool: string;
  args: Record<string, unknown>;
  result: string;
  chars: number;
  tokens?: number;
  output?: string;
}

export interface SpiderSettings {
  reading_rounds: number;
}

export interface Status {
  orders: number;
  sentences: number;
  pages: number;
  spider: boolean;
  reranker: boolean;
  spider_model: string;
  spider_error: string | null;
  kg_triples: number;
  version?: string;
  settings?: Record<string, string | number>;
}

export interface Access {
  mode: "local" | "free" | "paid";
  connected: boolean;
  login_url: string;
  billing_url: string;
  trial_available: boolean;
  free_daily_budget_usd?: number;
  free_budget_remaining_usd?: number;
}

export interface Filters {
  orders: boolean;
  chronicle: boolean;
  yearFrom: string;
  yearTo: string;
  parts: number[];
  volumes: number[];
}
