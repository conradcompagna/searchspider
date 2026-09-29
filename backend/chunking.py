"""Preserve short orders; attach short fragments only to chunks of the same order."""
from search_settings import SETTINGS


def merge_short_passages(passages, minimum=SETTINGS.short_chunk_words):
    out = []
    for passage in passages:
        if out and len(out[-1].split()) < minimum:
            out[-1] += " " + passage
        else:
            out.append(passage)
    if len(out) > 1 and len(out[-1].split()) < minimum:
        tail = out.pop()
        out[-1] += " " + tail
    return out
