from __future__ import annotations

from typing import Dict, List, Optional

from sqlalchemy import text
import logging
import re

from database.connection import async_session_maker

uvicorn_logger = logging.getLogger("uvicorn.error")


def _log(msg: str) -> None:
    uvicorn_logger.info(msg)


def _compact_sql(sql: str) -> str:
    return re.sub(r"\s+", " ", (sql or "").strip())


def _to_sql_literal(value: object) -> str:
    if value is None:
        return "NULL"
    if isinstance(value, (list, tuple)):
        inner = ", ".join(_to_sql_literal(v) for v in value)
        return f"ARRAY[{inner}]"
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    if isinstance(value, (int, float)):
        return str(value)
    text = str(value).replace("'", "''")
    return f"'{text}'"


def _render_sql_for_debug(sql: str, params: Dict[str, object]) -> str:
    rendered = sql
    # Replace longer keys first to avoid accidental partial replacement.
    for key in sorted(params.keys(), key=len, reverse=True):
        rendered = rendered.replace(f":{key}", _to_sql_literal(params[key]))
    return rendered


def _state_limits(states: List[str], total_limit: int) -> Dict[str, int]:
    if not states:
        return {}
    count = len(states)
    base = max(1, total_limit // count)
    rem = max(0, total_limit - (base * count))
    limits: Dict[str, int] = {}
    for idx, state in enumerate(states):
        limits[state] = base + (1 if idx < rem else 0)
    return limits


async def fetch_cutoff_recommendations(
    *,
    metric_type: str,
    metric_value: int,
    home_state: str,
    target_states: List[str],
    category: str,
    college_type_patterns: Optional[List[str]] = None,
    quota_patterns: Optional[List[str]] = None,
    total_limit: int = 20,
) -> List[Dict]:
    """
    Query cutoff table and return nearest eligible distinct colleges.
    - score flow: score <= user score
    - rank flow: air_rank >= user rank
    Distinctness is applied at institution+state level.
    """
    if not target_states:
        _log("[CUTOFF_SQL] No target states found; skipping query.")
        return []

    per_state_limits = _state_limits(target_states, total_limit)
    category_like = f"%{category.strip()}%"
    all_rows: List[Dict] = []

    score_sql_base = """
        SELECT DISTINCT ON (COALESCE(institution_name, college_name))
          state,
          COALESCE(institution_name, college_name) AS institution_name,
          college_name,
          college_type,
          course,
          category,
          sub_category,
          seat_type,
          quota,
          domicile,
          eligibility,
          score,
          air_rank,
          round
        FROM neet_ug_2025_cutoffs
        WHERE state ILIKE :state_like
          AND score IS NOT NULL
          AND score <= :user_score
          AND category ILIKE :category_like
          AND domicile ILIKE ANY(:domicile_patterns)
        ORDER BY
          COALESCE(institution_name, college_name),
          score DESC
        LIMIT :state_limit;
        """

    rank_sql_base = """
        SELECT DISTINCT ON (COALESCE(institution_name, college_name))
          state,
          COALESCE(institution_name, college_name) AS institution_name,
          college_name,
          college_type,
          course,
          category,
          sub_category,
          seat_type,
          quota,
          domicile,
          eligibility,
          score,
          air_rank,
          round
        FROM neet_ug_2025_cutoffs
        WHERE state ILIKE :state_like
          AND air_rank IS NOT NULL
          AND air_rank >= :user_rank
          AND category ILIKE :category_like
          AND domicile ILIKE ANY(:domicile_patterns)
        ORDER BY
          COALESCE(institution_name, college_name),
          air_rank ASC
        LIMIT :state_limit;
        """

    async with async_session_maker() as db:
        _log(
            "[CUTOFF_SQL] Starting query run | "
            f"metric_type={metric_type} metric_value={metric_value} "
            f"home_state={home_state} category={category} total_limit={total_limit} "
            f"target_states={target_states}"
        )
        for state in target_states:
            params = {
                "state": state,
                "state_like": state,
                "category_like": category_like,
                "state_limit": per_state_limits.get(state, 1),
            }
            if state.lower() == home_state.lower():
                params["domicile_patterns"] = ["%DOMICILE%"]
            else:
                params["domicile_patterns"] = ["%NON-DOMICILE%", "%NON DOMICILE%", "%OPEN%"]

            sql_base = score_sql_base if metric_type == "score" else rank_sql_base
            if college_type_patterns:
                sql_base = sql_base.replace(
                    "        ORDER BY",
                    "          AND college_type ILIKE ANY(:college_type_patterns)\n        ORDER BY",
                )
                params["college_type_patterns"] = college_type_patterns
            if quota_patterns:
                sql_base = sql_base.replace(
                    "        ORDER BY",
                    "          AND quota ILIKE ANY(:quota_patterns)\n        ORDER BY",
                )
                params["quota_patterns"] = quota_patterns
            sql_text_compact = _compact_sql(sql_base)

            if metric_type == "score":
                params["user_score"] = metric_value
                _log(f"[CUTOFF_SQL] Query text (score): {sql_text_compact}")
                _log(f"[CUTOFF_SQL] Query text (score, rendered): {_render_sql_for_debug(sql_text_compact, params)}")
                _log(f"[CUTOFF_SQL] Executing score query | state={state} params={params}")
                score_sql = text(sql_base)
                result = await db.execute(score_sql, params)
            else:
                params["user_rank"] = metric_value
                _log(f"[CUTOFF_SQL] Query text (rank): {sql_text_compact}")
                _log(f"[CUTOFF_SQL] Query text (rank, rendered): {_render_sql_for_debug(sql_text_compact, params)}")
                _log(f"[CUTOFF_SQL] Executing rank query | state={state} params={params}")
                rank_sql = text(sql_base)
                result = await db.execute(rank_sql, params)

            rows = [dict(row) for row in result.mappings().all()]
            _log(f"[CUTOFF_SQL] Rows fetched | state={state} count={len(rows)}")
            all_rows.extend(rows)

    if metric_type == "score":
        all_rows.sort(key=lambda r: (abs(metric_value - float(r.get("score") or 0)), -(float(r.get("score") or 0))))
    else:
        all_rows.sort(key=lambda r: (abs(int(r.get("air_rank") or 0) - metric_value), int(r.get("air_rank") or 10**9)))

    deduped: List[Dict] = []
    seen = set()
    for row in all_rows:
        key = (row.get("state"), row.get("institution_name") or row.get("college_name"))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)
        if len(deduped) >= total_limit:
            break

    _log(
        "[CUTOFF_SQL] Completed query run | "
        f"raw_rows={len(all_rows)} deduped_rows={len(deduped)} returned={min(len(deduped), total_limit)}"
    )
    return deduped


def format_cutoff_markdown(
    *,
    rows: List[Dict],
    metric_type: str,
    metric_value: int,
    category: str,
    home_state: str,
    target_states: List[str],
    display_limit: int = 10,
) -> str:
    if not rows:
        metric_label = "AIR Rank" if metric_type == "rank" else "Score"
        states_label = ", ".join(target_states) if target_states else "your selected states"
        domicile_mode = (
            "non-domicile/open"
            if any(s.lower() != (home_state or "").lower() for s in target_states)
            else "domicile"
        )
        return (
            "I checked the 2025 cutoff records carefully, but I could not find a direct match for your current preference set.\n\n"
            "### Profile I Used\n\n"
            f"- {metric_label}: **{metric_value}**\n"
            f"- Category: **{category}**\n"
            f"- Home state: **{home_state}**\n"
            f"- Target state(s): **{states_label}**\n"
            f"- Eligibility mode applied: **{domicile_mode}**\n\n"
            "### What This Means\n\n"
            "This usually happens when the selected combination is too strict for the available rows "
            "(for example: specific quota + college type + domicile mode at this rank/score range).\n\n"
            "### I Can Help You Explore Better Options\n\n"
            "If you want, I can immediately retry with one of these counselor-style refinements:\n"
            "1. Keep same state, but relax quota or college type\n"
            "2. Keep same category, but include nearby state options\n"
            "3. Show closest available colleges in the same rank/score range\n\n"
            "Reply with what you prefer, and I will refine it for you."
        )

    shown = rows[:display_limit]
    metric_label = "Score" if metric_type == "score" else "AIR Rank"
    states_label = ", ".join(target_states)

    lines = [
        "### Best-Match Colleges (Based on 2025 Cutoff Data)",
        "",
        f"- Profile used: **{metric_label} = {metric_value}**, **Category = {category}**",
        f"- Home state: **{home_state}**",
        f"- Target states: **{states_label}**",
        f"- Showing top **{len(shown)}** of **{len(rows)}** matched options",
        "",
        "| # | Institution | State | Category | Quota | Domicile | AIR | Score | Round |",
        "|---|---|---|---|---|---|---:|---:|---|",
    ]

    for idx, row in enumerate(shown, start=1):
        lines.append(
            "| {idx} | {inst} | {state} | {cat} | {quota} | {dom} | {air} | {score} | {round_} |".format(
                idx=idx,
                inst=(row.get("institution_name") or row.get("college_name") or "-").replace("|", " "),
                state=row.get("state") or "-",
                cat=row.get("category") or "-",
                quota=row.get("quota") or "-",
                dom=row.get("domicile") or "-",
                air=row.get("air_rank") if row.get("air_rank") is not None else "-",
                score=row.get("score") if row.get("score") is not None else "-",
                round_=row.get("round") or "-",
            )
        )

    lines.extend(
        [
            "",
            "> Disclaimer: Cutoffs vary year to year and by round/quota/sub-category. Please verify final options on official MCC/state counselling portals.",
            "",
            "Would you like me to refine this further by quota, college type, or a specific state?",
        ]
    )
    return "\n".join(lines)

