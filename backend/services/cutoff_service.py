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
    text_val = str(value).replace("'", "''")
    return f"'{text_val}'"


def _render_sql_for_debug(sql: str, params: Dict[str, object]) -> str:
    rendered = sql
    for key in sorted(params.keys(), key=len, reverse=True):
        rendered = rendered.replace(f":{key}", _to_sql_literal(params[key]))
    return rendered


def _is_central_scope(target_states: List[str]) -> bool:
    """
    True when we're querying MCC/central counselling (deemed, AIIMS, JIPMER, etc.)
    For these, we NEVER apply domicile or category filters.
    """
    if not target_states:
        return False
    return all(str(s).strip().upper() == "MCC" for s in target_states)


def _domicile_sql_filter(*, home_state: str, row_state: str) -> str:
    """
    Domicile column uses exact semantic values: DOMICILE, NON DOMICILE, OPEN.

    - OPEN: always eligible (any student from any state).
    - Same home state as the college's state: DOMICILE or OPEN.
    - Different home state: NON DOMICILE or OPEN.

    Never use ILIKE '%DOMICILE%' — it incorrectly matches NON DOMICILE.
    We normalize with TRIM/UPPER and hyphen→space so 'NON-DOMICILE' and 'NON DOMICILE' match.
    """
    same = (home_state or "").strip().lower() == (row_state or "").strip().lower()
    if same:
        allowed = "('DOMICILE', 'OPEN')"
    else:
        allowed = "('NON DOMICILE', 'OPEN')"
    return f"""AND (
          REPLACE(TRIM(UPPER(COALESCE(domicile, ''))), '-', ' ')
          IN {allowed}
        )"""


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


def _format_score_for_output(raw_score: object) -> str:
    if raw_score is None:
        return "-"
    try:
        return str(int(float(raw_score)))
    except (TypeError, ValueError):
        return str(raw_score)


async def fetch_cutoff_recommendations(
    *,
    metric_type: str,
    metric_value: int,
    home_state: str,
    target_states: List[str],
    category: str,
    sub_category: Optional[str] = None,
    college_type_filter: Optional[str] = None,
    college_type_filters: Optional[List[str]] = None,
    course_filter: Optional[str] = None,
    quota_keywords: Optional[List[str]] = None,
    seat_type_keywords: Optional[List[str]] = None,
    apply_category_filter: bool = True,
    apply_domicile_filter: bool = True,
    total_limit: int = 20,
) -> List[Dict]:
    """
    Query cutoff table and return nearest eligible distinct colleges.

    KEY RULES:
    - Treat all filters as optional across scopes.
    - Apply a filter only when it is present/enabled by caller intent.
    - No hard scope-based blocking of category/domicile/sub-category filters.
    """
    if not target_states:
        _log("[CUTOFF_SQL] No target states found; skipping query.")
        return []

    # ── Determine if this is central/MCC scope (used only for diagnostics) ──
    central_scope = _is_central_scope(target_states)

    per_state_limits = _state_limits(target_states, total_limit)
    category_norm = str(category or "").strip().upper()
    all_rows: List[Dict] = []

    # Base SQL templates — __CATEGORY_FILTER__ and __DOMICILE_FILTER__ are placeholders
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
          __CATEGORY_FILTER__
          __DOMICILE_FILTER__
          __COLLEGE_TYPE_FILTER__
          __COURSE_FILTER__
          __QUOTA_FILTER__
          __SEAT_TYPE_FILTER__
          __SUB_CATEGORY_FILTER__
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
          __CATEGORY_FILTER__
          __DOMICILE_FILTER__
          __COLLEGE_TYPE_FILTER__
          __COURSE_FILTER__
          __QUOTA_FILTER__
          __SEAT_TYPE_FILTER__
          __SUB_CATEGORY_FILTER__
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
            f"target_states={target_states} central_scope={central_scope} "
            f"college_type_filter={college_type_filter} college_type_filters={college_type_filters} "
            f"course_filter={course_filter} "
            f"quota_keywords={quota_keywords} seat_type_keywords={seat_type_keywords} "
            f"apply_category_filter={apply_category_filter} apply_domicile_filter={apply_domicile_filter}"
        )

        for state in target_states:
            params: Dict[str, object] = {
                "state_like": state,
                "state_limit": per_state_limits.get(state, 1),
            }
            sql_base = score_sql_base if metric_type == "score" else rank_sql_base

            # ── Category filter ──
            if apply_category_filter:
                params["category_exact"] = category_norm
                sql_base = sql_base.replace(
                    "__CATEGORY_FILTER__",
                    "AND TRIM(UPPER(COALESCE(category, ''))) = :category_exact"
                )
            else:
                sql_base = sql_base.replace("__CATEGORY_FILTER__", "")

            # ── Domicile filter ──
            if apply_domicile_filter:
                sql_base = sql_base.replace(
                    "__DOMICILE_FILTER__",
                    _domicile_sql_filter(home_state=home_state, row_state=state)
                )
            else:
                sql_base = sql_base.replace("__DOMICILE_FILTER__", "")

            # ── College type: exact match (user said "government" / "AIIMS" / "Deemed" etc.) ──
            resolved_college_types: List[str] = []
            if isinstance(college_type_filters, list) and college_type_filters:
                resolved_college_types = [str(v).strip() for v in college_type_filters if str(v).strip()]
            elif college_type_filter:
                resolved_college_types = [str(college_type_filter).strip()]

            if resolved_college_types:
                if len(resolved_college_types) == 1:
                    sql_base = sql_base.replace(
                        "__COLLEGE_TYPE_FILTER__",
                        "AND TRIM(college_type) = :college_type_filter_0"
                    )
                    params["college_type_filter_0"] = resolved_college_types[0]
                else:
                    type_clauses = " OR ".join(
                        f"TRIM(college_type) = :college_type_filter_{i}"
                        for i in range(len(resolved_college_types))
                    )
                    sql_base = sql_base.replace(
                        "__COLLEGE_TYPE_FILTER__",
                        f"AND ({type_clauses})"
                    )
                    for i, ctype in enumerate(resolved_college_types):
                        params[f"college_type_filter_{i}"] = ctype
            else:
                sql_base = sql_base.replace("__COLLEGE_TYPE_FILTER__", "")

            # ── Course: exact match (MBBS / BDS / B.Sc. Nursing) ──
            if course_filter:
                sql_base = sql_base.replace(
                    "__COURSE_FILTER__",
                    "AND TRIM(UPPER(COALESCE(course, ''))) = :course_filter"
                )
                params["course_filter"] = course_filter.strip().upper()
            else:
                sql_base = sql_base.replace("__COURSE_FILTER__", "")

            # ── Quota: ILIKE fuzzy OR across keywords ──
            if quota_keywords:
                quota_clauses = " OR ".join(
                    f"quota ILIKE :quota_kw_{i}"
                    for i in range(len(quota_keywords))
                )
                sql_base = sql_base.replace("__QUOTA_FILTER__", f"AND ({quota_clauses})")
                for i, kw in enumerate(quota_keywords):
                    params[f"quota_kw_{i}"] = f"%{kw}%"
            else:
                sql_base = sql_base.replace("__QUOTA_FILTER__", "")

            # ── Seat type: ILIKE fuzzy OR across keywords ──
            if seat_type_keywords:
                seat_clauses = " OR ".join(
                    f"seat_type ILIKE :seat_kw_{i}"
                    for i in range(len(seat_type_keywords))
                )
                sql_base = sql_base.replace("__SEAT_TYPE_FILTER__", f"AND ({seat_clauses})")
                for i, kw in enumerate(seat_type_keywords):
                    params[f"seat_kw_{i}"] = f"%{kw}%"
            else:
                sql_base = sql_base.replace("__SEAT_TYPE_FILTER__", "")

            # ── Sub-category: exact match (optional, from user profile) ──
            if sub_category:
                sql_base = sql_base.replace(
                    "__SUB_CATEGORY_FILTER__",
                    "AND TRIM(UPPER(COALESCE(sub_category, ''))) = :sub_category_exact"
                )
                params["sub_category_exact"] = str(sub_category).strip().upper()
            else:
                sql_base = sql_base.replace("__SUB_CATEGORY_FILTER__", "")

            sql_text_compact = _compact_sql(sql_base)
            if metric_type == "score":
                params["user_score"] = metric_value
            else:
                params["user_rank"] = metric_value

            _log(f"[CUTOFF_SQL] Query (state={state}): {sql_text_compact}")
            _log(f"[CUTOFF_SQL] Rendered: {_render_sql_for_debug(sql_text_compact, params)}")

            result = await db.execute(text(sql_base), params)
            rows = [dict(row) for row in result.mappings().all()]
            _log(f"[CUTOFF_SQL] Rows fetched | state={state} count={len(rows)}")
            all_rows.extend(rows)

    # Sort by closest to user's metric value
    if metric_type == "score":
        all_rows.sort(
            key=lambda r: (
                abs(metric_value - float(r.get("score") or 0)),
                -(float(r.get("score") or 0)),
            )
        )
    else:
        all_rows.sort(
            key=lambda r: (
                abs(int(r.get("air_rank") or 0) - metric_value),
                int(r.get("air_rank") or 10**9),
            )
        )

    # Deduplicate by state + institution
    deduped: List[Dict] = []
    seen: set = set()
    for row in all_rows:
        key = (row.get("state"), row.get("institution_name") or row.get("college_name"))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)
        if len(deduped) >= total_limit:
            break

    _log(
        "[CUTOFF_SQL] Completed | "
        f"raw_rows={len(all_rows)} deduped={len(deduped)} returned={min(len(deduped), total_limit)}"
    )
    return deduped


def format_cutoff_markdown(
    *,
    rows: List[Dict],
    metric_type: str,
    metric_value: int,
    category: str,
    sub_category: Optional[str],
    home_state: str,
    target_states: List[str],
    central_mode: bool = False,
    display_limit: int = 10,
) -> str:
    if not rows:
        metric_label = "AIR Rank" if metric_type == "rank" else "Score"
        states_label = ", ".join(target_states) if target_states else "your selected states"
        has_sub_category = bool(str(sub_category or "").strip())
        profile_lines = [
            "### Profile I Used",
            "",
            f"- {metric_label}: **{metric_value}**",
        ]
        if not central_mode:
            profile_lines.append(f"- Category: **{category}**")
        if has_sub_category:
            profile_lines.append(f"- Sub-category: **{str(sub_category).strip()}**")
        if central_mode:
            profile_lines.append(f"- Counselling scope: **{states_label} (All India / MCC)**")
        else:
            domicile_mode = (
                "NON-DOMICILE or OPEN"
                if any(s.lower() != (home_state or "").lower() for s in target_states)
                else "DOMICILE or OPEN"
            )
            profile_lines.extend(
                [
                    f"- Home state: **{home_state}**",
                    f"- Target state(s): **{states_label}**",
                    f"- Eligibility mode: **{domicile_mode}**",
                ]
            )
        profile_block = "\n".join(profile_lines)

        return (
            "I checked the 2025 cutoff records carefully, but I could not find a direct match "
            "for your current preference set.\n\n"
            f"{profile_block}\n\n"
            "### What This Means\n\n"
            "This usually happens when the selected combination is too strict for the available rows "
            "(for example: specific quota + college type + domicile mode at this rank/score range).\n\n"
            "### I Can Help You Explore Better Options\n\n"
            "If you want, I can immediately retry with one of these refinements:\n"
            "1. Keep same state, but relax quota or college type\n"
            "2. Keep same category, but include nearby state options\n"
            "3. Show closest available colleges in the same rank/score range\n\n"
            "Reply with what you prefer, and I will refine it for you."
        )

    shown = rows[:display_limit]
    metric_label = "Score" if metric_type == "score" else "AIR Rank"
    states_label = ", ".join(target_states)
    has_sub_category = bool(str(sub_category or "").strip())

    lines = [
        "### Best-Match Colleges (Based on 2025 Cutoff Data)",
        "",
        f"- Showing top **{len(shown)}** of **{len(rows)}** matched options",
        "",
        "| # | Institution | State | Category | Quota | Domicile | AIR | Score | Round |",
        "|---|---|---|---|---|---|---:|---:|---|",
    ]

    if central_mode:
        lines.insert(2, f"- Profile used: **{metric_label} = {metric_value}**")
        lines.insert(3, f"- Counselling scope: **All India (MCC)**")
    else:
        lines.insert(2, f"- Profile used: **{metric_label} = {metric_value}**, **Category = {category}**")
        lines.insert(3, f"- Home state: **{home_state}**")
        lines.insert(4, f"- Target states: **{states_label}**")

    if has_sub_category and not central_mode:
        lines.insert(5, f"- Sub-category: **{str(sub_category).strip()}**")

    for idx, row in enumerate(shown, start=1):
        lines.append(
            "| {idx} | {inst} | {state} | {cat} | {quota} | {dom} | {air} | {score} | {round_} |".format(
                idx=idx,
                inst=(
                    f"{(row.get('institution_name') or row.get('college_name') or '-').replace('|', ' ')} "
                    f"({str(row.get('course') or '-').replace('|', ' ')})"
                ),
                state=row.get("state") or "-",
                cat=row.get("category") or "-",
                quota=row.get("quota") or "-",
                dom=row.get("domicile") or "-",
                air=row.get("air_rank") if row.get("air_rank") is not None else "-",
                score=_format_score_for_output(row.get("score")),
                round_=row.get("round") or "-",
            )
        )

    return "\n".join(lines)