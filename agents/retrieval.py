from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Tuple
import pandas as pd

# -----------------------------
# Config 
# -----------------------------
@dataclass
class RetrievalConfig:
    tz: str = "Australia/Sydney"
    source_tz: str = "Australia/Sydney"
    previous_years: int = 5
    time_tolerance_minutes: int = 30
    sameweek_days: int = 3
    recent_hours_map: Dict[str, int] = field(default_factory=lambda: {
        "15min": 72, "30min": 96, "1H": 168, "1D": 24*365,
    })
    anchor_mode: str = "latest_by_region"   # "latest_by_region" | "filters"
    safe_default_recent_days: int = 7

    # NEW knobs for short/long analogs (used by ablations & harness)
    same_hour_back_days: int = 7                    # N previous days for short-term same-hour analogs
    include_same_hour_previous_days: bool = True
    include_same_woy_prior_years: bool = True

# -----------------------------
# Timezone utilities
# -----------------------------
def _to_target_tz(series: pd.Series, source_tz: str, target_tz: str) -> pd.Series:
    s = pd.to_datetime(series, errors="coerce", utc=False)
    if getattr(s.dt, "tz", None) is None:
        s = s.dt.tz_localize(source_tz, ambiguous="NaT", nonexistent="shift_forward")
    return s.dt.tz_convert(target_tz)

# -----------------------------
# Region & time helpers
# -----------------------------
def _apply_region_filter(df: pd.DataFrame, regions: List[str]) -> pd.DataFrame:
    if not regions or "REGION" not in df.columns:
        return df
    want = {str(r).upper() for r in regions}
    return df[df["REGION"].astype(str).str.upper().isin(want)]

def _normalize_times(times: List[str]) -> List[Tuple[int,int,int]]:
    out = []
    for t in (times or []):
        t = str(t)
        if len(t) == 5:
            t += ":00"
        try:
            hh, mm, ss = [int(x) for x in t.split(":")]
            out.append((hh, mm, ss))
        except Exception:
            pass
    return out

# -----------------------------
# Filter executor 
# -----------------------------
def filter_data(
    df: pd.DataFrame,
    f: Dict[str, Any],
    *,
    cfg: RetrievalConfig,
) -> pd.DataFrame:
    """
    Supports region filter, date_start/date_end (inclusive end-of-day if time not provided),
    components years/months/days, and exact times with +/- tolerance minutes.
    """
    if df is None or df.empty:
        return df.iloc[0:0]

    work = _apply_region_filter(df, f.get("regions") or [])
    if work.empty:
        return work

    ts = _to_target_tz(work["SETTLEMENTDATE"], cfg.source_tz, cfg.tz)
    mask = pd.Series(True, index=work.index)

    ds_raw, de_raw = f.get("date_start"), f.get("date_end")
    ds = pd.to_datetime(ds_raw, errors="coerce") if ds_raw is not None else None
    de = pd.to_datetime(de_raw, errors="coerce") if de_raw is not None else None

    centers = _normalize_times(f.get("times") or [])
    have_time = bool(centers)

    if ds is not None:
        if have_time:
            hh, mm, ss = centers[0]
            ds_local = pd.Timestamp(f"{ds.date()} {hh:02d}:{mm:02d}:{ss:02d}", tz=cfg.tz)
        else:
            ds_local = pd.Timestamp(ds.date(), tz=cfg.tz)
        mask &= ts >= ds_local

    if de is not None:
        if have_time:
            hh, mm, ss = centers[0]
            de_local = pd.Timestamp(f"{de.date()} {hh:02d}:{mm:02d}:{ss:02d}", tz=cfg.tz)
            mask &= ts <= de_local
        else:
            # inclusive end-of-day
            de_local = pd.Timestamp(de.date(), tz=cfg.tz) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
            mask &= ts <= de_local

    years  = [int(x) for x in (f.get("years")  or []) if str(x).isdigit()]
    months = [int(x) for x in (f.get("months") or []) if str(x).isdigit()]
    days   = [int(x) for x in (f.get("days")   or []) if str(x).isdigit()]

    if years:  mask &= ts.dt.year.isin(years)
    if months: mask &= ts.dt.month.isin(months)
    if days:   mask &= ts.dt.day.isin(days)

    out = work.loc[mask]
    if out.empty or not centers:
        return out

    # exact time tolerance
    tol = pd.Timedelta(minutes=cfg.time_tolerance_minutes)
    ts2 = _to_target_tz(out["SETTLEMENTDATE"], cfg.source_tz, cfg.tz)
    base = ts2.dt.normalize()
    keep = pd.Series(False, index=out.index)
    for (hh, mm, ss) in centers:
        anchor = base + pd.to_timedelta(hh, unit="h") + pd.to_timedelta(mm, unit="m") + pd.to_timedelta(ss, unit="s")
        keep |= (ts2 - anchor).abs() <= tol
    return out.loc[keep]

# -----------------------------
# Window helpers
# -----------------------------
def _freq_to_recent_span(freq: Optional[str], cfg: RetrievalConfig) -> pd.Timedelta:
    if not freq:
        return pd.Timedelta(days=cfg.safe_default_recent_days)
    f = str(freq).strip().lower()
    hours = cfg.recent_hours_map.get(f)
    if hours is None:
        return pd.Timedelta(days=cfg.safe_default_recent_days)
    return pd.Timedelta(hours=hours)

def _resolve_target_window(filters: Dict[str, Any], tz: str) -> Tuple[Optional[pd.Timestamp], Optional[pd.Timestamp], Optional[str]]:
    """Prefer 'start/end'; else (date_start/date_end + times); else Y/M/D."""
    def _ensure_ts(x) -> Optional[pd.Timestamp]:
        if x is None or x == "":
            return None
        try:
            t = pd.Timestamp(x)
            return t.tz_localize(tz) if t.tzinfo is None else t.tz_convert(tz)
        except Exception:
            return None

    start = _ensure_ts(filters.get("start"))
    end   = _ensure_ts(filters.get("end"))
    freq  = (filters.get("freq") or None)

    if start is not None or end is not None:
        return start, end, freq

    ds, de = filters.get("date_start"), filters.get("date_end")
    times = filters.get("times") or []
    tstr = str(times[0]) if times else "00:00:00"

    if ds and de:
        return _ensure_ts(f"{ds} {tstr}"), _ensure_ts(f"{de} {tstr}"), freq
    if ds:
        return _ensure_ts(f"{ds} {tstr}"), None, freq
    if de:
        return None, _ensure_ts(f"{de} {tstr}"), freq

    years = filters.get("years") or []
    months = filters.get("months") or []
    days = filters.get("days") or []
    if years and months and days:
        y, m, d = int(years[0]), int(months[0]), int(days[0])
        return _ensure_ts(f"{y:04d}-{m:02d}-{d:02d} {tstr}"), None, freq

    return None, None, freq

# -----------------------------
# Anchoring
# -----------------------------
def _latest_anchor_in_regions(
    df: pd.DataFrame, regions: List[str], cfg: RetrievalConfig
) -> Optional[pd.Timestamp]:
    work = _apply_region_filter(df, regions)
    if work.empty:
        return None
    ts = _to_target_tz(work["SETTLEMENTDATE"], cfg.source_tz, cfg.tz)
    mx = ts.max()
    return None if pd.isna(mx) else mx

# -----------------------------
# Prior-year slicing
# -----------------------------
def _slice_same_date_prior_years(
    df: pd.DataFrame,
    start: pd.Timestamp,
    end: Optional[pd.Timestamp],
    years_back: int,
    cfg: RetrievalConfig
) -> pd.DataFrame:
    """Same date/time window (duration preserved) in each of the previous N years."""
    parts = []
    duration = (end - start) if end is not None else None
    for i in range(1, years_back + 1):
        tgt_year = start.year - i
        try:
            s_y = start.replace(year=tgt_year)
        except Exception:
            s_y = start.replace(year=tgt_year, day=min(28, start.day))  # Feb 29 guard
        if duration is not None:
            e_y = s_y + duration
            f = {"date_start": s_y, "date_end": e_y, "regions": []}
        else:
            f = {"date_start": s_y - pd.Timedelta(minutes=1), "date_end": s_y + pd.Timedelta(minutes=1), "regions": []}
        part = filter_data(df, f, cfg=cfg)
        if not part.empty:
            part = part.copy()
            part["PRIOR_YEAR"] = tgt_year
            parts.append(part)
    return pd.concat(parts, ignore_index=True) if parts else df.iloc[0:0]

def _slice_same_week_prior_years(
    df: pd.DataFrame,
    anchor: pd.Timestamp,
    years_back: int,
    cfg: RetrievalConfig,
    duration: Optional[pd.Timedelta] = None,
    keep_exact_time: bool = False,
    times: Optional[List[str]] = None,
) -> pd.DataFrame:
    """ISO-week aligned approximation: +/- sameweek_days around anchor day in prior years."""
    parts = []
    for i in range(1, years_back + 1):
        y = anchor.year - i
        # clamp day to 28 to avoid Feb issues
        try:
            base = anchor.replace(year=y, day=min(anchor.day, 28))
        except Exception:
            base = anchor - pd.DateOffset(years=i)
        start = base - pd.Timedelta(days=cfg.sameweek_days)
        end   = base + (duration if duration is not None else pd.Timedelta(days=cfg.sameweek_days))
        f = {"date_start": start, "date_end": end, "regions": []}
        if keep_exact_time and times:
            f["times"] = [times[0]]
        part = filter_data(df, f, cfg=cfg)
        if not part.empty:
            parts.append(part)
    return pd.concat(parts, ignore_index=True) if parts else df.iloc[0:0]

# -----------------------------
# NEW: Short-term analogs — same hour on previous N days
# -----------------------------
def _slice_same_hour_previous_days(
    df: pd.DataFrame,
    anchor: pd.Timestamp,
    days_back: int,
    cfg: RetrievalConfig,
    regions: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    For each of the last `days_back` days relative to anchor, select rows within
    ± time_tolerance_minutes around the anchor's time-of-day.
    """
    work = _apply_region_filter(df, regions or [])
    if work.empty:
        return work.iloc[0:0]

    ts = _to_target_tz(work["SETTLEMENTDATE"], cfg.source_tz, cfg.tz)
    work = work.assign(__ts=ts).dropna(subset=["__ts"])
    anchor_local = anchor.tz_convert(cfg.tz) if anchor.tzinfo else anchor.tz_localize(cfg.tz)
    tol = pd.Timedelta(minutes=cfg.time_tolerance_minutes)
    tod = anchor_local - anchor_local.normalize()

    parts = []
    for d in range(1, days_back + 1):
        day_local = (anchor_local - pd.Timedelta(days=d)).normalize()
        target = day_local + tod
        pick = work[(work["__ts"] >= (target - tol)) & (work["__ts"] <= (target + tol))].drop(columns="__ts")
        if not pick.empty:
            pick = pick.copy()
            pick["LAG_DAYS"] = d
            parts.append(pick)
    return pd.concat(parts, ignore_index=True) if parts else work.iloc[0:0]

# -----------------------------
# NEW: Long-term analogs — same week-of-year across prior years (time-anchored window)
# -----------------------------
def _slice_same_woy_prior_years(
    df: pd.DataFrame,
    anchor: pd.Timestamp,
    years_back: int,
    cfg: RetrievalConfig,
    regions: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Same ISO week-of-year window in prior years, preserving time-of-day with
    ± time_tolerance_minutes tolerance.
    """
    work = _apply_region_filter(df, regions or [])
    if work.empty:
        return work.iloc[0:0]

    ts = _to_target_tz(work["SETTLEMENTDATE"], cfg.source_tz, cfg.tz)
    work = work.assign(__ts=ts).dropna(subset=["__ts"])

    tol = pd.Timedelta(minutes=cfg.time_tolerance_minutes)
    anchor_local = anchor.tz_convert(cfg.tz) if anchor.tzinfo else anchor.tz_localize(cfg.tz)
    tod = anchor_local - anchor_local.normalize()

    parts = []
    for i in range(1, years_back + 1):
        try:
            base = anchor_local.replace(year=anchor_local.year - i, day=min(anchor_local.day, 28))
        except Exception:
            base = anchor_local - pd.DateOffset(years=i)
        start = (base - pd.Timedelta(days=cfg.sameweek_days)).normalize() + tod - tol
        end   = (base + pd.Timedelta(days=cfg.sameweek_days)).normalize() + tod + tol
        pick = work[(work["__ts"] >= start) & (work["__ts"] <= end)].drop(columns="__ts")
        if not pick.empty:
            pick = pick.copy()
            # ✅ pandas weekofyear is deprecated; use ISO calendar:
            pick["PRIOR_WOY"] = int(base.isocalendar().week)
            parts.append(pick)
    return pd.concat(parts, ignore_index=True) if parts else work.iloc[0:0]

# -----------------------------
# Public API
# -----------------------------
def retrieve_context(
    df: pd.DataFrame,
    filters: Dict[str, Any],
    route: str,                        # "short_term" | "long_term"
    cfg: RetrievalConfig = RetrievalConfig(),
) -> Dict[str, Any]:
    """
    SHORT_TERM:
      1) Recent window prior to target (span from freq; fallback 7d)
      2) Same required date/time window in previous years
      3) Same-week prior years at exact time (if time given)
      4) NEW: same-hour previous N days (if enabled)

    LONG_TERM:
      1) Same required date/time window in previous years
      2) Same-week prior years (if enabled)
      3) NEW: (also returns same_hour_previous_days empty unless you enable it explicitly)
    """
    assert "SETTLEMENTDATE" in df.columns, "df must contain SETTLEMENTDATE"

    regions = filters.get("regions") or []
    work = _apply_region_filter(df, regions) if regions else df.copy()

    # resolve target window from filters
    f_start, f_end, freq = _resolve_target_window(filters, cfg.tz)

    # anchor selection
    if cfg.anchor_mode == "latest_by_region":
        anchor = _latest_anchor_in_regions(work, regions, cfg)
    else:
        anchor = f_end or f_start
    if anchor is None:
        anchor = _latest_anchor_in_regions(work, regions, cfg)

    recent_span = _freq_to_recent_span(freq, cfg)

    # init empty slices
    recent_window             = work.iloc[0:0]
    prior_years_same_dates    = work.iloc[0:0]
    prior_years_same_week     = work.iloc[0:0]
    same_hour_previous_days   = work.iloc[0:0]
    same_woy_prior_years      = work.iloc[0:0]

    # target window for prior-year replication
    target_start = f_start if f_start is not None else anchor
    target_end   = f_end
    times = filters.get("times") or []
    keep_exact_time = bool(times)

    # short-term: recent window + same-hour analogs
    if route == "short_term" and anchor is not None:
        r_start = anchor - recent_span
        recent_window = filter_data(work, {"date_start": r_start, "date_end": anchor, "regions": regions}, cfg=cfg)

        if cfg.include_same_hour_previous_days:
            same_hour_previous_days = _slice_same_hour_previous_days(
                work, anchor, cfg.same_hour_back_days, cfg, regions
            )

    # prior years: same dates
    if target_start is not None:
        prior_years_same_dates = _slice_same_date_prior_years(work, target_start, target_end, cfg.previous_years, cfg)
        if target_end is None and keep_exact_time and not prior_years_same_dates.empty:
            prior_years_same_dates = filter_data(prior_years_same_dates, {"times": times}, cfg=cfg)

    # prior years: same week
    if anchor is not None:
        duration = (target_end - target_start) if (target_start is not None and target_end is not None) else None
        prior_years_same_week = _slice_same_week_prior_years(
            work, anchor, cfg.previous_years, cfg, duration=duration,
            keep_exact_time=keep_exact_time, times=times
        )
        if cfg.include_same_woy_prior_years:
            same_woy_prior_years = _slice_same_woy_prior_years(work, anchor, cfg.previous_years, cfg, regions)

    meta = {
        "route": route,
        "target_start_iso": target_start.isoformat() if target_start is not None else None,
        "target_end_iso": target_end.isoformat() if target_end is not None else None,
        "anchor_iso": anchor.isoformat() if anchor is not None else None,
        "freq": freq,
        "recent_span": str(recent_span) if (route == "short_term" and anchor is not None) else None,
        "previous_years": cfg.previous_years,
        "tz": cfg.tz,
        "counts": {
            "recent_window": int(len(recent_window)),
            "prior_years_same_dates": int(len(prior_years_same_dates)),
            "prior_years_same_week": int(len(prior_years_same_week)),
            "same_hour_previous_days": int(len(same_hour_previous_days)),
            "same_woy_prior_years": int(len(same_woy_prior_years)),
        },
        "notes": [],
    }

    return {
        "recent_window": recent_window,
        "prior_years_same_dates": prior_years_same_dates,
        "prior_years_same_week": prior_years_same_week,
        "same_hour_previous_days": same_hour_previous_days,
        "same_woy_prior_years": same_woy_prior_years,
        "meta": meta,
    }