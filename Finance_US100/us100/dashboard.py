"""Widok panelu analizy technicznej dla jednego instrumentu (US100, złoto)."""

from __future__ import annotations

from datetime import datetime
from http.client import HTTPException

import pandas as pd
import streamlit as st

from . import advisor, archive, backtest, chart_annotations, contracts, data, delivery, engine, families, family_alerts, regime, gold_spot, harmonics, indicators as ind, journal, notify, paper, plan, price_structures, risk, signals, storage
from .instruments import ALL, GOLD, Instrument


@st.cache_data(ttl=300, show_spinner="Pobieram notowania…")
def get_frames(ticker: str) -> dict[str, pd.DataFrame]:
    return engine.load_frames(ticker)


@st.cache_data(ttl=300, show_spinner="Pobieram świece do wykresu…")
def get_intraday_chart(ticker: str, interval: str) -> pd.DataFrame:
    observed = pd.Timestamp.now(tz="UTC")
    frame = ind.add_indicators(data.load_intraday(ticker, interval))
    frame.attrs["pobrano"] = observed
    return frame


@st.cache_data(ttl=300, show_spinner=False)
def get_quote(ticker: str) -> tuple[float, float]:
    return data.last_value(ticker)


@st.cache_data(max_entries=8, show_spinner=False)
def get_monitor_frames(ticker: str, snapshot: dict) -> dict[str, pd.DataFrame]:
    return engine.load_monitor_frames(ticker, snapshot)


def _num(x: float, dec: int = 0) -> str:
    return f"{x:,.{dec}f}".replace(",", " ")


@st.cache_data(ttl=gold_spot.REFRESH_SECONDS, show_spinner=False)
def get_gold_spot() -> tuple[float, datetime]:
    return gold_spot.load_quote()


@st.fragment(run_every=gold_spot.REFRESH_SECONDS)
def render_gold_spot() -> None:
    try:
        price, updated = get_gold_spot()
    except (OSError, HTTPException, ValueError, TypeError, KeyError):
        st.info("Cena z Gold API jest chwilowo niedostępna.")
    else:
        age = pd.Timestamp.now(tz="UTC") - updated
        stale = not pd.Timedelta(0) <= age <= gold_spot.MAX_AGE
        local_updated = pd.Timestamp(updated).tz_convert(data.TZ_LOCAL)
        st.metric("Złoto spot XAU/USD", f"{_num(price, 2)} USD/oz")
        st.caption(f"Gold API · aktualizacja {local_updated:%d.%m.%Y %H:%M:%S} (czas polski)")
        if stale:
            st.warning("Cena spot jest nieaktualna. Porównanie będzie dostępne po odświeżeniu.")
        with st.form("gold_spot_comparison"):
            bid = st.number_input(
                "Aktualna cena GOLD BID z xStation (USD/oz)", min_value=0.01,
                value=None, step=0.01, format="%.2f", key="gold_spot_bid",
                help="Wpisz bieżącą cenę sprzedaży z xStation i od razu zapisz pomiar.",
            )
            submitted = st.form_submit_button("Zapisz porównanie", disabled=stale)
        if submitted:
            try:
                gold_spot.save_comparison(bid or 0.0, price, updated, pd.Timestamp.now(tz="UTC").to_pydatetime())
            except ValueError as exc:
                st.warning(str(exc))
            except OSError:
                st.error("Nie udało się zapisać porównania na dysku.")
            else:
                st.success("Zapisano porównanie.")

    rows = gold_spot.load_comparisons()
    if rows:
        history = pd.DataFrame(reversed(rows))
        history["Różnica BID − spot"] = history["xstation_bid"] - history["spot"]
        for column in ("pomiar", "aktualizacja_spot"):
            history[column] = pd.to_datetime(history[column], utc=True).dt.tz_convert(data.TZ_LOCAL).dt.strftime("%d.%m.%Y %H:%M:%S")
        history = history.rename(columns={
            "pomiar": "Czas pomiaru", "aktualizacja_spot": "Aktualizacja spot",
            "xstation_bid": "BID xStation", "spot": "XAU/USD spot",
        })
        st.caption("Historia porównań · ceny i różnice w USD/oz · czas polski")
        st.dataframe(history, hide_index=True, alt="Porównania cen złota spot i GOLD BID z xStation", column_config={
            column: st.column_config.NumberColumn(format="%.2f")
            for column in ("BID xStation", "XAU/USD spot", "Różnica BID − spot")
        })


def _table(rows: list[dict], kind: str) -> pd.DataFrame:
    """Tabela do edytora - z kolumnami także wtedy, gdy lista jest pusta."""
    return pd.DataFrame(rows, columns=storage.COLUMNS[kind])


def build_chart(
    df: pd.DataFrame, interval: str, n: int, off: float, name: str, annotations: dict, extent: list[float],
    entry: dict | None = None, *, show_volume: bool = True,
) -> dict:
    view = df.iloc[-n:].copy()
    for col in ("open", "high", "low", "close"):
        view[col] = view[col] + off
    fmt = "%Y-%m-%d" if interval == "D1" else "%m-%d %H:%M"
    view["czas"] = [t.strftime(fmt) for t in view.index]
    columns = ["czas", "open", "close", "low", "high"]
    volume = show_volume and "volume" in view and bool(view["volume"].gt(0).any())
    volume_groups = []
    if volume:
        position = ind.close_position(view)
        quartile = ind.CLOSE_QUARTILE
        top = position.ge(1 - quartile)
        bottom = position.le(quartile)
        volume_groups = [
            ("Zamknięcie wysoko", "#26a69a", top),
            ("Zamknięcie nisko", "#ef5350", bottom),
            ("Zamknięcie pośrodku", "#6888bb", ~(top | bottom)),
        ]
        for label, _, condition in volume_groups:
            view[label] = view["volume"].where(condition)
            columns.append(label)
    src = view[columns].round(2).reset_index(drop=True)
    src.attrs.clear()

    lo, hi = float(view["low"].min()), float(view["high"].max())
    price_range = {}
    if extent:
        bottom, top = min(lo, *extent), max(hi, *extent)
        pad = max((top - bottom) * 0.035, 0.01)
        price_range = {"min": bottom - pad, "max": top + pad}
    entry_series = []
    if entry is not None:
        index, price = entry["value"]
        point = {**entry, "value": [src["czas"].iat[index], price]}
        entry_series.append({
            "id": chart_annotations.ENTRY_SERIES_ID, "name": "Wejście Doradcy", "type": "scatter",
            "data": [point], "encode": {"x": 0, "y": 1}, "selectedMode": "single", "z": 10,
            "symbol": "arrow", "symbolSize": 24,
            "tooltip": {"trigger": "item", "formatter": "{b}"},
            "select": {"itemStyle": {"borderWidth": 2, "borderColor": "#f4bc43"}},
        })

    grids = [{"left": 42, "right": 70, "top": 28, "bottom": "26%" if volume else 50}]
    y_axes = [{"scale": True, "gridIndex": 0, "position": "right",
               "splitLine": {"lineStyle": {"opacity": 0.3}}, **price_range}]
    volume_series = []
    if volume:
        grids.append({"left": 42, "right": 70, "top": "79%", "bottom": 48})
        y_axes.append({"gridIndex": 1, "min": 0, "position": "right", "splitNumber": 2,
                       "name": "Wolumen", "splitLine": {"show": False}})
        volume_series = [{
            "type": "bar", "name": label, "stack": "wolumen",
            "encode": {"x": "czas", "y": label, "tooltip": [label]},
            "xAxisIndex": 1, "yAxisIndex": 1, "itemStyle": {"color": color},
        } for label, color, _ in volume_groups]
    axes = list(range(len(grids)))
    return {
        "animation": False,
        "tooltip": {"trigger": "axis", "axisPointer": {"type": "cross"}},
        "axisPointer": {"link": [{"xAxisIndex": "all"}]},
        "dataset": {"source": src},
        "grid": grids,
        "xAxis": [{"type": "category", "gridIndex": g, "axisLabel": {"show": g == axes[-1]}, "boundaryGap": True} for g in axes],
        "yAxis": y_axes,
        "dataZoom": [
            {"type": "inside", "xAxisIndex": axes, "start": 0, "end": 100},
            {"type": "slider", "xAxisIndex": axes, "bottom": 0, "height": 18},
        ],
        "series": [
            {
                "type": "candlestick", "name": name,
                "encode": {"x": "czas", "y": ["open", "close", "low", "high"]},
                "itemStyle": {"color": "#26a69a", "color0": "#ef5350", "borderColor": "#26a69a", "borderColor0": "#ef5350"},
                **annotations,
            },
            *volume_series,
            *entry_series,
        ],
    }


def render_market_chart(
    inst: Instrument, chart_frame: pd.DataFrame, interval: str, bars: int, off: float,
    levels: list[tuple[str, float]], frames: dict[str, pd.DataFrame], now: pd.Timestamp,
    proposal: advisor.Proposal | None, *, archived: bool = False,
) -> None:
    chart, legend = st.columns([5, 1], gap="medium")
    use = "archived" if archived else "live"
    with legend:
        legend_box = st.container(border=True)
    with legend_box:
        details = st.container()
        with st.expander("Warstwy wykresu"):
            layers = {
                layer for layer, label in chart_annotations.LAYERS.items()
                if st.checkbox(label, value=layer in chart_annotations.DEFAULT_LAYERS,
                               key=f"chart_{use}_{inst.key}_{layer}")
            }
    chart_key = f"market_chart_{use}_{inst.key}"
    selected = False
    if "entry" in layers and proposal is not None:
        chart_key += f"_{int(proposal.at.timestamp())}_{proposal.level.direction}"
        state = st.session_state.get(chart_key, {})
        selected = any(
            point["series_id"] == chart_annotations.ENTRY_SERIES_ID and 0 in point["data_indices"]
            for point in state.get("selection", {}).get("selected", [])
        )
    view = chart_frame.iloc[-bars:]
    patterns = chart_annotations.recent_patterns(chart_frame, interval, now, lookback=bars) if "harmonics" in layers else []
    context = chart_frame.tail(bars + 60)
    pof = price_structures.last_wave(context, interval, now) if "pof" in layers else None
    zones = tuple(price_structures.polarity_zones(context, interval, now)) if "zzb" in layers else ()
    annotations, extent, entry = chart_annotations.build(
        view, interval, off, layers, levels, frames, now, patterns, proposal, show_exit_levels=selected,
        pof=pof, zones=zones,
    )
    with details:
        if "entry" in layers:
            st.markdown("**Doradca · H1**")
            if proposal is not None:
                label = "Archiwalny sygnał" if archived else "Propozycja"
                st.caption(
                    f"{label} **{proposal.level.direction.upper()}** · {_num(proposal.entry + off, inst.decimals)}"
                )
                if entry is None:
                    st.caption("Sygnał poza zakresem wykresu.")
                elif selected:
                    st.caption(f"SL {_num(proposal.stop + off, inst.decimals)} · TP {_num(proposal.target + off, inst.decimals)}")
                else:
                    st.caption("Kliknij strzałkę, aby pokazać SL i TP.")
            else:
                st.markdown("**CZEKAJ**")
                st.caption("Brak potwierdzonego wejścia.")
        if "harmonics" in layers:
            st.markdown(f"**Układ AB=CD · {interval}**")
            if patterns:
                projection, status = patterns[-1]
                states = {"building": "W budowie", "reached": "Cena dotarła do strefy D",
                          "rejected": "Reakcja w strefie D", "invalid_c": "Układ zanegowany", "invalid_d": "Układ zanegowany"}
                st.caption(states[status.stage])
                if status.stage in ("building", "reached"):
                    low, high = projection.zone
                    direction = projection.reaction_direction
                    move = "w górę" if direction == "long" else "w dół"
                    st.caption(f"Strefa D {_num(low + off, inst.decimals)}–{_num(high + off, inst.decimals)}. "
                               f"Odbicie {move} może wskazać {direction.upper()}.")
                if status.stage == "rejected":
                    st.caption(f"Wykryta reakcja {projection.reaction_direction.upper()}.")
            else:
                st.caption("Brak układu w ostatnich świecach.")
        if "pof" in layers:
            st.markdown(f"**POF · {interval}**")
            if pof is None:
                st.caption("Czekam na potwierdzenie ostatniej fali.")
            else:
                levels_by_ratio = dict(pof.levels())
                st.caption(f"50% · {_num(levels_by_ratio[0.5] + off, inst.decimals)}  "
                           f"\n61,8% · {_num(levels_by_ratio[0.618] + off, inst.decimals)}")
        if "zzb" in layers:
            st.markdown(f"**ZZB · {interval}**")
            if not zones:
                st.caption("Brak aktywnej zmiany roli poziomu.")
            for zone in zones:
                low, high = zone.bounds
                status = "Po reteście" if zone.retested_at is not None else "Czeka na retest"
                st.caption(f"{zone.role} {_num(low + off, inst.decimals)}–{_num(high + off, inst.decimals)} · {status}")
                if pof is not None:
                    matches = [f"{ratio * 100:g}%".replace(".", ",") for ratio, price in pof.levels() if low <= price <= high]
                    if matches:
                        st.caption(f"W tej strefie wypada POF {', '.join(matches)}.")
        if "volume" in layers:
            if "volume" in view and view["volume"].gt(0).any():
                st.caption(f"Wolumen {inst.ticker} · Yahoo Finance. Zielony i czerwony oznaczają zamknięcie "
                           "w górnej lub dolnej ćwiartce świecy. To przybliżenie kierunku presji cenowej.")
            else:
                st.caption(f"Brak wolumenu dla {interval} w pobranych danych.")
    with chart:
        st.echarts_chart(
            build_chart(chart_frame, interval, bars, off, inst.name, annotations, extent, entry,
                        show_volume="volume" in layers),
            height=620, alt=f"Świece {inst.name} {interval}, wolumen i oznaczenia sygnałów",
            key=chart_key, on_select="rerun" if entry is not None else "ignore",
        )


@st.cache_data(ttl=3600, show_spinner="Sprawdzam podobne sygnały w historii…", max_entries=64)
def holding_cached(ticker: str, direction: str, sl_atr: float, tp_atr: float, limit_h: float):
    stats, trades = backtest.holding_times(get_frames(ticker)["H1"], direction, sl_atr, tp_atr, limit_h)
    return stats, trades


def _h(x: float | None) -> str:
    return "–" if x is None else f"{x:.0f}"


def render_holding(inst: Instrument, direction: str, ev: dict, entry: float, tp: float, atr_h1: float | None, limit_h: float) -> None:
    st.markdown("**Czas trzymania – podobne sygnały z historii H1**")
    if not ev["poprawny"] or not atr_h1 or ev["sl_atr_h1"] is None:
        st.caption("Uzupełnij poprawne SL i TP, aby porównać z historią.")
        return
    tp_atr = abs(tp - entry) / atr_h1
    stats, _ = holding_cached(inst.ticker, direction, round(ev["sl_atr_h1"], 2), round(tp_atr, 2), float(limit_h))
    if not stats.wystarczy:
        st.caption(f"Za mało danych: {stats.przypadki} porównywalnych przypadków (potrzeba co najmniej {backtest.MIN_SAMPLE}).")
        return
    with st.container(horizontal=True):
        for label, n, med, p25, p75 in (
            ("Do TP: mediana", stats.tp, stats.tp_mediana_h, stats.tp_p25_h, stats.tp_p75_h),
            ("Do SL: mediana", stats.sl, stats.sl_mediana_h, stats.sl_p25_h, stats.sl_p75_h),
        ):
            if n == 0:
                st.metric(label, "brak wyjść")
            else:
                st.metric(label, f"{_h(med)} godz.", f"przedział {_h(p25)}–{_h(p75)} godz.", delta_color="off", delta_arrow="off")
    st.caption(
        f"{stats.przypadki} przypadków: TP {stats.tp} ({stats.odsetek_tp:.0%}), SL {stats.sl}, "
        f"bez wyjścia przed limitem {limit_h:g} godz.: {stats.bez_wyjscia} ({stats.odsetek_bez_wyjscia:.0%}). "
        "Przedział to 25.–75. percentyl. To statystyka z przeszłości, nie prognoza."
    )


def render_plan_history(direction: str, res: "backtest.PlanHistory", vol: float, usd_pp: float, usdpln: float | None, limit_h: float) -> None:
    st.markdown(f"**{direction.capitalize()}**")
    if res.powod and res.trades.empty:
        st.info(res.powod, icon=":material/info:")
        return
    s = res.stats()
    st.caption(f"Komplet warunków wejścia spełniony {res.sygnaly} razy; ocenionych pozycji (bez nakładania się, z pełnym wynikiem): {s.przypadki}.")
    if s.przypadki == 0:
        st.info("Żaden przypadek nie dał pełnego wyniku (np. zamknięcie było już za TP albo za SL).", icon=":material/info:")
        return
    if not s.wystarczy:
        st.warning(f"Mała próba ({s.przypadki} < {backtest.MIN_SAMPLE}) – czasy i wyniki traktuj wyłącznie orientacyjnie.", icon=":material/warning:")
    with st.container(horizontal=True):
        st.metric("TP", f"{s.tp}", f"{s.odsetek_tp:.0%}", delta_color="off", delta_arrow="off")
        st.metric("SL", f"{s.sl}", f"{s.sl / s.przypadki:.0%}", delta_color="off", delta_arrow="off")
        st.metric(f"Limit {limit_h:g} godz.", f"{s.bez_wyjscia}", f"{s.odsetek_bez_wyjscia:.0%}", delta_color="off", delta_arrow="off")
    times = []
    for label, n, med, p25, p75 in (("TP", s.tp, s.tp_mediana_h, s.tp_p25_h, s.tp_p75_h), ("SL", s.sl, s.sl_mediana_h, s.sl_p25_h, s.sl_p75_h)):
        times.append(f"do {label}: " + (f"mediana {_h(med)} godz. ({_h(p25)}–{_h(p75)})" if n else "brak"))
    st.caption("Czas: " + "; ".join(times) + ".")
    pts = res.trades["wynik [pkt]"]
    pln = (lambda x: x * vol * usd_pp * usdpln) if usdpln else None
    with st.container(horizontal=True):
        st.metric("Wynik na pozycję (mediana)", f"{pts.median():+.1f} pkt", f"{pln(pts.median()):+.2f} PLN" if pln else None, delta_color="off", delta_arrow="off")
        st.metric("Suma wszystkich", f"{pts.sum():+.1f} pkt", f"{pln(pts.sum()):+.2f} PLN" if pln else None, delta_color="off", delta_arrow="off")
    with st.expander("Przypadki"):
        st.dataframe(res.trades, hide_index=True, alt=f"Historyczne przypadki planu {direction}",
                     column_config={"wejście": st.column_config.DatetimeColumn(format="YYYY-MM-DD HH:mm")})


def render_journal_tab(inst: Instrument, cfg: dict, frames: dict[str, pd.DataFrame], off: float, usdpln: float | None) -> None:
    k, dec = inst.key, inst.decimals
    step = 0.1 if dec > 1 else 1.0
    usd_pp = float(cfg["usd_za_punkt"])
    st.caption(
        "Faktyczne transakcje: z zamkniętych planów i dodane ręcznie. „Plan” pokazuje, jak skończyłaby się pozycja przy "
        "trzymaniu się SL, TP i limitu czasu z planu; „bez limitu” – przy trzymaniu do SL lub TP (maks. 30 dni). "
        "Porównanie używa ciągłej historii zamkniętych świec H1, bez spreadu i swapów. Ekstrema świecy rozpoczętej "
        "przed wejściem są pomijane. Jeśli limit wypada wewnątrz świecy i nie ma w niej SL ani TP, cenę wyjścia "
        "szacuje się z poprzedniego zamknięcia. Wynik w PLN z xStation ma pierwszeństwo przed szacunkiem."
    )
    df = journal.entries(cfg["plany"], frames["H1"], off, usd_pp, usdpln)
    if df.empty:
        st.info("Dziennik jest pusty. Dodaj transakcję poniżej.", icon=":material/menu_book:")
    else:
        s = journal.summary(df)
        broker_rate = s["xStation: skuteczność"]
        with st.container(horizontal=True):
            st.metric("Skuteczność w punktach", f"{s['skuteczność']:.0%}",
                      f"{s['zyskowne']} z {s['transakcje']} transakcji", delta_color="off", delta_arrow="off")
            st.metric("Skuteczność wg xStation", f"{broker_rate:.0%}" if broker_rate is not None else "–",
                      f"{s['xStation: zyskowne']} z {s['xStation: n']} transakcji" if broker_rate is not None else "brak wyników",
                      delta_color="off", delta_arrow="off")
            st.metric("Suma", f"{_num(s['suma PLN'], 2)} PLN" if s["suma PLN"] is not None else "brak USD/PLN")
            st.metric("Średnie R", f"{s['średnie R']:+.2f}" if s["średnie R"] is not None else "–", help="Wynik w wielokrotnościach początkowego ryzyka (wejście–SL).")
            st.metric("Mediana czasu", f"{s['mediana godzin']:.0f} godz.")
        notes = []
        if "ręczne: n" in s:
            notes.append(
                f"Ręczne wyjścia: {s['ręczne: n']}, lepiej niż plan w {s['ręczne: lepiej niż plan']} przypadkach; "
                f"łącznie {s['ręczne: różnica do planu [pkt]']:+.1f} pkt względem trzymania się planu."
            )
        if "limit: n" in s:
            notes.append(
                f"Zamknięcia z powodu limitu czasu: {s['limit: n']}, lepiej niż trzymanie dalej w {s['limit: lepiej niż trzymanie dalej']}; "
                f"łącznie {s['limit: różnica do trzymania dalej [pkt]']:+.1f} pkt względem trzymania do SL/TP."
            )
        if "trzymane po limicie: n" in s:
            inside = s.get("w limicie: średni wynik [pkt]")
            notes.append(
                f"Trzymane dłużej niż limit: {s['trzymane po limicie: n']}, średnio {s['trzymane po limicie: średni wynik [pkt]']:+.1f} pkt"
                + (f" (w limicie: {s['w limicie: n']}, średnio {inside:+.1f} pkt)." if inside is not None else ".")
            )
        if notes:
            st.markdown("\n".join(f"- {n}" for n in notes))
        elif len(df) < 5:
            st.caption("Porównania z planem pojawią się, gdy w dzienniku będą ręczne wyjścia albo zamknięcia z powodu limitu.")
        st.dataframe(df, hide_index=True, alt=f"Dziennik transakcji {inst.name}",
                     column_config={"po limicie": st.column_config.CheckboxColumn(width="small")})
        st.download_button("Pobierz dziennik CSV", df.to_csv(index=False).encode("utf-8-sig"), file_name=f"dziennik_{k}.csv",
                           mime="text/csv", icon=":material/download:", key=f"jr_csv_{k}")

    with st.expander("Dodaj transakcję spoza planu"):
        with st.form(f"jr_add_{k}"):
            c1, c2, c3, c4 = st.columns(4)
            direction = c1.selectbox("Kierunek", ["long", "short"])
            entry = c2.number_input("Cena wejścia", min_value=0.0, step=step)
            exit_px = c3.number_input("Cena wyjścia", min_value=0.0, step=step)
            vol = c4.number_input("Wolumen [lot]", min_value=0.001, value=0.01, step=0.001, format="%.3f")
            c5, c6, c7, c8 = st.columns(4)
            now_ts = pd.Timestamp.now()
            d_in = c5.date_input("Data wejścia", value=now_ts.date())
            t_in = c6.time_input("Godzina wejścia", value=now_ts.time().replace(second=0, microsecond=0))
            d_out = c7.date_input("Data wyjścia", value=now_ts.date())
            t_out = c8.time_input("Godzina wyjścia", value=now_ts.time().replace(second=0, microsecond=0))
            c9, c10 = st.columns(2)
            reason = c9.selectbox("Powód wyjścia", journal.REASONS)
            pln_real = c10.number_input("Wynik w PLN z xStation (opcjonalnie)", value=None, step=1.0)
            note = st.text_input("Notatka")
            if st.form_submit_button("Dodaj do dziennika", icon=":material/add:"):
                opened, closed_iso = f"{d_in:%Y-%m-%d}T{t_in:%H:%M}", f"{d_out:%Y-%m-%d}T{t_out:%H:%M}"
                entry_rec = journal.new_offplan(direction, entry, exit_px, vol, opened, closed_iso, reason, pln_real, note)
                errs = (["cena wejścia musi być większa od zera"] if entry <= 0 else []) + journal.validate_close(entry_rec, exit_px, closed_iso, reason)
                if errs:
                    st.error("\n".join(f"- {e}" for e in errs), title="Nie dodano transakcji")
                else:
                    cfg["plany"].append(entry_rec)
                    storage.save_settings(cfg, k)
                    st.toast("Dodano transakcję do dziennika.", icon=":material/check:")
                    st.rerun()


def render_replay_tab(inst: Instrument, cfg: dict, frames: dict[str, pd.DataFrame]) -> None:
    k = inst.key
    st.caption(
        "Te same reguły co monitor – reguły zamknięć, dotknięcia i przeskoki poziomów (przerwa 4 godz.), strefy RSI H1/H4 – "
        "uruchomione po kolei na historycznych, zamkniętych świecach z obecnymi ustawieniami. Pozwala sprawdzić, "
        "czy alerty przychodziłyby za rzadko albo za często. Dywergencje nie są odtwarzane."
    )
    with st.container(horizontal=True, vertical_alignment="bottom"):
        days = st.select_slider("Okres", options=[7, 14, 30, 60, 90, 180], value=30, format_func=lambda d: f"{d} dni", key=f"rp_days_{k}")
        run = st.button("Odtwórz alerty", icon=":material/history:", key=f"rp_run_{k}")
    if run:
        st.session_state[f"rp_res_{k}"] = (days, backtest.replay_alerts(frames, cfg, days=days))
    res = st.session_state.get(f"rp_res_{k}")
    if not res:
        st.caption("Kliknij „Odtwórz alerty”.")
        return
    days_run, df = res
    if df.empty:
        st.info(f"W ostatnich {days_run} dniach nie byłoby żadnego alertu.", icon=":material/info:")
        return
    per_day = len(df) / days_run
    with st.container(horizontal=True):
        st.metric("Alerty", str(len(df)), f"{per_day:.1f} dziennie", delta_color="off", delta_arrow="off")
        for src, n in df["źródło"].value_counts().items():
            s = str(src)
            st.metric(s[:1].upper() + s[1:], str(n))
    st.dataframe(
        df, hide_index=True, alt=f"Odtworzone alerty {inst.name} z ostatnich {days_run} dni",
        column_config={"czas": st.column_config.DatetimeColumn(format="YYYY-MM-DD HH:mm")},
    )
    st.download_button(
        "Pobierz CSV", df.to_csv(index=False).encode("utf-8-sig"), file_name=f"alerty_{k}_{days_run}d.csv",
        mime="text/csv", icon=":material/download:", key=f"rp_csv_{k}",
    )


def render_entry_watches(
    inst: Instrument, cfg: dict, scenarios: dict[str, tuple[float, float, float]],
    volume: float, limit_h: float, cost: float, min_rr: float, frames: dict[str, pd.DataFrame], off: float,
) -> None:
    k, dec = inst.key, inst.decimals
    watches = cfg["obserwacje wejścia"]
    sent = notify.load_state(k).get("obserwacje wejścia", {})
    now = pd.Timestamp.now(tz=data.TZ_LOCAL)
    h1 = frames["H1"]
    latest = h1.iloc[-1]
    latest_closed = signals.last_closed(h1, "H1", now)
    st.markdown("#### Obserwuj wejście")
    st.caption(
        "Zapisz poziomy z karty Long lub Short. Monitor sprawdzi świecę H1 zamkniętą po zapisaniu obserwacji "
        "i wyśle jeden alert, gdy warunki będą spełnione, a cena będzie między SL i TP. "
        "Zysk/ryzyko w alercie jest przeliczone dla ceny sygnału."
    )
    for direction, col in zip(plan.DIRECTIONS, st.columns(2)):
        with col, st.container(border=True):
            st.markdown(f"**{direction.capitalize()}**")
            watch = watches.get(direction)
            if watch:
                expires = pd.Timestamp(watch["ważny do"]).tz_convert(data.TZ_LOCAL)
                signal = sent.get(watch["id"])
                if signal:
                    closed = pd.Timestamp(signal["świeca"]).tz_convert(data.TZ_LOCAL)
                    if "ekstremum odrzucenia" in signal:
                        reaction = plan.watch_rejection_text(direction, watch["próg"], signal["ekstremum odrzucenia"], signal["cena"], dec)
                        st.success(f"OTWÓRZ {direction.upper()} · {closed:%d.%m %H:%M}\n\n{reaction} Zysk do ryzyka {signal['rr']:.2f}:1 po kosztach.")
                    else:
                        st.success(f"Sygnał H1 {closed:%d.%m %H:%M} po {_num(signal['cena'], dec)}. Zysk/ryzyko {signal['rr']:.2f}.")
                elif now < expires and watch.get("odrzucenie H1"):
                    created = pd.Timestamp(watch["utworzono"])
                    sign = 1 if direction == "long" else -1
                    level = watch["próg"]
                    closed_since_creation = latest_closed is not None and latest_closed.name >= created
                    if (latest.name >= created and not data.bar_closed("H1", latest.name, now)
                            and plan.watch_rejects_level(direction, watch, latest, off)):
                        st.info("CZEKAJ NA ZAMKNIĘCIE H1 · Cena przebiła poziom i wróciła. Świeca nadal trwa.")
                    elif closed_since_creation and plan.watch_rejects_level(direction, watch, latest_closed, off):
                        when = data.bar_end("H1", latest_closed.name)
                        price = float(latest_closed["close"]) + off
                        candidate = plan.entry_signal(direction, watch, frames, price, when)
                        extreme = float(latest_closed["low" if direction == "long" else "high"]) + off
                        reaction = plan.watch_rejection_text(direction, watch["próg"], extreme, price, dec)
                        if candidate is None:
                            st.info(f"CZEKAJ · {reaction} Trend H4/D1 lub parametry planu nie potwierdzają wejścia.")
                        elif candidate["rr"] < watch["min rr"]:
                            st.info(f"CZEKAJ · {reaction} Zysk do ryzyka {candidate['rr']:.2f}:1 przy wymaganym {watch['min rr']:g}:1.")
                        else:
                            st.success(f"OTWÓRZ {direction.upper()} · {reaction} Zysk do ryzyka {candidate['rr']:.2f}:1 po kosztach.")
                    elif (closed_since_creation and (latest_closed["open"] + off - level) * sign > 0
                          and (latest_closed["close"] + off - level) * sign < 0):
                        st.info("CZEKAJ · H1 zamknęła się po drugiej stronie poziomu. Brak odrzucenia.")
                    else:
                        st.caption("CZEKAJ NA ODRZUCENIE POZIOMU · Monitor czeka na przebicie i powrót na zamkniętej H1.")
                else:
                    st.caption("Obserwacja wygasła." if now >= expires else "Czeka na spełnienie warunków.")
                st.caption(
                    f"Próg {_num(watch['próg'], dec)} · SL {_num(watch['sl'], dec)} · TP {_num(watch['tp'], dec)} · "
                    f"ważny do {expires:%d.%m.%Y %H:%M} (czas polski)"
                )
                if st.button("Usuń obserwację", key=f"watch_cancel_{k}_{direction}"):
                    del watches[direction]
                    storage.save_settings(cfg, k)
                    st.rerun()
            else:
                with st.form(f"watch_{k}_{direction}"):
                    require_rejection = st.checkbox(
                        "Wymagaj odrzucenia progu na H1", key=f"watch_rejection_{k}_{direction}",
                        help="Świeca musi przebić próg i zamknąć się po tej samej stronie, po której się otworzyła.",
                    )
                    expires_at = st.datetime_input(
                        "Ważny do (czas polski)", value=(now.tz_localize(None).floor("min") + pd.Timedelta(days=1)).isoformat(),
                        format="DD/MM/YYYY", required=True, key=f"watch_expiry_{k}_{direction}",
                    )
                    if st.form_submit_button("Obserwuj wejście"):
                        entry, sl, tp = scenarios[direction]
                        errors = plan.validate_plan(direction, entry, sl, tp, volume, limit_h)
                        if expires_at is None:
                            st.error("Podaj datę i godzinę ważności obserwacji.")
                        else:
                            expires = pd.Timestamp(expires_at).tz_localize(data.TZ_LOCAL, ambiguous="NaT", nonexistent="NaT")
                            if not isinstance(expires, pd.Timestamp) or expires <= now:
                                errors.append("wybierz przyszłą, jednoznaczną godzinę w czasie polskim")
                            if isinstance(expires, pd.Timestamp) and not errors:
                                watches[direction] = plan.new_watch(
                                    entry, sl, tp, volume, limit_h, cost, min_rr, now, expires,
                                    require_rejection=require_rejection,
                                )
                                storage.save_settings(cfg, k)
                                st.rerun()
                            st.error("Popraw obserwację: " + ", ".join(errors))


def render_rejection_history(inst: Instrument, frames: dict[str, pd.DataFrame], limit_h: float, cost: float) -> None:
    details = st.expander("Test H1 · Dotknięcie a odrzucenie", on_change="rerun", key=f"rejection_history_{inst.key}")
    if not details.open:
        return
    with details:
        st.caption(
            "Poziomy to minimum i maksimum ostatniej zakończonej sesji D1. Wejście na następnym otwarciu H1. "
            f"SL 1 ATR · TP 2 ATR · limit {limit_h:g} godz. · koszt {cost:g} pkt na pozycję. "
            "Każdy przypadek jest oceniany osobno, także gdy pozycje nakładają się w czasie."
        )
        trades = backtest.rejection_history(frames, limit_h, cost)
        if trades.empty:
            st.info("Brak zakończonych przypadków do porównania.")
            return
        rows = []
        for direction in ("LONG", "SHORT"):
            by_direction = trades[trades["kierunek"] == direction]
            for label, sample in (("Dotknięcie", by_direction), ("Odrzucenie", by_direction[by_direction["odrzucenie"]])):
                count = len(sample)
                wins = int((sample["wynik"] == "TP").sum())
                rows.append({
                    "Kierunek": direction, "Warunek": label, "Przypadki": count, "TP": wins,
                    "SL": int((sample["wynik"] == "SL").sum()),
                    "Limit": int((sample["wynik"] == "limit").sum()),
                    "TP przed SL": f"{wins / count:.1%}" if count >= backtest.MIN_SAMPLE else "Za mało danych",
                    "Średni wynik po kosztach [pkt]": sample["wynik [pkt]"].mean(),
                })
        st.dataframe(pd.DataFrame(rows), hide_index=True, alt="Porównanie dotknięć i odrzuceń na H1",
                     column_config={"Średni wynik po kosztach [pkt]": st.column_config.NumberColumn(format="%.2f")})
        st.caption(f"Dotknięcia obejmują odrzucenia. Odsetek TP pokazujemy od {backtest.MIN_SAMPLE} rozstrzygniętych przypadków.")


def render_plan_tab(
    inst: Instrument, cfg: dict, frames: dict[str, pd.DataFrame], off: float, price: float,
    levels: list[tuple[str, float]], usdpln: float | None,
) -> None:
    k, dec = inst.key, inst.decimals
    step = 0.1 if dec > 1 else 1.0
    ps = cfg["plan_ustawienia"]
    st.caption(
        "Porównanie warunków, które sam ustawiasz – aplikacja nie wybiera kierunku. Alerty dotyczą poziomów wykrytych "
        f"w danych panelu ({inst.source}, opóźnienie do ok. 15 min, z przesunięciem). Cenę wykonania sprawdzasz w xStation; "
        "XTB zastrzega, że cena realizacji SL może się różnić od ustawionego poziomu."
    )

    with st.container(horizontal=True, vertical_alignment="bottom"):
        vol = st.number_input("Wolumen [lot]", min_value=0.001, value=float(ps["wolumen"]), step=0.001, format="%.3f", key=f"pl_vol_{k}")
    cost, min_rr, limit_h = (float(ps[name]) for name in ("koszt_pkt", "min_rr", "limit_h"))
    new_ps = {**ps, "wolumen": vol}
    if new_ps != ps:
        cfg["plan_ustawienia"] = new_ps
        storage.save_settings(cfg, k)

    def _reset() -> None:
        for d in plan.DIRECTIONS:
            for f in ("in", "sl", "tp"):
                st.session_state.pop(f"pl_{k}_{d}_{f}", None)

    st.button("Przywróć propozycję poziomów", icon=":material/restart_alt:", key=f"pl_reset_{k}", on_click=_reset)

    usd_pp = float(cfg["usd_za_punkt"])
    results: dict[str, dict] = {}
    scen: dict[str, tuple[float, float, float]] = {}
    col_l, col_s, col_w = st.columns(3)
    for direction, col in (("long", col_l), ("short", col_s)):
        d = plan.default_scenario(direction, frames, levels, price, off, dec)
        with col, st.container(border=True):
            st.subheader(direction.capitalize())
            side = "powyżej" if direction == "long" else "poniżej"
            entry = st.number_input(f"Wejście: zamknięcie H1 {side}", value=d["wejście"], step=step, key=f"pl_{k}_{direction}_in")
            sl = st.number_input("Stop-loss", value=d["sl"], step=step, key=f"pl_{k}_{direction}_sl",
                                 help="Domyślnie za ostatnim dołkiem (long) / szczytem (short) H4, z buforem 0,1 ATR H4.")
            tp = st.number_input("Take-profit", value=d["tp"], step=step, key=f"pl_{k}_{direction}_tp",
                                 help="Domyślnie kolejny poziom z listy w kierunku pozycji.")
            ev = plan.evaluate_scenario(direction, entry, sl, tp, frames, off, vol, usd_pp, usdpln, cost, min_rr)
            results[direction], scen[direction] = ev, (entry, sl, tp)
            st.dataframe(
                pd.DataFrame(ev["checks"]), hide_index=True,
                column_config={"Spełniony": st.column_config.CheckboxColumn(width="small")},
                alt=f"Warunki scenariusza {direction}",
            )
            if not ev["poprawny"]:
                st.warning("SL i TP muszą leżeć po przeciwnych stronach wejścia.", icon=":material/warning:")
            atr_txt = " / ".join(f"{v:.1f}" if v is not None else "–" for v in (ev["sl_atr_h1"], ev["sl_atr_d1"]))
            with st.container(horizontal=True):
                st.metric("Ryzyko", f"{_num(ev['ryzyko_pkt'], 1)} pkt", f"{atr_txt} ATR H1/D1", delta_color="off", delta_arrow="off")
                st.metric("Zysk/ryzyko", f"{ev['rr']:.2f}" if ev["rr"] is not None else "–")
            with st.container(horizontal=True):
                st.metric("Strata przy SL", f"{_num(-ev['strata_pln'], 2)} PLN" if ev["strata_pln"] is not None else "brak USD/PLN")
                st.metric("Wynik przy TP", f"{_num(ev['zysk_pln'], 2)} PLN" if ev["zysk_pln"] is not None else "brak USD/PLN")
            atr_h1 = frames["H1"]["atr"].iloc[-1]
            render_holding(inst, direction, ev, entry, tp, float(atr_h1) if pd.notna(atr_h1) else None, limit_h)

    with col_w, st.container(border=True):
        st.subheader("Czekaj")
        complete = [d for d, ev in results.items() if ev["spełnione"] == len(ev["checks"])]
        for d, ev in results.items():
            missing = [c["Warunek"] for c in ev["checks"] if not c["Spełniony"]]
            st.markdown(f"**{d.capitalize()}**: spełnione {ev['spełnione']}/{len(ev['checks'])}")
            if missing:
                st.markdown("\n".join(f"- brakuje: {m}" for m in missing))
        if not complete:
            st.info("Żaden scenariusz nie ma kompletu ustawionych warunków.", icon=":material/hourglass_top:")
        with st.expander("Jak liczony jest czas trzymania"):
            st.markdown(
                "Podobny sygnał to zamknięcie H1 powyżej najwyższego szczytu (long) albo poniżej najniższego dołka "
                "(short) z 20 poprzednich świec. SL i TP stoją w tej samej odległości w ATR H1 co w scenariuszu. "
                "Każdy przypadek śledzimy do TP, SL albo limitu czasu (liczonego zegarowo, jak w monitorze). "
                "Gdy w jednej świecy padł i SL, i TP, liczymy SL. Bez spreadu i poślizgu. "
                f"Przy mniej niż {backtest.MIN_SAMPLE} przypadkach pokazujemy „za mało danych”."
            )

    render_entry_watches(inst, cfg, scen, vol, limit_h, cost, min_rr, frames, off)

    # -------------------------------------------------------- ten plan na historii
    st.markdown("#### Sprawdź ten plan na historii")
    st.caption(
        "Dokładnie warunki z kart powyżej: zamknięcie H1 za poziomem wejścia, trend H4 i D1 nie przeciwny "
        "(z zamkniętych świec, bez zaglądania w przyszłość), zysk/ryzyko po kosztach ≥ minimum. Wejście na zamknięciu "
        "świecy, w której komplet warunków pojawił się po raz pierwszy; SL i TP to ceny z planu; wynik po kosztach. "
        "Ponieważ poziomy są stałymi cenami, liczą się tylko okresy, gdy rynek był w ich pobliżu – próba bywa mała."
    )
    params = (round(off, 4), vol, cost, min_rr, limit_h, tuple(scen["long"]), tuple(scen["short"]))
    if st.button("Sprawdź ten plan na historii", icon=":material/manage_search:", key=f"pl_hist_{k}"):
        st.session_state[f"pl_hist_res_{k}"] = (
            params,
            {d: backtest.plan_history(frames, d, *scen[d], off, cost, min_rr, limit_h) for d in plan.DIRECTIONS},
        )
    saved = st.session_state.get(f"pl_hist_res_{k}")
    if saved:
        saved_params, res = saved
        if saved_params != params:
            st.caption("Poziomy lub ustawienia zmieniły się od ostatniego sprawdzenia – kliknij przycisk ponownie.")
        cols = st.columns(2)
        for d, col in zip(plan.DIRECTIONS, cols):
            with col, st.container(border=True):
                render_plan_history(d, res[d], vol, usd_pp, usdpln, limit_h)

    # -------------------------------------------------------- zapis planu po otwarciu pozycji
    st.markdown("#### Zapisz plan po otwarciu pozycji w xStation")
    direction = st.segmented_control("Kierunek otwartej pozycji", plan.DIRECTIONS, default="long", key=f"pl_dir_{k}") or "long"
    s_entry, s_sl, s_tp = scen[direction]
    watches = cfg["obserwacje wejścia"]
    watch = watches.get(direction)
    use_watch = watch is not None and st.checkbox("Użyj zapisanej obserwacji wejścia", key=f"pl_use_watch_{k}_{direction}")
    volume, holding_limit = vol, limit_h
    source = "scenariusz"
    if use_watch:
        s_entry, s_sl, s_tp = watch["próg"], watch["sl"], watch["tp"]
        volume, holding_limit = watch["wolumen"], watch["limit godzin"]
        source = watch["id"]
        st.caption(f"Limit trzymania z obserwacji: {holding_limit:g} godz.")
    with st.form(f"pl_save_{k}_{direction}_{source}"):
        c1, c2, c3, c4 = st.columns(4)
        real_entry = c1.number_input("Faktyczna cena otwarcia z xStation", value=None if use_watch else float(s_entry),
                                     step=step, format=f"%.{dec}f")
        p_sl = c2.number_input("SL", value=float(s_sl), step=step, format=f"%.{dec}f")
        p_tp = c3.number_input("TP", value=float(s_tp), step=step, format=f"%.{dec}f")
        real_volume = c4.number_input("Wolumen transakcji [lot]", min_value=0.001, value=float(volume), step=0.001,
                                      format="%.3f", key=f"pl_open_vol_{k}_{direction}_{source}",
                                      help="Wpisz faktyczny wolumen transakcji z xStation.")
        note = st.text_input("Notatka (opcjonalnie)")
        if st.form_submit_button("Zapisz plan", icon=":material/save:"):
            errs = plan.validate_plan(direction, real_entry, p_sl, p_tp, real_volume, holding_limit) if real_entry is not None else []
            if real_entry is None:
                st.error("Podaj faktyczną cenę otwarcia z xStation.")
            elif errs:
                st.error("\n".join(f"- {e}" for e in errs), title="Nie zapisano planu")
            else:
                cfg["plany"].append(plan.new_plan(direction, real_entry, p_sl, p_tp, real_volume, holding_limit, s_entry, note))
                if use_watch:
                    del watches[direction]
                storage.save_settings(cfg, k)
                st.toast("Zapisano plan. Monitor zacznie go śledzić w najbliższym cyklu.", icon=":material/check:")
                st.rerun()

    open_plans = [p for p in cfg["plany"] if p.get("status") == "otwarty"]
    st.markdown("#### Otwarte plany")
    if not open_plans:
        st.caption("Brak otwartych planów.")
    else:
        fx = f" · USD/PLN {_num(usdpln, 4)}" if usdpln is not None else " · brak kursu USD/PLN"
        st.caption(f"Wycena wg danych panelu, bez spreadu, swapów i prowizji. 1 lot: {usd_pp:g} USD za punkt{fx}.")
        st.dataframe(
            pd.DataFrame([plan.plan_status(p, price, usd_pp, usdpln) for p in open_plans]),
            column_config={"Wolumen [lot]": st.column_config.NumberColumn(format="%.3f")},
            hide_index=True, alt="Otwarte plany pozycji z odległością do SL i TP",
        )
        with st.expander("Skoryguj wolumen otwartej pozycji"):
            selected = st.selectbox(
                "Pozycja do korekty", list(range(len(open_plans))), key=f"pl_volume_select_{k}",
                format_func=lambda i: (
                    f"{open_plans[i]['kierunek'].upper()} · wejście {_num(open_plans[i]['cena wejścia'], dec)} · "
                    f"{open_plans[i]['wolumen']:.3f} lota · {open_plans[i]['id']}"
                ),
            )
            p = open_plans[selected]
            with st.form(f"pl_correct_volume_{k}_{p['id']}"):
                corrected = st.number_input("Faktyczny wolumen [lot]", min_value=0.001, value=float(p["wolumen"]),
                                            step=0.001, format="%.3f", key=f"pl_volume_corrected_{k}_{p['id']}")
                if st.form_submit_button("Zapisz poprawiony wolumen"):
                    if 0.001 <= corrected < float("inf"):
                        p["wolumen"] = corrected
                        storage.save_settings(cfg, k)
                        st.rerun()
                    st.error("Wpisz prawidłowy wolumen od 0,001 lota.")
        with st.form(f"pl_close_form_{k}"):
            st.markdown("**Zamknij plan – wpis do dziennika**")
            c1, c2, c3 = st.columns(3)
            pid = c1.selectbox("Plan", [p["id"] for p in open_plans], key=f"pl_close_id_{k}")
            exit_px = c2.number_input("Cena wyjścia z xStation", value=float(round(price, dec)), step=step, key=f"pl_close_px_{k}")
            reason = c3.selectbox("Powód wyjścia", journal.REASONS, key=f"pl_close_reason_{k}")
            c4, c5, c6 = st.columns(3)
            now_ts = pd.Timestamp.now()
            d = c4.date_input("Data wyjścia", value=now_ts.date(), key=f"pl_close_d_{k}")
            t = c5.time_input("Godzina wyjścia", value=now_ts.time().replace(second=0, microsecond=0), key=f"pl_close_t_{k}")
            pln_real = c6.number_input("Wynik w PLN z xStation (opcjonalnie)", value=None, step=1.0, key=f"pl_close_pln_{k}",
                                       help="Wynik z historii transakcji XTB, ze spreadem i swapami. Bez niego dziennik pokaże szacunek.")
            note = st.text_input("Notatka (dlaczego tak, czego się nauczyłem)", key=f"pl_close_note_{k}")
            if st.form_submit_button("Zamknij plan i zapisz w dzienniku", icon=":material/check_circle:"):
                p = next(x for x in cfg["plany"] if x["id"] == pid)
                closed_iso = f"{d:%Y-%m-%d}T{t:%H:%M}"
                errs = journal.validate_close(p, exit_px, closed_iso, reason)
                if errs:
                    st.error("\n".join(f"- {e}" for e in errs), title="Nie zamknięto planu")
                else:
                    journal.close_plan(p, exit_px, closed_iso, reason, pln_real, note)
                    storage.save_settings(cfg, k)
                    st.toast("Plan zamknięty i zapisany w dzienniku.", icon=":material/check:")
                    st.rerun()
    n_closed = sum(1 for p in cfg["plany"] if p.get("status") == "zamknięty")
    if n_closed:
        st.caption(f"Zamknięte transakcje ({n_closed}) są w zakładce „Dziennik”.")


def render_simulation(inst: Instrument, cfg: dict) -> None:
    k = inst.key
    active = cfg["symulacja"]
    settings = cfg["plan_ustawienia"]
    volume = paper.VOLUME_BY_INSTRUMENT[k]
    st.subheader("Symulacja na bieżących notowaniach")
    st.button("Odśwież symulację", key=f"paper_refresh_{k}")
    state = paper.load(k)
    st.caption(
        "Po sygnale Doradcy monitor symuluje wejście po ostatniej dostępnej cenie w następnej H1. "
        "SL, TP i limit rozlicza na świecach H1 rozpoczynających się po wejściu. "
        "Monitor musi działać w tle."
    )
    st.caption(f"Stały wolumen nowych pozycji w symulacji: {volume:.3f} lota.")
    enabled = st.checkbox("Symuluj propozycje Doradcy", value=bool(cfg["symulacja doradcy"]), key=f"paper_advisor_{k}")
    unit = "USD/oz ceny złota" if k == GOLD.key else "pkt indeksu US100"
    with st.container(horizontal=True, vertical_alignment="bottom"):
        cost = st.number_input(
            f"Koszt wejścia i wyjścia [{unit}]", min_value=0.0, value=float(settings["koszt_pkt"]),
            step=0.1, key=f"paper_cost_{k}",
            help="Łączny koszt transakcji przeliczony na ruch ceny instrumentu. "
                 "Ta wartość pomniejsza zysk i zwiększa ryzyko przy ocenie każdego sygnału Doradcy.",
        )
        min_rr = st.number_input("Min. zysk/ryzyko", min_value=0.1, value=float(settings["min_rr"]), step=0.1, key=f"paper_rr_{k}")
        limit = st.number_input("Limit czasu [godz.]", min_value=1.0, value=float(settings["limit_h"]), step=1.0, key=f"paper_limit_{k}")
    typical = ALL[k].typical_cost
    if cost > 5 * typical:
        st.warning(f"Koszt {cost:g} to ponad 5× typowy koszt w XTB (ok. {typical:g}: spread i poślizg). "
                   "Tak duży koszt odrzuca prawie każdy sygnał Doradcy. Sprawdź, czy nie wpisano kwoty w PLN lub USD za lot.")
    updated = {**settings, "koszt_pkt": cost, "min_rr": min_rr, "limit_h": limit}
    if updated != settings or enabled != cfg["symulacja doradcy"]:
        cfg["symulacja doradcy"] = enabled
        cfg["plan_ustawienia"] = updated
        storage.save_settings(cfg, k)
        st.rerun()
    for direction in tuple(active):
        if st.button(f"Wstrzymaj zapisany scenariusz {direction.upper()}", key=f"paper_pause_{k}_{direction}"):
            del active[direction]
            storage.save_settings(cfg, k)
            st.rerun()

    if decision := notify.load_state(k).get("ocena doradcy"):
        close = decision.get("świeca")
        label = f"H1 {pd.Timestamp(close).tz_convert(data.TZ_LOCAL):%d.%m %H:%M}" if close else "ostatni cykl"
        st.caption(f"Ocena monitora · {label} · {decision['powód']}")
    if last := state.get("sprawdzono"):
        stamp = pd.Timestamp(last).tz_convert(data.TZ_LOCAL)
        st.caption(f"Ostatnia przetworzona świeca zakończyła się {stamp:%d.%m.%Y o %H:%M} (czas polski).")
    elif (enabled or active) and not decision:
        st.info("Oczekiwanie na pierwszą propozycję spełniającą warunki wejścia.")

    positions = paper.positions(state)
    opened = positions[positions["status"].eq("otwarta")] if not positions.empty else positions
    if not opened.empty:
        st.markdown("#### Otwarte pozycje wirtualne")
        columns = ["źródło", "kierunek", "otwarto", "cena wejścia", "wolumen", "sl początkowy", "sl", "tp",
                   "zmiany SL", "wynik otwarty pkt", "maks. wynik pkt", "limit godzin", "description"]
        st.dataframe(opened.reindex(columns=columns),
                     hide_index=True, alt="Otwarte pozycje wirtualne i ich poziomy wyjścia")
        st.caption("Po wstrzymaniu nowych wejść monitor nadal śledzi otwarte pozycje.")

    closed = positions[positions["status"].eq("zamknięta")] if not positions.empty else positions
    st.markdown("#### Zamknięte pozycje wirtualne")
    if closed.empty:
        st.caption("Brak zamkniętych pozycji w symulacji.")
    else:
        with st.container(horizontal=True):
            st.metric("Pozycje", len(closed))
            st.metric("Dodatni wynik", f"{(closed['wynik pkt'] > 0).mean():.0%}")
            st.metric("Suma po kosztach", f"{closed['wynik pkt'].sum():+.2f} pkt")
        columns = ["źródło", "kierunek", "otwarto", "zamknięto", "cena wejścia", "cena wyjścia", "wolumen", "sl początkowy", "sl", "tp",
                   "zmiany SL", "powód", "wynik pkt", "wynik R", "godziny", "description"]
        st.dataframe(closed.iloc[::-1].reindex(columns=columns), hide_index=True, alt="Historia zamkniętych pozycji wirtualnych")
    if not positions.empty:
        st.caption(f"Monitor zapisuje otwarte i zamknięte pozycje w dane/{paper.csv_path(k).name}.")
        st.download_button("Pobierz dane symulacji CSV", paper.csv_bytes(positions),
                           file_name=paper.csv_path(k).name, mime="text/csv", key=f"paper_export_{k}")

    if records := [*state.get("otwarte", {}).values(), *reversed(state.get("zamknięte", []))]:
        with st.expander("Pełny opis decyzji"):
            selected = st.selectbox(
                "Pozycja wirtualna", range(len(records)), key=f"paper_description_{k}",
                format_func=lambda i: (
                    f"{records[i]['kierunek'].upper()} · {pd.Timestamp(records[i]['otwarto']).tz_convert(data.TZ_LOCAL):%d.%m.%Y %H:%M} · "
                    f"{records[i].get('powód', 'otwarta')}"
                ),
            )
            if selected is not None:
                description = records[selected].get("description")
                if description:
                    st.markdown("\n\n".join(description.splitlines()))
                else:
                    st.caption("Opis decyzji nie był zapisywany w tej wersji symulacji.")

    with st.expander("Jak symulacja rozlicza pozycje"):
        st.markdown(
            "- Przy wejściu zapisujemy poziomy, koszty, przesunięcie ceny i wskaźniki H1 widoczne w chwili sygnału.\n"
            "- TP pozostaje na poziomie z planu. SL przesuwa się tylko po nowym dodatnim maksimum wyniku po kosztach "
            "na zamknięciu H1. Zachowuje początkowy dystans od ceny i obowiązuje od następnej świecy.\n"
            "- Jeśli w jednej świecy osiągnięto SL i TP, przyjmujemy SL. Przy otwarciu świecy za SL "
            "przyjmujemy jej cenę otwarcia, a przy luce za TP przyjmujemy cenę TP.\n"
            "- Limit czasu rozliczamy na końcu świecy obejmującej limit lub po cenie otwarcia pierwszej świecy po przerwie.\n"
            "- CSV obejmuje otwarte i zamknięte pozycje, aktualny SL, liczbę jego przesunięć i opis decyzji. "
            "Czas wyjścia wewnątrz świecy jest przybliżony jej zamknięciem."
        )


@st.cache_data(ttl=600, max_entries=8, show_spinner="Sprawdzam rodziny sygnałów…")
def family_evaluations(
    frames: dict[str, pd.DataFrame], cost: float, min_rr: float, limit: float, as_of: pd.Timestamp, instrument: str,
    live_starts: dict[str, pd.Timestamp],
) -> list[families.Evaluation]:
    return families.evaluate_all(frames, families.Params(cost, min_rr, limit), as_of, instrument, live_starts)


def render_family_live(inst: Instrument, cfg: dict, report: list[families.Evaluation], now: pd.Timestamp, unit: str) -> None:
    """Symulacja na żywo: nowe sygnały rodzin rozliczane na kolejnych świecach, jak w testach."""
    st.markdown("**Symulacja na żywo**")
    state = cfg["symulacja rodzin"]
    if not state.get("start"):
        st.caption("Symulacja na żywo rozlicza tylko sygnały, które pojawią się od chwili startu, na kolejnych "
                   "zamkniętych świecach H1 – dokładnie tak jak w testach historycznych. To sprawdzian na danych, "
                   "których reguły nie mogły „widzieć”. Panel i monitor nie muszą działać bez przerwy: wynik liczy się "
                   "z zapisanych notowań.")
        if st.button("Rozpocznij symulację na żywo", icon=":material/play_arrow:", key=f"family_live_start_{inst.key}"):
            cfg["symulacja rodzin"] = {"start": now.isoformat(), "wersje": {}}
            storage.save_settings(cfg, inst.key)
            family_evaluations.clear()
            st.rerun()
        return
    since = pd.Timestamp(state["start"]).tz_convert(data.TZ_LOCAL)
    notify_sim = st.toggle(
        "Powiadomienia z symulacji (wirtualne wejścia i zamknięcia)", value=bool(cfg[family_alerts.ALERT_SETTING]),
        key=f"family_live_notify_{inst.key}",
        help="Monitor wysyła je po zamknięciu H1 z dopiskiem, że to nie jest sygnał do handlu. "
             "Alert „OTWÓRZ” dla rodziny, która spełni wszystkie warunki, jest wysyłany zawsze.",
    )
    if notify_sim != bool(cfg[family_alerts.ALERT_SETTING]):
        cfg[family_alerts.ALERT_SETTING] = notify_sim
        storage.save_settings(cfg, inst.key)
    st.caption(f"Symulacja trwa od {since:%d.%m.%Y %H:%M}. Wejście na otwarciu H1 po sygnale, koszt jak w testach, "
               "jedna pozycja naraz w każdej rodzinie. Zmiana reguł rodziny zaczyna jej symulację od nowa.")
    off = float(cfg["przesunięcie"])
    rows = []
    for ev in report:
        live = ev.live or advisor.History(0, 0, 0, 0, None)
        rows.append({
            "Rodzina": ev.name, "Od": f"{ev.live_start.tz_convert(data.TZ_LOCAL):%d.%m.%Y}" if ev.live_start is not None else "–",
            "Pozycje": live.count, "TP": live.targets, "SL": live.stops, "Limit": live.timeouts,
            f"Średnio [{unit}]": live.mean_points,
            f"Łącznie [{unit}]": live.mean_points * live.count if live.mean_points is not None else None,
            "Otwarta pozycja": ev.live_open.candidate.direction.upper() if ev.live_open else "–",
        })
    st.dataframe(pd.DataFrame(rows), hide_index=True, alt="Wyniki rodzin w symulacji na żywo",
                 column_config={name: st.column_config.NumberColumn(format="%+.2f")
                                for name in (f"Średnio [{unit}]", f"Łącznie [{unit}]")})
    for ev in report:
        position = ev.live_open
        if position is None:
            continue
        opened = position.opened.tz_convert(data.TZ_LOCAL)
        st.info(f"{ev.name}: wirtualna pozycja {position.candidate.direction.upper()} od {opened:%d.%m %H:%M} · "
                f"wejście {position.entry + off:.{inst.decimals}f} · SL {position.stop + off:.{inst.decimals}f} · "
                f"TP {position.target + off:.{inst.decimals}f} (ceny xStation). {position.candidate.note}.",
                icon=":material/science:")

    checks = []
    for ev in report:
        row = {"Rodzina": ev.name}
        for label, ok in families.readiness(ev, now):
            row[label.split(" (")[0]] = "✅" if ok else "⬜"
        checks.append(row)
    st.caption("Alerty „OTWÓRZ” dostanie tylko rodzina, która spełni wszystkie warunki:")
    st.dataframe(pd.DataFrame(checks), hide_index=True, alt="Gotowość rodzin do alertów")

    closed = sorted((trade for ev in report for trade in ev.live_trades), key=lambda t: t.closed, reverse=True)
    if closed:
        names = {ev.key: ev.name for ev in report}
        st.dataframe(pd.DataFrame([{
            "Rodzina": names.get(t.candidate.family, t.candidate.family), "Kierunek": t.candidate.direction.upper(),
            "Wejście": f"{t.opened.tz_convert(data.TZ_LOCAL):%d.%m %H:%M}",
            "Wyjście": f"{t.closed.tz_convert(data.TZ_LOCAL):%d.%m %H:%M}", "Wynik": t.outcome,
            f"Wynik [{unit}]": t.points,
        } for t in closed[:30]]), hide_index=True, alt="Ostatnie zamknięte pozycje symulacji na żywo",
            column_config={f"Wynik [{unit}]": st.column_config.NumberColumn(format="%+.2f")})
    with st.container(horizontal=True, vertical_alignment="center"):
        sure = st.checkbox("Potwierdzam: zacznij symulację od nowa", key=f"family_live_reset_ok_{inst.key}")
        if st.button("Zacznij od nowa", disabled=not sure, key=f"family_live_reset_{inst.key}"):
            cfg["symulacja rodzin"] = {"start": now.isoformat(), "wersje": {}}
            storage.save_settings(cfg, inst.key)
            family_evaluations.clear()
            st.rerun()


def render_families(inst: Instrument, cfg: dict, frames: dict[str, pd.DataFrame]) -> None:
    box = st.expander("Rodziny sygnałów – bramka dopuszczenia", key=f"families_{inst.key}", on_change="rerun")
    if not box.open:
        return
    with box:
        params = cfg["plan_ustawienia"]
        cost, min_rr, limit = (float(params[name]) for name in ("koszt_pkt", "min_rr", "limit_h"))
        h1 = frames["H1"]
        if len(h1) < families.WARMUP_BARS + advisor.MIN_VALIDATION_BARS:
            st.info("Za mało świec H1, żeby sprawdzić rodziny sygnałów.")
            return
        now = pd.Timestamp.now(tz="UTC")
        starts, changed = families.live_starts(cfg["symulacja rodzin"], inst.key, now)
        if changed:
            storage.save_settings(cfg, inst.key)  # nowa wersja reguł rodziny - jej symulacja liczy się od teraz
        report = family_evaluations(frames, cost, min_rr, limit, data.bar_end("H1", h1.index[-1]).tz_convert("UTC"),
                                    inst.key, starts)
        unit = "USD" if inst.key == GOLD.key else "pkt"
        tested = len(report)
        z = families.margin_z(tested)
        st.caption(
            f"Każda rodzina to osobna hipoteza z regułami ustalonymi z góry, rozliczana tak samo: wejście na otwarciu "
            f"następnej H1, koszt {cost:g} {unit}, SL przed TP w tej samej świecy, jedna pozycja naraz. "
            f"Alerty „OTWÓRZ” mogą dostać tylko rodziny ze statusem „przewaga”: co najmniej {families.MIN_TRADES} pozycji, "
            f"średni wynik minus {z:.2f} błędu standardowego (korekta na {tested} sprawdzane rodziny) "
            f"powyżej zera i losowych wejść oraz wynik dodatni w {families.MIN_POSITIVE_WINDOWS} z {families.WINDOWS} okien. "
            f"Ostatnie {families.HOLDOUT_DAYS} dni to dane nietknięte – nie wchodzą do oceny."
        )
        st.info(f"Reżim rynku teraz (ostatnia zamknięta H4): {regime.current(frames['H4'], now).text()}. "
                "Rodzina B handluje tylko w trendzie zgodnym z D1 i przy zwykłej zmienności.",
                icon=":material/insights:")
        icons = {"przewaga": "✅ przewaga", "brak przewagi": "⛔ brak przewagi", "za mało danych": "⏳ za mało danych"}
        rows = []
        for ev in report:
            m = families.margin(ev.total, tested)
            positive = sum(1 for w in ev.windows if w.stats.mean_points is not None and w.stats.mean_points > 0)
            rows.append({
                "Rodzina": ev.name, "Status": icons[ev.status], "Pozycje": ev.total.count,
                "Odsetek TP": f"{ev.total.targets / ev.total.count:.0%}" if ev.total.count else "–",
                f"Średnio [{unit}]": ev.total.mean_points, f"Margines ± [{unit}]": m,
                f"Losowe wejścia [{unit}]": ev.random.mean_points,
                "Okna z plusem": f"{positive}/{len(ev.windows)}",
            })
        numbers = {name: st.column_config.NumberColumn(format="%+.2f")
                   for name in (f"Średnio [{unit}]", f"Losowe wejścia [{unit}]")}
        numbers[f"Margines ± [{unit}]"] = st.column_config.NumberColumn(format="%.2f")
        st.dataframe(pd.DataFrame(rows), hide_index=True, column_config=numbers, alt="Ocena rodzin sygnałów")
        for ev in report:
            kind, text = ev.verdict
            {"success": st.success, "info": st.info, "warning": st.warning}[kind](text)

        st.markdown("**Rodziny i ich reguły**")
        off = float(cfg["przesunięcie"])
        for ev in report:
            st.markdown(f"**{ev.name}** – {ev.description}")
            fresh = ev.latest
            if fresh is None or now - fresh.at > families.MAX_LIVE_AGE:
                continue
            at = fresh.at.tz_convert(data.TZ_LOCAL)
            prices = (f"{fresh.direction.upper()} · świeca do {at:%d.%m %H:%M} · wejście ok. {fresh.reference + off:.{inst.decimals}f} · "
                      f"SL {fresh.stop + off:.{inst.decimals}f} · TP {fresh.target + off:.{inst.decimals}f} (ceny xStation). {fresh.note}.")
            if ev.status == "przewaga":
                st.success(f"Świeży sygnał: {prices}", icon=":material/notifications_active:")
            else:
                st.caption(f"Świeży kandydat tylko do obserwacji (rodzina bez przewagi): {prices}")

        window_rows = []
        for ev in report:
            for w in ev.windows:
                first, last = (t.tz_convert(data.TZ_LOCAL) for t in (w.start, w.end))
                window_rows.append({
                    "Rodzina": ev.name, "Okno": f"{first:%d.%m.%Y} – {last:%d.%m.%Y}", "Pozycje": w.stats.count,
                    f"Średnio [{unit}]": w.stats.mean_points, f"Losowe wejścia [{unit}]": w.random.mean_points,
                })
        st.dataframe(pd.DataFrame(window_rows), hide_index=True, alt="Rodziny sygnałów w kolejnych oknach",
                     column_config={name: st.column_config.NumberColumn(format="%+.2f")
                                    for name in (f"Średnio [{unit}]", f"Losowe wejścia [{unit}]")})

        render_family_live(inst, cfg, report, now, unit)

        holdout_start = report[0].holdout_start.tz_convert(data.TZ_LOCAL)
        if st.toggle(f"Pokaż dane nietknięte (od {holdout_start:%d.%m.%Y})", key=f"families_holdout_{inst.key}"):
            st.warning("Dane nietknięte oglądaj raz, gdy reguły są już ostateczne. Jeśli po ich obejrzeniu zmienisz "
                       "reguły, ten okres przestaje być niezależnym sprawdzianem.", icon=":material/visibility:")
            st.dataframe(pd.DataFrame([{
                "Rodzina": ev.name, "Pozycje": ev.holdout.count,
                "Odsetek TP": f"{ev.holdout.targets / ev.holdout.count:.0%}" if ev.holdout.count else "–",
                f"Średnio [{unit}]": ev.holdout.mean_points,
            } for ev in report]), hide_index=True, alt="Wyniki rodzin na danych nietkniętych",
                column_config={f"Średnio [{unit}]": st.column_config.NumberColumn(format="%+.2f")})


@st.cache_data(ttl=600, max_entries=12, show_spinner="Odtwarzam sygnały doradcy na historii…")
def advisor_history(
    frames: dict[str, pd.DataFrame], cost: float, min_rr: float, limit: float, as_of: pd.Timestamp,
) -> dict[tuple[str, str, int], advisor.History]:
    return advisor.replay(frames, cost, min_rr, limit, as_of)


@st.cache_data(ttl=600, max_entries=12, show_spinner="Porównuję warianty w kolejnych oknach…")
def advisor_validation(
    frames: dict[str, pd.DataFrame], cost: float, min_rr: float, limit: float, as_of: pd.Timestamp,
) -> advisor.WalkForward | None:
    return advisor.validate_walk_forward(frames, cost, min_rr, limit, as_of)


def render_advisor_validation(inst: Instrument, cfg: dict, frames: dict[str, pd.DataFrame]) -> None:
    test = st.expander("Porównanie wariantów Doradcy", key=f"advisor_validation_{inst.key}", on_change="rerun")
    if not test.open:
        return
    with test:
        params = cfg["plan_ustawienia"]
        cost, min_rr, limit = (float(params[name]) for name in ("koszt_pkt", "min_rr", "limit_h"))
        now = pd.Timestamp.now(tz="UTC")
        closed = {}
        for interval, frame in frames.items():
            last = signals.last_closed(frame, interval, now)
            closed[interval] = frame.iloc[:0] if last is None else frame.loc[:last.name]
        h1 = closed["H1"]
        if len(h1) < advisor.MIN_VALIDATION_BARS:
            st.info(f"Potrzeba co najmniej {advisor.MIN_VALIDATION_BARS} zamkniętych H1. Źródło zawiera {len(h1)}.")
            return
        report = advisor_validation(closed, cost, min_rr, limit, data.bar_end("H1", h1.index[-1]))
        if report is None:
            st.info("Za mało zamkniętych świec do podziału.")
            return
        total = report.total
        since, end = (t.tz_convert(data.TZ_LOCAL) for t in (total.cutoff, total.end))
        st.caption(f"Okres testowy {since:%d.%m.%Y %H:%M} – {end:%d.%m.%Y %H:%M} (czas polski) · "
                   f"{total.test_bars} H1 w trzech oknach. Przed pierwszym oknem dostępne było {total.preparation_bars} H1.")
        st.caption("Pełny Doradca dodaje filtry wskaźników, wolumenu i wcześniejszych wyników. "
                   "Obie opcje mają te same poziomy, reguły wejścia i wyjścia oraz maksymalnie jedną otwartą pozycję. "
                   f"Limit {limit:g} godz. · koszt {cost:g} pkt · minimalny zysk do ryzyka {min_rr:g}:1.")

        def comparison_rows(period: advisor.Validation) -> list[dict]:
            return [{
                "Wariant": label, "Zakończone pozycje": stats.count, "TP": stats.targets,
                "SL": stats.stops, "Limit": stats.timeouts,
                "Odsetek TP": f"{stats.targets / stats.count:.1%}" if stats.count >= advisor.MIN_HISTORY else "Za mało danych",
                "Średni wynik po kosztach [pkt]": stats.mean_points,
                "Łączny wynik po kosztach [pkt]": stats.mean_points * stats.count if stats.mean_points is not None else None,
            } for label, stats in (("Odrzucenie H1", period.baseline), ("Pełny Doradca", period.advisor),
                                   (f"Losowe wejścia (×{advisor.RANDOM_DRAWS})", period.random))
              if stats is not None]

        columns = {name: st.column_config.NumberColumn(format="%.2f")
                   for name in ("Średni wynik po kosztach [pkt]", "Łączny wynik po kosztach [pkt]")}
        st.dataframe(pd.DataFrame(comparison_rows(total)), hide_index=True, alt="Łączne wyniki obu wariantów",
                     column_config=columns)
        st.caption(f"Losowe wejścia: dla każdej pozycji wariantu „Odrzucenie H1” {advisor.RANDOM_DRAWS} razy losujemy "
                   "godzinę wejścia z tego samego okna i kierunek, z tymi samymi odległościami SL i TP oraz kosztem. "
                   "Reguła, która nie bije tego wyniku, nie ma przewagi nad przypadkiem.")
        for label, stats in (("Odrzucenie H1", total.baseline), ("Pełny Doradca", total.advisor)):
            kind, text = advisor.against_chance(label, stats, total.random, unit="USD" if inst.key == GOLD.key else "pkt")
            {"success": st.success, "info": st.info, "warning": st.warning}[kind](text)
        rows = []
        for window in report.windows:
            first, last = (t.tz_convert(data.TZ_LOCAL) for t in (window.cutoff, window.end))
            label = f"{first:%d.%m.%Y %H:%M} – {last:%d.%m.%Y %H:%M}"
            rows.extend({"Okno testowe": label, **row} for row in comparison_rows(window))
        st.dataframe(pd.DataFrame(rows), hide_index=True, alt="Wyniki obu wariantów w kolejnych oknach",
                     column_config=columns)
        st.caption("Na początku każdego okna Doradca używa wyników już zakończonych pozycji. "
                   "Pozycję zaliczamy do okna jej sygnału.")


def assess_for_display(
    frames: dict[str, pd.DataFrame], settings: dict, now: pd.Timestamp, *, live: bool,
) -> tuple[advisor.Assessment, advisor.History | None, str]:
    cost, min_rr, limit = (float(settings[name]) for name in ("koszt_pkt", "min_rr", "limit_h"))
    outcome = advisor.assess(frames, cost, min_rr, now, live=live)
    if outcome.reason or outcome.proposal is None:
        return outcome, None, ""
    proposal = outcome.proposal
    report = advisor_history(frames, cost, min_rr, limit, proposal.at)
    estimate, sample = advisor.comparable(proposal, report)
    return advisor.Assessment(advisor.history_hold(estimate), proposal), estimate, sample


def monitor_view(inst: Instrument, fallback: dict[str, pd.DataFrame]) -> tuple[dict[str, pd.DataFrame], dict]:
    snapshot = delivery.read_json(notify.DATA_DIR / "monitor_status.json", {}).get("notowania", {}).get(inst.key, {})
    if snapshot.get("H1"):
        try:
            return get_monitor_frames(inst.ticker, {"H1": snapshot["H1"]}), snapshot
        except (OSError, ValueError, KeyError):
            pass
    return fallback, {}


def monitor_assessment(
    frames: dict[str, pd.DataFrame], settings: dict, snapshot: dict,
) -> tuple[advisor.Assessment, advisor.History | None, str]:
    now = pd.Timestamp.now(tz="UTC")
    observed = pd.Timestamp(snapshot["H1"]["pobrano"]) if snapshot.get("H1") else frames["H1"].attrs.get("pobrano", now)
    if now - observed > advisor.MAX_SIGNAL_AGE:
        return advisor.Assessment("Dane są nieaktualne. Czekam na nowe pobranie."), None, ""
    outcome, estimate, sample = assess_for_display(frames, settings, observed, live=True)
    if outcome.proposal is not None and now - outcome.proposal.at > advisor.MAX_SIGNAL_AGE:
        return advisor.Assessment("Sygnał wygasł. Czekam na nową świecę H1."), None, ""
    return outcome, estimate, sample


@st.fragment(run_every="45s")
def render_live_market(
    inst: Instrument, cfg: dict, bars: int, off: float, levels: list[tuple[str, float]],
    fallback: dict[str, pd.DataFrame],
) -> None:
    frames, snapshot = monitor_view(inst, fallback)
    st.subheader("Wykres")
    interval = st.segmented_control("Interwał wykresu", ("M5", "M15", "H1", "H4", "D1"),
                                    default="H1", key=f"interval_{inst.key}") or "H1"
    frame = frames.get(interval)
    short = interval in data.CHART_INTRADAY
    if short:
        try:
            frame = engine.load_short_chart(inst.ticker, interval, snapshot) if snapshot else get_intraday_chart(inst.ticker, interval)
        except (FileNotFoundError, ValueError) as exc:
            st.info("Monitor przygotowuje archiwum M5." if snapshot else f"Brak świec {interval}: {exc}")
            return
        except Exception as exc:
            st.warning(f"Nie udało się odczytać świec {interval}: {exc}")
            return
    if frame is None or frame.empty:
        st.info(f"Brak świec {interval}.")
        return

    source = snapshot.get("M5" if short else "H1", {})
    observed = frame.attrs.get("pobrano", pd.Timestamp.now(tz="UTC"))
    if source:
        observed = pd.Timestamp(source["pobrano"])
    observed = observed.tz_convert(data.TZ_LOCAL)
    price = float(frame["close"].iloc[-1]) + off
    st.metric(inst.name, _num(price, inst.decimals))
    last = signals.last_closed(frame, interval, observed)
    stamps = []
    if last is not None:
        end = data.bar_end(interval, last.name).tz_convert(data.TZ_LOCAL)
        stamps.append(f"Świeca {interval} zamknięta {end:%d.%m %H:%M}")
    if source or "pobrano" in frame.attrs:
        label = "Pobranie przez monitor" if source else "Pobranie przez panel"
        stamps.append(f"{label} {observed:%d.%m %H:%M:%S}")
    if stamps:
        st.caption(" · ".join(stamps) + " (czas polski)")
    outcome, _, _ = monitor_assessment(frames, cfg["plan_ustawienia"], snapshot)
    render_market_chart(inst, frame, interval, bars, off, levels, frames, observed,
                        outcome.proposal if not outcome.reason else None)


@st.fragment(run_every="45s")
def render_live_advisor(
    inst: Instrument, cfg: dict, fallback: dict[str, pd.DataFrame], off: float,
    context: list[tuple[str, float | None, float | None, bool]],
) -> None:
    frames, snapshot = monitor_view(inst, fallback)
    outcome, estimate, sample = monitor_assessment(frames, cfg["plan_ustawienia"], snapshot)
    price = float(frames["H1"]["close"].iloc[-1]) + off
    rules_df = signals.evaluate_rules(cfg["reguły"], frames, off, now=frames["H1"].attrs.get("pobrano"))
    render_advisor(inst, cfg, frames, off, price, rules_df, context, pd.Timestamp.now(tz="UTC"), outcome, estimate, sample)


def render_advisor(
    inst: Instrument, cfg: dict, frames: dict[str, pd.DataFrame], off: float, price: float,
    rules_df: pd.DataFrame, context: list[tuple[str, float | None, float | None, bool]],
    now: pd.Timestamp, outcome: advisor.Assessment, estimate: advisor.History | None, sample: str,
) -> None:
    settings = cfg["plan_ustawienia"]
    cost, min_rr, limit = (float(settings[name]) for name in ("koszt_pkt", "min_rr", "limit_h"))
    proposal = outcome.proposal
    if proposal is None:
        st.info(outcome.reason, title="CZEKAJ")
        if outcome.nearest:
            level = outcome.nearest
            st.caption(f"Najbliższy obserwowany poziom · {level.name} {_num(level.price + off, inst.decimals)}. "
                       f"Odległość od ceny · {_num(abs(price - level.price - off), inst.decimals)}.")
    else:
        reason = outcome.reason
        direction = proposal.level.direction.upper()
        if reason:
            st.info(reason, title=f"CZEKAJ · obserwowany kierunek {direction}")
        else:
            details = f"Świeca H1 odrzuciła poziom {_num(proposal.level.price + off, inst.decimals)}."
            label = "Propozycja wstępna" if estimate is not None and estimate.count < advisor.MIN_HISTORY else "Propozycja"
            st.success(details, title=f"{label} {direction} · {proposal.level.name}")
        rr = f"{proposal.rr:.2f}:1" if proposal.rr is not None else "—"
        at = proposal.at.tz_convert(data.TZ_LOCAL)
        st.markdown(f"**H1 {at:%d.%m %H:%M} · Wejście {_num(proposal.entry + off, inst.decimals)} · "
                    f"SL {_num(proposal.stop + off, inst.decimals)} · TP {_num(proposal.target + off, inst.decimals)} · "
                    f"Zysk do ryzyka {rr} po kosztach**")
        if proposal.additional_levels:
            st.caption(f"Świeca odrzuciła także · {', '.join(proposal.additional_levels)}.")
        labels = {-1: "Przeciw", 0: "Neutralne", 1: "Za"}
        st.dataframe(pd.DataFrame([{"Warunek": detail, "Ocena": labels[vote]} for detail, vote in proposal.evidence]),
                     hide_index=True, alt="Warunki wspierające propozycję i przemawiające przeciw")
        if estimate is not None:
            st.markdown(f"**{estimate.description()}**")
            st.caption(f"Porównanie · {sample}. Limit {limit:g} godz. · koszt {cost:g} pkt na pozycję.")

    alert_enabled = bool(cfg.get(advisor.ALERT_SETTING))
    notify_advisor = st.checkbox("Powiadamiaj o nowych propozycjach doradcy", value=alert_enabled, key=f"advisor_alerts_{inst.key}")
    if notify_advisor != alert_enabled:
        cfg[advisor.ALERT_SETTING] = notify_advisor
        storage.save_settings(cfg, inst.key)

    market_context = st.expander("Kontekst rynku i własne reguły")
    with market_context:
        st.caption("Dodatkowy kontekst do własnej oceny. Porównanie historyczne korzysta z reguł automatycznych.")
        quotes = [f"{label} {value:.2f}" for label, value, _, _ in context if value is not None]
        if quotes:
            st.caption(" · ".join(quotes))
        if not rules_df.empty:
            passed = rules_df[rules_df["Spełniona"]]
            if not passed.empty:
                st.write("Spełnione reguły użytkownika")
                st.dataframe(passed[["Reguła", "Warunek"]], hide_index=True, alt="Spełnione reguły użytkownika")
        closer = plan.nearest_levels([(str(x["nazwa"]), float(x["poziom"])) for x in cfg["poziomy"] if x.get("poziom")], price)
        for level in closer:
            if level:
                st.caption(f"{level[0]} · {_num(level[1], inst.decimals)}")

    history = st.expander("Wyniki historyczne doradcy", key=f"advisor_history_{inst.key}", on_change="rerun")
    if history.open:
        with history:
            as_of = proposal.at if proposal else now.floor("h")
            report = advisor_history(frames, cost, min_rr, limit, as_of)
            rows = []
            for (family, direction, groups), stats in sorted(report.items()):
                if groups == 0:
                    rows.append({"Źródło sygnału": family, "Kierunek": direction.upper(), "Przypadki": stats.count,
                                 "TP przed SL i limitem": f"{stats.targets / stats.count:.1%}" if stats.count >= advisor.MIN_HISTORY else "Za mało danych",
                                 "Średni wynik po kosztach [pkt]": stats.mean_points})
            if rows:
                st.dataframe(pd.DataFrame(rows), hide_index=True, alt="Wyniki historyczne według źródła sygnału i kierunku",
                             column_config={"Średni wynik po kosztach [pkt]": st.column_config.NumberColumn(format="%.2f")})
            else:
                st.info("Brak zakończonych przypadków dla tych parametrów.")
            st.caption("W teście wejście następuje na kolejnym otwarciu H1. Gdy świeca obejmie SL i TP, test liczy SL. "
                       "Pozycje tej samej rodziny i kierunku są testowane bez nakładania się.")


@st.cache_data(ttl=600, max_entries=8, show_spinner="Porównuję wejścia H1 i M5…")
def timing_validation(
    frames: dict[str, pd.DataFrame], m5: pd.DataFrame, cost: float, min_rr: float, limit: float, now: pd.Timestamp,
) -> advisor.TimingComparison | None:
    return advisor.compare_entry_timing(frames, m5, cost, min_rr, limit, now)


def render_timing_validation(inst: Instrument, cfg: dict, frames: dict[str, pd.DataFrame]) -> None:
    details = st.expander("Wejścia H1 i M5 na wspólnym okresie", key=f"timing_validation_{inst.key}", on_change="rerun")
    if not details.open:
        return
    with details:
        try:
            m5 = engine.load_short_chart(inst.ticker, "M5")
        except (OSError, ValueError):
            st.info("Porównanie stanie się dostępne po zapisaniu świec M5 przez monitor.")
            return
        params = cfg["plan_ustawienia"]
        cost, min_rr, limit = (float(params[name]) for name in ("koszt_pkt", "min_rr", "limit_h"))
        now = min(pd.Timestamp.now(tz="UTC"), data.bar_end("M5", m5.index[-1]))
        report = timing_validation(frames, m5, cost, min_rr, limit, now)
        if report is None:
            st.info(f"Do porównania potrzeba co najmniej {advisor.MIN_VALIDATION_BARS} zamkniętych H1 we wspólnym okresie.")
            return
        first = report.start.tz_convert(data.TZ_LOCAL)
        last = report.end.tz_convert(data.TZ_LOCAL)
        st.caption(f"Okres testowy {first:%d.%m.%Y %H:%M}–{last:%d.%m.%Y %H:%M} · koszt {cost:g} pkt · "
                   f"minimum {min_rr:g}:1 · limit {limit:g} godz.")
        st.dataframe(pd.DataFrame([
            {"Sygnał": interval, "Zamknięte pozycje": stats.count, "TP": stats.targets, "SL": stats.stops,
             "Limit czasu": stats.timeouts, "Średni wynik po kosztach": stats.mean_points}
            for interval, stats in (("H1", report.h1), ("M5 z kontekstem H1/H4/D1", report.m5))
        ]), hide_index=True, alt="Historyczne porównanie sygnałów H1 i M5")
        st.caption("Oba warianty używają tych samych notowań M5 do symulacji wykonania wejść i wyjść. "
                   "Krótszy interwał służy tu do porównania historycznego.")


def render_archived(inst: Instrument, cfg: dict, frames: dict[str, pd.DataFrame], bars: int) -> None:
    st.subheader("Archiwum notowań")
    h1 = frames["H1"]
    end = data.bar_end("H1", h1.index[-1]).tz_convert(data.TZ_LOCAL)
    st.warning(f"Ostatnia zapisana H1 zamknęła się {end:%d.%m.%Y o %H:%M} (czas polski). "
               "Widok służy do wykresów i testów historycznych.", title="Dane z plików Parquet")
    now = min(end, pd.Timestamp.now(tz="UTC"))
    patterns = harmonics.projections(frames["H4"], now)
    status = harmonics.track(patterns[-1], h1, now) if patterns else None
    off = float(cfg["przesunięcie"])
    short_available = archive.path_for(inst.ticker, "M5").exists()
    options = (*data.INTERVALS, *data.CHART_INTRADAY) if short_available else data.INTERVALS
    interval = st.segmented_control("Interwał wykresu archiwalnego", options, default="H1",
                                    key=f"archived_interval_{inst.key}") or "H1"
    chart_frame = frames.get(interval)
    if interval in data.CHART_INTRADAY:
        try:
            chart_frame = engine.load_short_chart(inst.ticker, interval)
        except (OSError, ValueError) as exc:
            st.warning(f"Brak zapisanego interwału {interval}: {exc}")
    if chart_frame is not None and not chart_frame.empty:
        levels = [(str(x["nazwa"]), float(x["poziom"])) for x in cfg["poziomy"] if x.get("poziom")]
        outcome, _, _ = assess_for_display(frames, cfg["plan_ustawienia"], now, live=False)
        chart_now = max(now, data.bar_end(interval, chart_frame.index[-1])) if interval in data.CHART_INTRADAY else now
        render_market_chart(inst, chart_frame, interval, bars, off, levels, frames, chart_now,
                            outcome.proposal if not outcome.reason else None, archived=True)
    simulation, candles = st.tabs(["Symulacja historyczna", "Świece i układy harmoniczne"])
    with simulation:
        render_families(inst, cfg, frames)
        render_advisor_validation(inst, cfg, frames)
        render_timing_validation(inst, cfg, frames)
    with candles:
        render_candles(inst, frames, off)
        render_harmonics(inst, cfg, off, float(h1["close"].iloc[-1]) + off, h1, patterns, status, read_only=True)


def render_candles(inst: Instrument, frames: dict[str, pd.DataFrame], off: float) -> None:
    st.subheader("Japońskie formacje świecowe")
    interval = st.segmented_control("Interwał formacji", ("H1", "H4", "D1"), default="H1",
                                    key=f"candle_interval_{inst.key}") or "H1"
    frame = frames[interval]
    last = signals.last_closed(frame, interval)
    if last is None:
        st.info("Brak zamkniętych świec dla tego interwału.")
        return
    closed = frame.loc[:last.name].tail(11)
    rows = []
    for i in range(max(0, len(closed) - 10), len(closed)):
        bar = closed.iloc[i]
        previous = closed.iloc[i - 1] if i else None
        names = signals.candle_patterns(bar, previous)
        rows.append({
            "Zamknięcie (czas polski)": data.bar_end(interval, bar.name).tz_convert(data.TZ_LOCAL).strftime("%d.%m.%Y %H:%M"),
            "Otwarcie": round(float(bar["open"]) + off, inst.decimals),
            "Maksimum": round(float(bar["high"]) + off, inst.decimals),
            "Minimum": round(float(bar["low"]) + off, inst.decimals),
            "Zamknięcie": round(float(bar["close"]) + off, inst.decimals),
            "Formacja": ", ".join(names) or "—",
        })
    st.dataframe(pd.DataFrame(rows[::-1]), hide_index=True, alt="Ostatnie zamknięte świece i wykryte formacje")
    st.caption("Doji, świece z długim cieniem oraz objęcia są oznaczane na podstawie geometrii świec.")


def render_harmonics(
    inst: Instrument, cfg: dict, off: float, price: float, h1: pd.DataFrame,
    patterns: list[harmonics.Projection], status: harmonics.Status | None, *, read_only: bool = False,
) -> None:
    st.subheader("Układ harmoniczny AB=CD")
    if not read_only:
        enabled = bool(cfg.get(harmonics.ALERT_SETTING))
        alerts = st.checkbox("Powiadamiaj o zmianie etapu AB=CD", value=enabled, key=f"abcd_alerts_{inst.key}")
        if alerts != enabled:
            cfg[harmonics.ALERT_SETTING] = alerts
            storage.save_settings(cfg, inst.key)

    if not patterns or status is None:
        st.info("Brak potwierdzonych punktów A, B i C spełniających warunki AB=CD.")
        return
    projection = patterns[-1]
    title, body = harmonics.describe(projection, status, off, inst.decimals)
    st.info(body, title=title)
    lower, upper = projection.zone
    if status.stage == "building":
        distance = max(lower + off - price, price - upper - off, 0.0)
        st.caption(f"Odległość bieżącej ceny od strefy D · {_num(distance, inst.decimals)}.")
    table = pd.DataFrame([
        {"Punkt": label, "Cena": _num(point.price + off, inst.decimals),
         "Świeca H4": point.time.tz_convert(data.TZ_LOCAL).strftime("%d.%m.%Y %H:%M"),
         "Potwierdzenie": point.confirmed_at.tz_convert(data.TZ_LOCAL).strftime("%d.%m.%Y %H:%M")}
        for label, point in zip("ABC", (projection.a, projection.b, projection.c))
    ] + [{"Punkt": "D · projekcja", "Cena": _num(projection.d + off, inst.decimals), "Świeca H4": "—", "Potwierdzenie": "—"}])
    st.dataframe(table, hide_index=True, alt="Punkty i czasy potwierdzenia układu AB=CD")
    st.caption(f"Ostatnia zmiana etapu · {status.at.tz_convert(data.TZ_LOCAL):%d.%m.%Y %H:%M} (czas polski).")

    history = st.expander("Poprzednie układy AB=CD", key=f"abcd_history_{inst.key}", on_change="rerun")
    if history.open:
        with history:
            rows = []
            recent = patterns[-31:]
            for old, next_pattern in reversed(list(zip(recent, recent[1:]))):
                old_status = harmonics.track(old, h1, next_pattern.c.confirmed_at)
                label, _ = harmonics.describe(old, old_status, off, inst.decimals)
                rows.append({
                    "Potwierdzenie C (czas polski)": old.c.confirmed_at.tz_convert(data.TZ_LOCAL).strftime("%d.%m.%Y %H:%M"),
                    "Cel D": _num(old.d + off, inst.decimals),
                    "Stan przed kolejną projekcją": label.removeprefix("AB=CD · "),
                })
            if rows:
                st.dataframe(pd.DataFrame(rows), hide_index=True, alt="Ostatnie projekcje AB=CD na historii świec")
            else:
                st.caption("Brak wcześniejszych układów w dostępnej historii.")


def render(inst: Instrument) -> None:
    k = inst.key
    dec = inst.decimals
    skey = f"settings_{k}"
    if skey not in st.session_state:
        st.session_state[skey] = storage.load_settings(k)
    cfg: dict = st.session_state[skey]

    # ------------------------------------------------------------ nagłówek i panel boczny
    st.title(f"{inst.icon} {inst.name} – panel analizy technicznej")
    st.caption(
        f"Dane: {inst.source} z Yahoo Finance, opóźnione do ok. 10–15 min. W xStation: {inst.xtb_symbol}. "
        "Narzędzie pokazuje dane i sygnały techniczne – nie jest poradą inwestycyjną, decyzje podejmujesz sam."
    )

    step = 0.1 if dec > 1 else 1.0

    def _calibrate() -> None:
        """Callback przycisku Kalibruj - działa przed utworzeniem widżetów, więc może ustawić
        wartość pola przesunięcia (zmiana stanu widżetu po jego utworzeniu rzuca wyjątek)."""
        xtb = float(st.session_state.get(f"calib_price_{k}") or 0)
        if xtb <= 0:
            st.session_state[f"calib_msg_{k}"] = ("warning", "Podaj cenę z xStation większą od zera.")
            return
        try:
            h1_now = get_frames(inst.ticker)["H1"]
        except Exception as exc:
            st.session_state[f"calib_msg_{k}"] = ("error", f"Nie udało się pobrać notowań: {exc}")
            return
        if h1_now.empty:
            st.session_state[f"calib_msg_{k}"] = ("error", "Brak świec H1 do porównania.")
            return
        src = float(h1_now["close"].iloc[-1])
        problem = contracts.offset_problem(round(xtb - src, dec), src)
        if problem:
            st.session_state[f"calib_msg_{k}"] = ("error", f"Cena z xStation {xtb:g}, {inst.ticker} {src:.{dec}f}. {problem}")
            return
        cfg["przesunięcie"] = round(xtb - src, dec)
        cfg["kalibracja"] = {
            "czas": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
            "cena_xtb": xtb,
            "cena_zrodla": round(src, dec),
            "świeca": f"{h1_now.index[-1]:%Y-%m-%d %H:%M}",
        }
        storage.save_settings(cfg, k)
        st.session_state[f"off_{k}"] = cfg["przesunięcie"]
        st.session_state[f"calib_msg_{k}"] = ("success", f"Przesunięcie: {cfg['przesunięcie']:+.{dec}f}")

    with st.sidebar:
        st.header(f"Ustawienia: {inst.name}")
        archive_only = st.checkbox("Czytaj zapisane świece", key=f"archive_only_{k}",
                                   help="Pokaż wykres i test z lokalnego archiwum Parquet.")
        if f"off_{k}" not in st.session_state:
            st.session_state[f"off_{k}"] = float(cfg["przesunięcie"])
        manual_off = st.number_input(
            f"Przesunięcie względem XTB [{'pkt' if dec <= 1 else 'USD'}]",
            step=step,
            key=f"off_{k}",
            help=f"Różnica: cena {inst.xtb_symbol} w xStation minus cena {inst.ticker}. Zmiana zapisuje się od razu.",
        )
        with st.form(f"calib_{k}", border=True):
            st.number_input(
                "Kalibracja: aktualna cena z xStation",
                min_value=0.0,
                value=0.0,
                step=step,
                key=f"calib_price_{k}",
                help="Przesunięcie zostanie wyliczone raz, w chwili kliknięcia, i zapisane razem z czasem obu cen.",
            )
            st.form_submit_button("Kalibruj", icon=":material/tune:", width="stretch", on_click=_calibrate, disabled=archive_only)
        msg = st.session_state.pop(f"calib_msg_{k}", None)
        if msg:
            kind, text = msg
            {"success": st.success, "warning": st.warning, "error": st.error}[kind](text)
        cal = cfg.get("kalibracja") or {}
        if cal:
            st.caption(f"Ostatnia kalibracja: {cal.get('czas', '?')} – xStation {cal.get('cena_xtb')} vs {inst.ticker} {cal.get('cena_zrodla')} (świeca {cal.get('świeca', '?')}).")
        try:
            calibrated = pd.Timestamp(cal["czas"]).date() if cal.get("czas") else None
        except ValueError:
            calibrated = None
        roll = contracts.roll_warning(k, inst.ticker, calibrated, pd.Timestamp.now(tz=data.TZ_LOCAL).date())
        if roll:
            st.warning(roll, icon=":material/event_repeat:")
        bars = st.slider("Liczba świec na wykresie", 60, 600, 200, step=20, key=f"bars_{k}")
        if st.button("Odśwież dane", icon=":material/refresh:", width="stretch", key=f"refresh_{k}"):
            get_frames.clear()
            get_intraday_chart.clear()
            get_quote.clear()

    if k == GOLD.key and not archive_only:
        spot_panel = st.expander("Cena spot złota i porównanie z xStation", key="gold_spot_panel", on_change="rerun")
        if spot_panel.open:
            with spot_panel:
                render_gold_spot()

    if not archive_only:
        frames, _ = monitor_view(inst, {})
        if not frames:
            try:
                frames = get_frames(inst.ticker)
            except Exception as exc:
                st.warning(f"Nie udało się pobrać notowań {inst.ticker}: {exc}", title="Błąd pobierania")
                archive_only = True
    if archive_only:
        try:
            archived = engine.load_archived_frames(inst.ticker)
        except FileNotFoundError:
            st.info("Archiwum tego instrumentu powstanie po pierwszym udanym pobraniu świec D1 i H1.")
        except (OSError, ValueError, KeyError, ImportError) as exc:
            st.error(f"Nie udało się odczytać archiwum {inst.ticker}: {exc}", title="Błąd archiwum")
        else:
            render_archived(inst, cfg, archived, bars)
        st.stop()

    # Dane pomocnicze są opcjonalne - ich brak nie zatrzymuje wykresu ani reguł
    usdpln: float | None
    try:
        usdpln = get_quote(data.TICKER_USDPLN)[0]
    except Exception:
        usdpln = None
    context = []
    for label, t, inv in inst.context:
        try:
            context.append((label, *get_quote(t), inv))
        except Exception:
            context.append((label, None, None, inv))

    if any(frames[i].empty for i in data.INTERVALS):
        st.warning("Źródło zwróciło za mało notowań, aby pokazać panel. Spróbuj odświeżyć dane później.", icon=":material/hourglass_empty:")
        st.stop()

    last = float(frames["H1"]["close"].iloc[-1])
    if manual_off != float(cfg["przesunięcie"]):
        problem = contracts.offset_problem(manual_off, last)
        if problem:
            st.sidebar.error(problem)
        else:
            cfg["przesunięcie"] = manual_off
            storage.save_settings(cfg, k)  # monitor w tle korzysta z tego samego przesunięcia
    off = float(cfg["przesunięcie"])
    saved_problem = contracts.offset_problem(off, last)
    if saved_problem and manual_off == off:
        st.error(f"Zapisane przesunięcie {off:+g} jest nierealnie duże – ceny, poziomy i alerty są przesunięte. "
                 "Skalibruj je ponownie z ceną z xStation.", icon=":material/error:")
    price = last + off
    d1, h1 = frames["D1"].iloc[-1], frames["H1"].iloc[-1]
    levels = [(str(x["nazwa"]), float(x["poziom"])) for x in cfg["poziomy"] if x.get("poziom")]
    pattern_now = pd.Timestamp.now(tz="UTC")
    abcd_patterns = harmonics.projections(frames["H4"], pattern_now)
    abcd = abcd_patterns[-1] if abcd_patterns else None
    abcd_status = harmonics.track(abcd, frames["H1"], pattern_now) if abcd else None

    rules_df = signals.evaluate_rules(cfg["reguły"], frames, off, now=frames["H1"].attrs.get("pobrano"))

    render_live_market(inst, cfg, bars, off, levels, frames)

    # ------------------------------------------------------------ zakładki
    tab_simulation, tab_advisor, tab_candles, tab_fib, tab_journal, tab_rules, tab_risk, tab_notify, tab_replay, tab_edit = st.tabs(
        ["Symulacja", "Doradca", "Świece i układy harmoniczne", "Fibonacci", "Dziennik", "Reguły", "Ryzyko pozycji", "Powiadomienia", "Historia alertów", "Ustawienia poziomów"]
    )

    with tab_simulation:
        render_simulation(inst, cfg)
        render_families(inst, cfg, frames)
        render_advisor_validation(inst, cfg, frames)
        render_timing_validation(inst, cfg, frames)
    with tab_advisor:
        render_live_advisor(inst, cfg, frames, off, context)
        technical = st.expander("Szczegóły techniczne", key=f"technicals_{k}", on_change="rerun")
        if technical.open:
            with technical:
                summary = pd.DataFrame([signals.summarize(frames[i], i) for i in data.INTERVALS])
                st.dataframe(summary, hide_index=True, alt=f"Trend i wskaźniki {inst.name} dla D1, H4 i H1")
                div_rows = [{"Interwał": i, **d} for i in data.INTERVALS for d in ind.detect_divergences(frames[i])]
                if div_rows:
                    st.dataframe(pd.DataFrame(div_rows), hide_index=True, alt="Dywergencje RSI i MACD")
    with tab_candles:
        render_candles(inst, frames, off)
        render_harmonics(inst, cfg, off, price, frames["H1"], abcd_patterns, abcd_status)
        params = cfg["plan_ustawienia"]
        render_rejection_history(inst, frames, float(params["limit_h"]), float(params["koszt_pkt"]))
    with tab_journal:
        render_journal_tab(inst, cfg, frames, off, usdpln)
    with tab_replay:
        render_replay_tab(inst, cfg, frames)

    with tab_rules:
        if rules_df.empty:
            st.caption("Brak reguł.")
        else:
            st.dataframe(
                rules_df,
                hide_index=True,
                alt="Reguły poziomów i ich stan na ostatnich zamkniętych świecach",
                column_config={"Spełniona": st.column_config.CheckboxColumn(width="small")},
            )

    with tab_fib:
        lo_v, lo_t, hi_v, hi_t = ind.auto_fib_swing(frames["D1"])
        top_v, top_t, bot_v, bot_t = ind.auto_fib_decline(frames["D1"])
        st.markdown(
            f"Automatycznie z D1 (ostatni rok): wzrost **{_num(lo_v + off, dec)}** ({lo_t:%Y-%m-%d}) → "
            f"szczyt **{_num(hi_v + off, dec)}** ({hi_t:%Y-%m-%d}), potem spadek do **{_num(bot_v + off, dec)}** ({bot_t:%Y-%m-%d})."
        )
        swings = [
            ("Wzrost do szczytu (auto)", lo_v + off, hi_v + off, True),
            ("Spadek od szczytu (auto) – opory", bot_v + off, top_v + off, False),
        ] + [(str(s["nazwa"]), float(s["dołek"]), float(s["szczyt"]), True) for s in cfg["fibo"]]
        fib_rows = []
        for nm, lo, hi, down in swings:
            if hi <= lo:
                continue
            for ratio, lvl in ind.fib_levels(lo, hi, downtrend=down):
                fib_rows.append({"Fala": nm, "Poziom": ratio, "Cena": round(lvl, dec), "Dystans": round(lvl - price, dec)})
        fib_df = pd.DataFrame(fib_rows, columns=["Fala", "Poziom", "Cena", "Dystans"])
        near = fib_df[fib_df["Dystans"].abs() <= 2 * float(d1["atr"])].sort_values("Cena", ascending=False).reset_index(drop=True)
        st.markdown("**Poziomy w zasięgu 2× ATR D1 od ceny**")
        st.dataframe(near, hide_index=True, alt="Poziomy Fibonacciego blisko bieżącej ceny")
        with st.expander("Wszystkie poziomy"):
            st.dataframe(fib_df, hide_index=True, alt="Wszystkie poziomy Fibonacciego")

    with tab_risk:
        usd_pp = st.number_input(
            "Wartość 1 punktu dla 1 lota [USD]",
            value=float(cfg["usd_za_punkt"]),
            step=1.0,
            key=f"usdpp_{k}",
            help=f"Domyślnie: {inst.lot_note}. Sprawdź w specyfikacji instrumentu w xStation.",
        )
        if usd_pp != cfg["usd_za_punkt"]:
            cfg["usd_za_punkt"] = usd_pp
            storage.save_settings(cfg, k)
        if usdpln is None:
            st.warning("Brak kursu USD/PLN – wyniki w PLN są chwilowo niedostępne. Wykres i reguły działają.", icon=":material/currency_exchange:")
        else:
            pv = risk.point_value_pln(1.0, usdpln, usd_pp)
            st.caption(f"1 lot ≈ {pv:.2f} PLN za punkt. Wynik bez spreadu, swapów i prowizji – dokładne liczby pokazuje xStation.")
        if not cfg["pozycje"]:
            st.caption("Brak pozycji. Dodaj je w zakładce Ustawienia poziomów.")
        elif usdpln is None:
            pass
        else:
            scen = risk.scenario_table(cfg["pozycje"], levels, price, usdpln, usd_pp)
            st.dataframe(scen, hide_index=True, alt="Łączny wynik pozycji przy kluczowych poziomach")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Odległość stop-lossa w ATR H1**")
            st.dataframe(risk.atr_stops(float(h1["atr"]), price), hide_index=True, alt="Stop-loss w wielokrotnościach ATR H1")
        with c2:
            st.markdown("**Odległość stop-lossa w ATR D1**")
            st.dataframe(risk.atr_stops(float(d1["atr"]), price), hide_index=True, alt="Stop-loss w wielokrotnościach ATR D1")

    with tab_notify:
        status = delivery.read_json(notify.DATA_DIR / "monitor_status.json", {})
        mon = notify.load_state(k)
        if not status.get("cykl"):
            st.info(
                "Monitor w tle jeszcze nie zgłosił żadnego cyklu. Uruchom go: `.\\uruchom_monitor.ps1` "
                "albo zadanie Windows „US100 monitor sygnałów”.",
                title="Brak danych z monitora",
                icon=":material/notifications_off:",
            )
        else:
            cycle = pd.Timestamp(status["cykl"])
            age = (pd.Timestamp.now() - cycle).total_seconds() / 60
            with st.container(horizontal=True):
                st.metric("Ostatni cykl monitora", f"{age:.0f} min temu", border=True)
                st.metric("Czeka na ponowienie", str(len(notify.pending(k))), border=True)
                dl = status.get("pobrania", {})
                st.metric("Pobrania (pełne / przyrostowe)", f"{dl.get('pełne', 0)} / {dl.get('przyrostowe', 0)}", border=True)
            if age > 15:
                st.warning(
                    "Monitor nie zgłosił się od ponad 15 minut – mógł zostać zatrzymany. "
                    "Sprawdź zadanie Windows albo dane/monitor.log.",
                    icon=":material/warning:",
                )
            for src, err in (status.get("błędy") or {}).items():
                st.error(f"{src}: {err}", title="Błąd w ostatnim cyklu", icon=":material/error:")
            if mon.get("sprawdzono"):
                st.caption(f"{inst.name} sprawdzony: {mon['sprawdzono'].replace('T', ' ')}.")

        st.markdown(
            "Monitor zgłasza: spełnienie reguły, nową dywergencję RSI/MACD, wyjście RSI z wyprzedania lub "
            "wykupienia (H1, H4), dotknięcie poziomu z listy oraz przeskok poziomu luką (raz na 4 godziny dla poziomu). "
            f"Nieudana wysyłka jest ponawiana co cykl, do {delivery.MAX_ATTEMPTS} prób lub 6 godzin."
        )
        waiting = notify.pending(k)
        if waiting:
            st.markdown("**Czekają na wysyłkę**")
            st.dataframe(
                pd.DataFrame(
                    [{"czas": e["czas"], "tytuł": e["tytuł"], "próby": e.get("próby", 0), **e.get("kanały", {})} for e in waiting]
                ),
                hide_index=True,
                alt="Powiadomienia oczekujące na ponowienie wysyłki",
            )
        log, bad = notify.read_log(instrument=k)
        st.markdown("**Historia**")
        if bad:
            st.warning(f"Pominięto {bad} uszkodzonych wierszy w dane/powiadomienia.jsonl.", icon=":material/warning:")
        if log.empty:
            st.caption("Brak powiadomień.")
        else:
            st.dataframe(log, hide_index=True, alt=f"Historia powiadomień {inst.name} ze statusem wysyłki")

        with st.form(f"telegram_{k}"):
            st.markdown("**Telegram (opcjonalnie, wspólny dla wszystkich instrumentów)**")
            st.caption(
                "Utwórz bota u @BotFather, wklej jego token, napisz do bota dowolną wiadomość, "
                "a chat_id odczytaj z https://api.telegram.org/bot<token>/getUpdates. "
                f"Token jest przechowywany w: {delivery.telegram_storage()}."
            )
            tg = notify.load_telegram()
            token = st.text_input("Token bota", value=tg.get("token", ""), type="password")
            chat_id = st.text_input("chat_id", value=tg.get("chat_id", ""))
            if st.form_submit_button("Zapisz Telegram", icon=":material/save:"):
                where = notify.save_telegram({"token": token.strip(), "chat_id": chat_id.strip()})
                st.toast(f"Zapisano ustawienia Telegrama ({where})", icon=":material/check:")

        if st.button("Wyślij powiadomienie testowe", icon=":material/notifications:", key=f"test_{k}"):
            ev = notify.make_event("Test powiadomień", "Jeśli to widzisz, powiadomienia działają.", k)
            notify.dispatch([ev])
            ch = ev.get("kanały", {})
            st.toast(f"Windows: {ch.get('windows')}, Telegram: {ch.get('telegram')}")

    with tab_edit:
        if cfg["plany"] or cfg["obserwacje wejścia"]:
            legacy = st.expander("Dotychczasowe plany i obserwacje", key=f"legacy_plans_{k}", on_change="rerun")
            if legacy.open:
                with legacy:
                    render_plan_tab(inst, cfg, frames, off, price, levels, usdpln)
        st.caption(f"Zmiany zapisują się w pliku dane/{storage.settings_path(k).name} po kliknięciu Zapisz.")
        with st.form(f"edit_{k}"):
            price_col = st.column_config.NumberColumn(min_value=0.01, required=True)
            st.markdown("**Poziomy na wykresie**")
            poz = st.data_editor(
                _table(cfg["poziomy"], "poziomy"), num_rows="dynamic", hide_index=True, key=f"ed_poz_{k}",
                column_config={"poziom": price_col},
            )
            st.markdown("**Reguły**")
            reg = st.data_editor(
                _table(cfg["reguły"], "reguły"),
                num_rows="dynamic",
                hide_index=True,
                key=f"ed_reg_{k}",
                column_config={
                    "poziom": price_col,
                    "warunek": st.column_config.SelectboxColumn(options=list(storage.CONDITIONS), required=True),
                    "interwał": st.column_config.SelectboxColumn(options=list(data.INTERVALS), required=True),
                },
            )
            st.markdown("**Fale Fibonacciego** (dołek → szczyt)")
            fib_ed = st.data_editor(
                _table(cfg["fibo"], "fibo"), num_rows="dynamic", hide_index=True, key=f"ed_fib_{k}",
                column_config={"dołek": price_col, "szczyt": price_col},
            )
            st.markdown("**Pozycje** (wpisz według „Moich Transakcji” w xStation)")
            pos = st.data_editor(
                _table(cfg["pozycje"], "pozycje"),
                num_rows="dynamic",
                hide_index=True,
                key=f"ed_pos_{k}",
                column_config={
                    "kierunek": st.column_config.SelectboxColumn(options=["long", "short"], required=True),
                    "wolumen": st.column_config.NumberColumn(min_value=0.001, step=0.001, format="%.3f", required=True),
                    "cena otwarcia": price_col,
                },
            )
            if st.form_submit_button("Zapisz", icon=":material/save:"):
                tables = {"poziomy": poz, "reguły": reg, "fibo": fib_ed, "pozycje": pos}
                clean, errors = storage.validate({name: df.to_dict("records") for name, df in tables.items()})
                if errors:
                    st.error("\n".join(f"- {e}" for e in errors), title="Nie zapisano – popraw dane")
                else:
                    cfg.update(clean)
                    storage.save_settings(cfg, k)
                    st.toast("Zapisano ustawienia", icon=":material/check:")
                    st.rerun()
