from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from .features.alpha158 import calculate_alpha158
from .features.alpha360 import calculate_alpha360
from .features.technical import calculate_technicals, calculate_market_information

from .networks.correlation import build_pos_neg_correlation_adjacency_matrix
from .networks.incidence_matrix import (
    align_and_concat_matrices,
    create_incidence_from_sector_industry,
    create_incidence_from_wikidata,
)
from .networks.knowledge_graph import (
    build_wikidata_adjacency_matrix,
    find_ticker_wikidata_mapping,
    get_first_order_relation_indices,
)
from .networks.sector import build_sector_industry_adjacency_matrix

from .utils import eodhd, yahoo, fred
from .utils.storage import load_config, read_stock_dataset


BASE_DIR = Path(__file__).resolve().parent


# =============================================================================
# Context
# =============================================================================

@dataclass
class PipelineContext:
    universe: str
    base_dir: Path = BASE_DIR
    force: bool = False
    verbose: bool = True
    workers: int = 4

    _config: dict | None = None

    @property
    def config(self) -> dict:
        if self._config is None:
            self._config = load_config(self.base_dir / "config.json")
        return self._config

    @property
    def out_dir(self) -> Path:
        path = self.base_dir / "data" / self.universe
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def constituents_file(self) -> Path:
        return self.base_dir / "data" / "constituents" / "eodhd" / f"{self.universe}.csv"


# =============================================================================
# Paths
# =============================================================================

def prices_path(ctx: PipelineContext) -> Path:
    return ctx.out_dir / f"{ctx.universe}.csv"


def info_path(ctx: PipelineContext) -> Path:
    return ctx.out_dir / f"{ctx.universe}_info.csv"


def market_cap_path(ctx: PipelineContext) -> Path:
    return ctx.out_dir / f"{ctx.universe}_market_cap.csv"


def alpha158_path(ctx: PipelineContext) -> Path:
    return ctx.out_dir / f"{ctx.universe}_alpha158.csv"


def alpha360_path(ctx: PipelineContext) -> Path:
    return ctx.out_dir / f"{ctx.universe}_alpha360.csv"


def technicals_path(ctx: PipelineContext) -> Path:
    return ctx.out_dir / f"{ctx.universe}_tech.csv"


def sector_industry_matrix_path(ctx: PipelineContext) -> Path:
    return ctx.out_dir / f"{ctx.universe}_sector_industry_matrix.npz"


def corr_matrix_path(ctx: PipelineContext) -> Path:
    return ctx.out_dir / f"{ctx.universe}_corr_matrix.npz"


def wikidata_json_path(ctx: PipelineContext) -> Path:
    return ctx.out_dir / f"{ctx.universe}_wikidata.json"


def wikidata_matrix_path(ctx: PipelineContext) -> Path:
    return ctx.out_dir / f"{ctx.universe}_wikidata_matrix.npz"


def incidence_matrix_path(ctx: PipelineContext) -> Path:
    return ctx.out_dir / f"{ctx.universe}_inc_matrix.npz"


# =============================================================================
# Shared helpers
# =============================================================================

def _format_dataset(
    data_series: dict[str, pd.DataFrame],
    start_date: str = "2000-01-01",
) -> pd.DataFrame:
    if not data_series:
        raise ValueError("data_series cannot be empty or None")

    complete_dataset = []

    for instrument_name, df in data_series.items():
        instrument_df = df.copy()
        instrument_df["instrument"] = instrument_name
        instrument_df = instrument_df[instrument_df.index > start_date]
        instrument_df.index.name = "date"
        instrument_df = instrument_df.reset_index()
        instrument_df = instrument_df.set_index(["instrument", "date"])
        complete_dataset.append(instrument_df)

    if not complete_dataset:
        raise ValueError(f"No valid data found after filtering for dates > {start_date}")

    return pd.concat(complete_dataset, axis=0).sort_index()


def _format_universal_features(
    data_series: dict[str, pd.DataFrame],
    feature: str,
    start_date: str = "2000-01-01",
) -> pd.DataFrame:
    df = _format_dataset(data_series, start_date)
    df = df[feature]
    return df.unstack(level="instrument")


def get_universe_symbols(ctx: PipelineContext) -> list[str]:
    historical = pd.read_csv(ctx.constituents_file)

    missing_symbols = historical.loc[historical["EODHD"].isnull(), "Name"].unique().tolist()
    if missing_symbols and ctx.verbose:
        print(f"[{ctx.universe}] Missing EODHD codes for: {missing_symbols}")

    return historical.loc[historical["EODHD"].notnull(), "EODHD"].unique().tolist()


def read_prices(ctx: PipelineContext) -> pd.DataFrame:
    return read_stock_dataset(prices_path(ctx))


def read_info(ctx: PipelineContext) -> pd.DataFrame:
    info = pd.read_csv(info_path(ctx))

    if "instrument" not in info.columns:
        raise ValueError(
            f"{info_path(ctx)} must contain an 'instrument' column. "
            "Check the info downloader output."
        )

    return info


# =============================================================================
# Atomic build functions
# =============================================================================

def build_prices(ctx: PipelineContext) -> None:
    symbols = get_universe_symbols(ctx)

    prices_dict, errors = eodhd.download_multi(
        symbols,
        ctx.config.get("eodhd"),
        data_type="price",
        verbose=ctx.verbose,
        p=ctx.workers,
    )

    if errors:
        print(f"[{ctx.universe}] Price download errors: {[err.get('Ticker') for err in errors]}")

    prices = _format_dataset(prices_dict)
    prices.to_csv(prices_path(ctx), index=True)


def build_info(ctx: PipelineContext) -> None:
    symbols = get_universe_symbols(ctx)

    info_dict, errors = eodhd.download_multi(
        symbols,
        ctx.config.get("eodhd"),
        data_type="info",
        verbose=ctx.verbose,
        p=ctx.workers,
    )

    if errors:
        print(f"[{ctx.universe}] Info download errors: {[err.get('Ticker') for err in errors]}")

    frames = []

    for ticker, df in info_dict.items():
        tmp = df.copy()

        if "instrument" not in tmp.columns:
            tmp.insert(0, "instrument", ticker)
        else:
            tmp["instrument"] = tmp["instrument"].fillna(ticker)

        frames.append(tmp)

    if not frames:
        raise RuntimeError(f"[{ctx.universe}] No info data downloaded.")

    info = pd.concat(frames, axis=0, ignore_index=True)
    info.to_csv(info_path(ctx), index=False)


def build_market_cap(ctx: PipelineContext) -> None:
    symbols = get_universe_symbols(ctx)

    market_caps, errors = eodhd.download_multi(
        symbols,
        ctx.config.get("eodhd"),
        data_type="market_cap",
        verbose=ctx.verbose,
        p=ctx.workers,
    )

    if errors:
        print(f"[{ctx.universe}] Market cap download errors: {[err.get('Ticker') for err in errors]}")

    _format_dataset(market_caps).to_csv(market_cap_path(ctx), index=True)


def build_alpha158(ctx: PipelineContext) -> None:
    prices = read_prices(ctx)
    df_alpha158 = calculate_alpha158(prices)
    df_alpha158.to_csv(alpha158_path(ctx), index=True)


def build_alpha360(ctx: PipelineContext) -> None:
    prices = read_prices(ctx)
    df_alpha360 = calculate_alpha360(prices)
    df_alpha360.to_csv(alpha360_path(ctx), index=True)


def build_technicals(ctx: PipelineContext) -> None:
    prices = read_prices(ctx)
    df_techs = calculate_technicals(prices)
    df_techs.to_csv(technicals_path(ctx), index=True)


def build_sector_industry_matrix(ctx: PipelineContext) -> None:
    prices = read_prices(ctx)
    tickers = prices.index.get_level_values("instrument").unique().tolist()

    info = read_info(ctx)

    if not info["instrument"].is_unique:
        raise ValueError(f"[{ctx.universe}] info['instrument'] must be unique.")

    if info["Name"].isnull().any():
        raise ValueError(f"[{ctx.universe}] info['Name'] contains null values.")

    tck_to_sector = info.set_index("instrument")["GicSector"].dropna().to_dict()
    tck_to_industry = info.set_index("instrument")["GicIndustry"].dropna().to_dict()

    adj_mat, rels, tcks = build_sector_industry_adjacency_matrix(
        tickers,
        tck_to_sector,
        tck_to_industry,
        True,
    )

    np.savez_compressed(
        sector_industry_matrix_path(ctx),
        adj_matrix=adj_mat,
        dates=np.nan,
        relations=rels,
        tickers=tcks,
    )


def build_corr_matrix(ctx: PipelineContext) -> None:
    prices = read_prices(ctx)

    features = ["adj_high", "adj_low", "adj_close", "adj_open", "volume"]
    window_size = 20
    threshold = 0.6

    adj_mat, dates, rels, tcks = build_pos_neg_correlation_adjacency_matrix(
        prices,
        features,
        window_size,
        threshold,
    )

    np.savez_compressed(
        corr_matrix_path(ctx),
        adj_matrix=adj_mat,
        dates=pd.to_datetime(dates).values,
        relations=rels,
        tickers=tcks,
    )


def build_wikidata_json(ctx: PipelineContext) -> None:
    info = read_info(ctx)

    if not info["instrument"].is_unique:
        raise ValueError(f"[{ctx.universe}] info['instrument'] must be unique.")

    if info["Name"].isnull().any():
        raise ValueError(f"[{ctx.universe}] info['Name'] contains null values.")

    ticker_to_name = info.set_index("instrument")["Name"].str.strip().to_dict()

    ticker_to_wikidata = find_ticker_wikidata_mapping(
        ticker_to_name,
        verbose=ctx.verbose,
    )

    with open(wikidata_json_path(ctx), "w") as f:
        json.dump(ticker_to_wikidata, f, indent=2)


def build_wikidata_matrix(ctx: PipelineContext) -> None:
    prices = read_prices(ctx)

    with open(wikidata_json_path(ctx), "r") as f:
        ticker_to_wikidata = json.load(f)

    wiki_chk = ctx.base_dir / "data" / "wikidata_chk"
    wiki_chk.mkdir(parents=True, exist_ok=True)

    adj_mat, dates, rels, tcks = build_wikidata_adjacency_matrix(
        prices,
        ticker_to_wikidata,
        str(wiki_chk),
        "MS",
        True,
    )

    np.savez_compressed(
        wikidata_matrix_path(ctx),
        adj_matrix=adj_mat,
        dates=pd.to_datetime(dates).values,
        relations=rels,
        tickers=tcks,
    )


def build_incidence_matrix(ctx: PipelineContext) -> None:
    wiki_date_idx = -1

    with np.load(sector_industry_matrix_path(ctx)) as data:
        sect = {
            "adj_matrix": data["adj_matrix"],
            "relations": data["relations"],
            "tickers": data["tickers"],
        }

    sect_inc_matrix = create_incidence_from_sector_industry(
        adj_matrix=sect["adj_matrix"],
    )

    with np.load(wikidata_matrix_path(ctx)) as data:
        wiki = {
            "adj_matrix": data["adj_matrix"],
            "dates": data["dates"],
            "relations": data["relations"],
            "tickers": data["tickers"],
        }

    first_order_indices = get_first_order_relation_indices(wiki["relations"])

    wiki_inc_matrix = create_incidence_from_wikidata(
        wiki["adj_matrix"][wiki_date_idx],
        first_order_indices,
    )

    inc_matrix, tickers, relations = align_and_concat_matrices(
        matrix1=sect_inc_matrix,
        tickers1=sect["tickers"],
        categories1=sect["relations"],
        matrix2=wiki_inc_matrix,
        tickers2=wiki["tickers"],
        categories2=[f"E{i}" for i in range(wiki_inc_matrix.shape[1])],
    )

    np.savez_compressed(
        incidence_matrix_path(ctx),
        inc_matrix=inc_matrix,
        relations=relations,
        tickers=tickers,
    )


# =============================================================================
# Dependency graph
# =============================================================================

@dataclass
class Target:
    outputs: Callable[[PipelineContext], list[Path]]
    deps: tuple[str, ...]
    build: Callable[[PipelineContext], None]


TARGETS: dict[str, Target] = {
    "prices": Target(
        outputs=lambda ctx: [prices_path(ctx)],
        deps=(),
        build=build_prices,
    ),
    "info": Target(
        outputs=lambda ctx: [info_path(ctx)],
        deps=(),
        build=build_info,
    ),
    "market_cap": Target(
        outputs=lambda ctx: [market_cap_path(ctx)],
        deps=(),
        build=build_market_cap,
    ),
    "alpha158": Target(
        outputs=lambda ctx: [alpha158_path(ctx)],
        deps=("prices",),
        build=build_alpha158,
    ),
    "alpha360": Target(
        outputs=lambda ctx: [alpha360_path(ctx)],
        deps=("prices",),
        build=build_alpha360,
    ),
    "technicals": Target(
        outputs=lambda ctx: [technicals_path(ctx)],
        deps=("prices",),
        build=build_technicals,
    ),
    "sector_industry_matrix": Target(
        outputs=lambda ctx: [sector_industry_matrix_path(ctx)],
        deps=("prices", "info"),
        build=build_sector_industry_matrix,
    ),
    "corr_matrix": Target(
        outputs=lambda ctx: [corr_matrix_path(ctx)],
        deps=("prices",),
        build=build_corr_matrix,
    ),
    "wikidata_json": Target(
        outputs=lambda ctx: [wikidata_json_path(ctx)],
        deps=("info",),
        build=build_wikidata_json,
    ),
    "wikidata_matrix": Target(
        outputs=lambda ctx: [wikidata_matrix_path(ctx)],
        deps=("prices", "wikidata_json"),
        build=build_wikidata_matrix,
    ),
    "inc_matrix": Target(
        outputs=lambda ctx: [incidence_matrix_path(ctx)],
        deps=("sector_industry_matrix", "wikidata_matrix"),
        build=build_incidence_matrix,
    ),
}


def target_exists(ctx: PipelineContext, target_name: str) -> bool:
    target = TARGETS[target_name]
    return all(path.exists() for path in target.outputs(ctx))


def ensure_target(
    ctx: PipelineContext,
    target_name: str,
    visited: set[str] | None = None,
) -> None:
    if visited is None:
        visited = set()

    if target_name in visited:
        return

    if target_name not in TARGETS:
        valid = ", ".join(sorted(TARGETS))
        raise ValueError(f"Unknown target '{target_name}'. Valid targets: {valid}")

    target = TARGETS[target_name]

    for dep in target.deps:
        ensure_target(ctx, dep, visited)

    outputs = target.outputs(ctx)
    should_build = ctx.force or not all(path.exists() for path in outputs)

    if should_build:
        if ctx.verbose:
            print(f"[{ctx.universe}] BUILD {target_name}")

        target.build(ctx)

        missing_outputs = [path for path in outputs if not path.exists()]
        if missing_outputs:
            raise RuntimeError(
                f"[{ctx.universe}] Target '{target_name}' completed but did not create: "
                f"{missing_outputs}"
            )
    else:
        if ctx.verbose:
            print(f"[{ctx.universe}] SKIP  {target_name}")

    visited.add(target_name)


def run_universe_pipeline(
    universe: str,
    targets: list[str] | tuple[str, ...],
    *,
    force: bool = False,
    verbose: bool = True,
    workers: int = 4,
) -> None:
    ctx = PipelineContext(
        universe=universe,
        force=force,
        verbose=verbose,
        workers=workers,
    )

    visited: set[str] = set()

    for target_name in targets:
        ensure_target(ctx, target_name, visited)

# =============================================================================
# Market / nation-level pipeline
# =============================================================================

@dataclass
class MarketContext:
    nation: str
    base_dir: Path = BASE_DIR
    force: bool = False
    verbose: bool = True
    workers: int = 4

    _config: dict | None = None

    @property
    def config(self) -> dict:
        if self._config is None:
            self._config = load_config(self.base_dir / "config.json")
        return self._config

    @property
    def out_dir(self) -> Path:
        path = self.base_dir / "data"
        path.mkdir(parents=True, exist_ok=True)
        return path


def nation_indexes_path(ctx: MarketContext) -> Path:
    return ctx.out_dir / f"{ctx.nation}_indexes.csv"


def nation_market_path(ctx: MarketContext) -> Path:
    return ctx.out_dir / f"{ctx.nation}_market.csv"


def cnnpred_market_path(ctx: MarketContext) -> Path:
    return ctx.out_dir / "cnnpred_market.csv"


def get_nation_index_symbols(nation: str) -> list[str]:
    symbols_by_nation = {
        "us": ["GSPC.INDX", "DJI.INDX", "NDX.INDX"],
        "eu": ["SXXP.INDX", "SX5E.INDX"],
    }

    if nation not in symbols_by_nation:
        valid = ", ".join(symbols_by_nation)
        raise ValueError(f"Unknown nation '{nation}'. Valid values: {valid}")

    return symbols_by_nation[nation]


def build_nation_indexes(ctx: MarketContext) -> None:
    symbols = get_nation_index_symbols(ctx.nation)

    index_series, errors = eodhd.download_multi(
        symbols,
        ctx.config.get("eodhd"),
        data_type="price",
        verbose=ctx.verbose,
        p=ctx.workers,
    )

    if errors:
        print(f"[{ctx.nation}] Index download errors: {[err.get('Ticker') for err in errors]}")

    df = _format_dataset(index_series)
    df.to_csv(nation_indexes_path(ctx), index=True)


def build_nation_market(ctx: MarketContext) -> None:
    symbols = get_nation_index_symbols(ctx.nation)

    df_indexes = read_stock_dataset(nation_indexes_path(ctx))
    df_market = calculate_market_information(df_indexes, symbols)
    df_market.to_csv(nation_market_path(ctx), index=True)


def build_cnnpred_market(ctx: MarketContext) -> None:
    df_commodities = yahoo.download_commodities_returns(
        ["HG=F", "NG=F", "GC=F", "SI=F"]
    )

    downloader = fred.FREDDownloader(api_key=ctx.config.get("fred"))
    df_fred = downloader.download_all_indicators(start_date="2000-01-01")

    world_indices = [
        "GSPC.INDX",
        "IXIC.INDX",
        "DJI.INDX",
        "NYA.INDX",
        "HSI.INDX",
        "000001.SHG",
        "FCHI.INDX",
        "GDAXI.INDX",
        "IWM.US",
        "ISF.LSE",
    ]

    exchange_rate = [
        "USDJPY.FOREX",
        "USDGBP.FOREX",
        "USDCAD.FOREX",
        "USDCNY.FOREX",
        "USDAUD.FOREX",
        "USDNZD.FOREX",
        "USDCHF.FOREX",
        "USDEUR.FOREX",
        "XAUUSD.FOREX",
        "XAGUSD.FOREX",
    ]

    us_dollar_index = ["DXY.INDX"]

    us_companies = [
        "XOM.US",
        "JPM.US",
        "AAPL.US",
        "MSFT.US",
        "GE.US",
        "JNJ.US",
        "WFC.US",
        "AMZN.US",
    ]

    tickers = world_indices + exchange_rate + us_dollar_index + us_companies

    data_series, errors = eodhd.download_multi(
        tickers,
        ctx.config.get("eodhd"),
        data_type="price",
        verbose=ctx.verbose,
        p=ctx.workers,
    )

    if errors:
        print(f"[cnnpred_market] Download errors: {[err.get('Ticker') for err in errors]}")

    df_us_indx_forex = _format_universal_features(data_series, "adj_close")
    df_us_indx_forex = df_us_indx_forex.pct_change(fill_method=None)

    df = pd.concat(
        [df_fred, df_commodities, df_us_indx_forex],
        axis=1,
    ).sort_index(ascending=True)

    df.to_csv(cnnpred_market_path(ctx), index=True)


@dataclass
class MarketTarget:
    outputs: Callable[[MarketContext], list[Path]]
    deps: tuple[str, ...]
    build: Callable[[MarketContext], None]


MARKET_TARGETS: dict[str, MarketTarget] = {
    "indexes": MarketTarget(
        outputs=lambda ctx: [nation_indexes_path(ctx)],
        deps=(),
        build=build_nation_indexes,
    ),
    "market": MarketTarget(
        outputs=lambda ctx: [nation_market_path(ctx)],
        deps=("indexes",),
        build=build_nation_market,
    ),
    "cnnpred_market": MarketTarget(
        outputs=lambda ctx: [cnnpred_market_path(ctx)],
        deps=(),
        build=build_cnnpred_market,
    ),
}


def ensure_market_target(
    ctx: MarketContext,
    target_name: str,
    visited: set[str] | None = None,
) -> None:
    if visited is None:
        visited = set()

    if target_name in visited:
        return

    if target_name not in MARKET_TARGETS:
        valid = ", ".join(sorted(MARKET_TARGETS))
        raise ValueError(f"Unknown market target '{target_name}'. Valid targets: {valid}")

    target = MARKET_TARGETS[target_name]

    for dep in target.deps:
        ensure_market_target(ctx, dep, visited)

    outputs = target.outputs(ctx)
    should_build = ctx.force or not all(path.exists() for path in outputs)

    if should_build:
        if ctx.verbose:
            print(f"[{ctx.nation}] BUILD {target_name}")

        target.build(ctx)

        missing_outputs = [path for path in outputs if not path.exists()]
        if missing_outputs:
            raise RuntimeError(
                f"[{ctx.nation}] Target '{target_name}' completed but did not create: "
                f"{missing_outputs}"
            )
    else:
        if ctx.verbose:
            print(f"[{ctx.nation}] SKIP  {target_name}")

    visited.add(target_name)


def run_market_pipeline(
    nation: str,
    targets: list[str] | tuple[str, ...],
    *,
    force: bool = False,
    verbose: bool = True,
    workers: int = 4,
) -> None:
    ctx = MarketContext(
        nation=nation,
        force=force,
        verbose=verbose,
        workers=workers,
    )

    visited: set[str] = set()

    for target_name in targets:
        ensure_market_target(ctx, target_name, visited)
