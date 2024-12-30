#!/usr/bin/env python3

import pandas as pd
import numpy as np
import requests
from typing import Dict, List
from datetime import datetime
import logging
import time
import os

import click  # For CLI

import matplotlib

matplotlib.use("Agg")  # Use a headless backend on macOS to avoid GUI thread issues
import matplotlib.pyplot as plt  # For plotting
import concurrent.futures  # For ThreadPoolExecutor

INDICATOR_LIBRARY_URL = "https://data.worldbank.org/indicator"


class PritchettAnalyzer:
    """
    Analyzer for Lant Pritchett's four-part "smell tests" of development importance.
    Minimal changes: now uses ThreadPoolExecutor (with n_threads) to speed up data fetching,
    plus scatterplots for each test's data.
    """

    def __init__(
        self,
        exclude_singapore=False,
        outdir="~/Documents/pritchett_tests",
    ):
        self.base_url = "https://api.worldbank.org/v2"

        self.outdir = os.path.expanduser(outdir)
        if not os.path.exists(self.outdir):
            os.makedirs(self.outdir, exist_ok=True)

        # ISO 3 -> ISO 2 codes
        self.country_codes = {
            "DNK": "DK",
            "CAN": "CA",
            "DEU": "DE",
            "JPN": "JP",
            "USA": "US",
            "GBR": "GB",
            "FRA": "FR",
            "MLI": "ML",
            "NPL": "NP",
            "BFA": "BF",
            "ETH": "ET",
            "TCD": "TD",
            "KOR": "KR",
            "SGP": "SG",
            "CHN": "CN",
            "VNM": "VN",
            "HTI": "HT",
            "NGA": "NG",
            "COD": "CD",
            "CAF": "CF",
            "SLE": "SL",
            # More developed countries
            "SWE": "SE",
            "NLD": "NL",
            "CHE": "CH",
            "NOR": "NO",
            "AUS": "AU",
            "ITA": "IT",
            # More developing countries
            "BGD": "BD",
            "KHM": "KH",
            "LAO": "LA",
            "MMR": "MM",
            "ZMB": "ZM",
            "TZA": "TZ",
            # More success cases
            "IRL": "IE",
            "ISR": "IL",
            "POL": "PL",
            "CHL": "CL",
            "EST": "EE",
            # More failure cases
            "VEN": "VE",
            "ZWE": "ZW",
            "MDG": "MG",
            "YEM": "YE",
            # Additional reform cases
            "IDN": "ID",
            "BRA": "BR",
            "PER": "PE",
            "GHA": "GH",
            "UGA": "UG",
        }
        # Reverse mapping
        self.country_codes_reverse = {v: k for k, v in self.country_codes.items()}

        # Update test cases
        self.test_cases = {
            "developed": [
                "DNK",
                "CAN",
                "DEU",
                "JPN",
                "USA",
                "GBR",
                "FRA",
                "SWE",
                "NLD",
                "CHE",
                "NOR",
                "AUS",
                "ITA",
            ],
            "developing": [
                "MLI",
                "NPL",
                "BFA",
                "ETH",
                "TCD",
                "BGD",
                "KHM",
                "LAO",
                "MMR",
                "ZMB",
                "TZA",
            ],
            "success_cases": [
                "KOR",
                "SGP",
                "CHN",
                "VNM",
                "IRL",
                "ISR",
                "POL",
                "CHL",
                "EST",
            ],
            "failure_cases": [
                "HTI",
                "NGA",
                "COD",
                "CAF",
                "SLE",
                "VEN",
                "ZWE",
                "MDG",
                "YEM",
            ],
            "reform_cases": {
                "CHN": {"pivot_year": 1978, "expected_direction": "increase"},
                "CIV": {"pivot_year": 1978, "expected_direction": "decrease"},
                "VNM": {"pivot_year": 1986, "expected_direction": "increase"},
                "IDN": {"pivot_year": 1998, "expected_direction": "increase"},
                "BRA": {"pivot_year": 1994, "expected_direction": "increase"},
                "PER": {"pivot_year": 1990, "expected_direction": "increase"},
                "GHA": {"pivot_year": 1983, "expected_direction": "increase"},
                "UGA": {"pivot_year": 1987, "expected_direction": "increase"},
            },
        }

        # Indicators to test
        self.indicators = {
            "SP.URB.TOTL.IN.ZS": "Urban population (% of total)",
            "NY.GDP.PCAP.KD": "GDP per capita (constant 2015 US$)",
            "NV.IND.MANF.ZS": "Manufacturing, value added (% of GDP)",
            "SE.SEC.ENRR": "School enrollment, secondary (% gross)",
            "SL.EMP.SELF.ZS": "Self-employed (% of total employment)",
            "EG.ELC.ACCS.ZS": "Access to electricity (% of population)",
            "IT.NET.USER.ZS": "Internet users (% of population)",
            "SP.POP.SCIE.RD.P6": "Researchers in R&D (per million)",
            "SI.POV.GINI": "GINI index (World Bank estimate)",
            "NY.GDP.PCAP.PP.KD": "GDP per capita, PPP (constant 2017 intl $)",
            "NE.CON.PRVT.ZS": "Household final consumption exp. PPP (% of GDP)",
            # Additional real-world indicators:
            "SP.POP.TOTL": "Total population",
            "SP.DYN.LE00.IN": "Life expectancy at birth (years)",
            "SH.XPD.CHEX.GD.ZS": "Current health expenditure (% of GDP)",
            "MS.MIL.XPND.GD.ZS": "Military expenditure (% of GDP)",
            "BX.KLT.DINV.WD.GD.ZS": "Foreign direct investment, net inflows (% of GDP)",
            "GC.TAX.TOTL.GD.ZS": "Tax revenue (% of GDP)",
            "SE.XPD.TOTL.GD.ZS": "Govt. expenditure on education (% of GDP)",
            "SH.STA.MMRT": "Maternal mortality ratio (per 100,000 live births)",
            "SL.UEM.TOTL.ZS": "Unemployment, total (% of labor force)",
            # Additional indicators:
            "NV.IND.TOTL.ZS": "Industry, value added (% of GDP)",
            "BX.KLT.DINV.CD.WD": "FDI, net inflows (current US$)",
            "NY.ADJ.NNTY.PC.KD": "Adj. net national income pc (const 2015 US$)",
            "GB.XPD.RSDV.GD.ZS": "R&D expenditure (% of GDP)",
            "IP.PAT.RESD": "Patent applications, residents",
            "IP.JRN.ARTC.SC": "Scientific & technical journal articles",
            "TX.VAL.TECH.MF.ZS": "High-tech exports (% of manuf. exports)",
            "BX.GSR.ROYL.CD": "Charges for use of IP, receipts (current US$)",
            "SE.TER.ENRR": "School enrollment, tertiary (% gross)",
            "SH.XPD.CHEX.PC.CD": "Health exp. per capita (current US$)",
            "EN.ATM.CO2E.PC": "CO2 emissions (metric tons per capita)",
            "EG.USE.PCAP.KG.OE": "Energy use (kg oil equiv. per capita)",
            "FB.AST.NPER.ZS": "Bank nonperforming loans (% of total loans)",
            "CM.MKT.LCAP.GD.ZS": "Stock market capitalization to GDP (%)",
            "IC.REG.DURS": "Time required to start a business (days)",
            "GC.DOD.TOTL.GD.ZS": "Central govt. debt (% of GDP)",
            "FS.AST.DOMS.GD.ZS": "Domestic credit provided by financial sector (% of GDP)",
            "FS.AST.PRVT.GD.ZS": "Domestic credit to private sector (% of GDP)",
            "IC.CRD.INFO.XQ": "Credit depth of information index (0=low to 8=high)",
            "SI.DST.FRST.20": "Income share held by lowest 20%",
            "IQ.CPA.PUBS.XQ": "CPIA public sector management and institutions cluster average (1=low to 6=high)",
            "IQ.CPA.ECON.XQ": "CPIA economic management cluster average (1=low to 6=high)",
            "IC.GOV.DURS.ZS": "Time spent dealing with the requirements of government regulations (% of senior management time)",
        }

        # Add inverse indicator classification
        self.inverse_indicators = {
            "SI.POV.GINI": True,  # GINI index (lower is better)
            "IC.REG.DURS": True,  # Time to start business (lower is better)
            "FB.AST.NPER.ZS": True,  # Non-performing loans (lower is better)
            "SH.STA.MMRT": True,  # Maternal mortality ratio (lower is better)
            "SL.UEM.TOTL.ZS": True,  # Unemployment rate (lower is better)
            "EN.ATM.CO2E.PC": True,  # CO2 emissions per capita (lower is better)
            "IC.GOV.DURS.ZS": True,  # Time spent on govt. regulations (lower is better)
        }

        if exclude_singapore:
            if "SGP" in self.test_cases["success_cases"]:
                self.test_cases["success_cases"].remove("SGP")

        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO)

    def _generate_wb_url(self, indicator: str, iso2: str) -> str:
        return f"https://data.worldbank.org/indicator/{indicator}?locations={iso2}"

    def fetch_world_bank_data(
        self, indicator: str, start_year: int = 1960
    ) -> pd.DataFrame:
        try:
            end_year = datetime.now().year
            self.logger.info(f"Fetching '{indicator}' data for {start_year}-{end_year}")

            all_countries = set()
            for group in self.test_cases.values():
                if isinstance(group, list):
                    for iso3 in group:
                        if iso3 in self.country_codes:
                            all_countries.add(self.country_codes[iso3])
                elif isinstance(group, dict):
                    for iso3_sub in group.keys():
                        if iso3_sub in self.country_codes:
                            all_countries.add(self.country_codes[iso3_sub])

            if not all_countries:
                self.logger.warning("No countries to query. Returning empty DataFrame.")
                return pd.DataFrame()

            countries_str = ";".join(sorted(all_countries))
            self.logger.info(f"Requesting data for countries: {countries_str}")

            url = f"{self.base_url}/country/{countries_str}/indicator/{indicator}"

            records = []
            page = 1
            pages = 1
            total_entries = 0
            null_count = 0

            while page <= pages:
                params = {
                    "date": f"{start_year}:{end_year}",
                    "format": "json",
                    "per_page": 1000,
                    "page": page,
                }
                resp = requests.get(url, params=params, timeout=30)
                if resp.status_code != 200:
                    self.logger.error(f"API request failed: {resp.status_code}")
                    break
                data_json = resp.json()
                if not data_json or len(data_json) < 2 or not data_json[1]:
                    break

                if "page" in data_json[0] and "pages" in data_json[0]:
                    page = data_json[0]["page"]
                    pages = data_json[0]["pages"]
                else:
                    break

                for entry in data_json[1]:
                    total_entries += 1
                    val = entry["value"]
                    cid = entry["country"]["id"]
                    if val is not None:
                        try:
                            records.append(
                                {
                                    "country": cid,
                                    "year": int(entry["date"]),
                                    "value": float(val),
                                }
                            )
                        except (ValueError, TypeError):
                            null_count += 1
                    else:
                        null_count += 1

                page += 1

            self.logger.info(
                f"Fetched {len(records)} valid records (of {total_entries} total). null={null_count}"
            )
            if not records:
                return pd.DataFrame()

            df = pd.DataFrame(records)
            df_pivot = df.pivot(index="country", columns="year", values="value")
            df_pivot._wb_indicator_code = indicator
            return df_pivot

        except requests.exceptions.RequestException as e:
            self.logger.error(f"RequestException: {e}")
            return pd.DataFrame()
        except Exception as ex:
            self.logger.error(f"Unexpected error: {ex}")
            return pd.DataFrame()

    def _categorize_test_result(
        self,
        metric: float,
        pass_threshold: float,
        borderline_threshold: float,
        higher_is_better: bool = True,
    ) -> str:
        if np.isnan(metric):
            return "fail"
        if higher_is_better:
            if metric >= pass_threshold:
                return "pass"
            elif metric >= borderline_threshold:
                return "borderline"
            else:
                return "fail"
        else:
            if metric <= pass_threshold:
                return "pass"
            elif metric <= borderline_threshold:
                return "borderline"
            else:
                return "fail"

    #########################################################################
    # TEST 1: CROSS-SECTIONAL
    #########################################################################

    def test_1_cross_sectional(
        self, data: pd.DataFrame, needed_per_group: int = 3
    ) -> Dict:
        audit_data = {}
        if data.empty:
            return {"error": "No data available for Test 1", "audit_data": audit_data}

        years = sorted(data.columns.dropna(), reverse=True)
        best_year = None

        # Get indicator code for inverse check
        indicator_code = self._find_original_indicator(data)
        is_inverse = self.inverse_indicators.get(indicator_code, False)

        dev_iso2_list = [
            self.country_codes[c]
            for c in self.test_cases["developed"]
            if c in self.country_codes and self.country_codes[c] in data.index
        ]
        deving_iso2_list = [
            self.country_codes[c]
            for c in self.test_cases["developing"]
            if c in self.country_codes and self.country_codes[c] in data.index
        ]

        for yr in years:
            dev_data_this_year = [
                iso2 for iso2 in dev_iso2_list if not pd.isna(data.loc[iso2, yr])
            ]
            deving_data_this_year = [
                iso2 for iso2 in deving_iso2_list if not pd.isna(data.loc[iso2, yr])
            ]
            dev_count = len(dev_data_this_year)
            deving_count = len(deving_data_this_year)
            if dev_count >= needed_per_group and deving_count >= needed_per_group:
                best_year = yr
                break

        if not best_year:
            return {
                "error": "No single year with sufficient coverage",
                "audit_data": audit_data,
            }

        dev_vals = [
            data.loc[iso2, best_year]
            for iso2 in dev_iso2_list
            if not pd.isna(data.loc[iso2, best_year])
        ]
        deving_vals = [
            data.loc[iso2, best_year]
            for iso2 in deving_iso2_list
            if not pd.isna(data.loc[iso2, best_year])
        ]

        dev_mean = np.mean(dev_vals)
        deving_mean = np.mean(deving_vals)

        ratio = np.nan
        pass_status = "fail"
        if is_inverse:
            # For inverse indicators, we want developing/developed to be > threshold
            ratio = deving_mean / dev_mean if dev_mean else float("nan")
        else:
            # For regular indicators, we want developed/developing to be > threshold
            ratio = dev_mean / deving_mean if deving_mean else float("nan")

        pass_status = self._categorize_test_result(
            ratio,
            pass_threshold=2.0,
            borderline_threshold=1.5,
            higher_is_better=not is_inverse,  # Invert the comparison for inverse indicators
        )

        audit_data["best_year"] = best_year
        audit_data["developed_countries_used"] = dev_iso2_list
        audit_data["developing_countries_used"] = deving_iso2_list
        audit_data["developed_mean"] = dev_mean
        audit_data["developing_mean"] = deving_mean
        audit_data["ratio"] = ratio

        # For plotting: store each dev & deving country's individual value
        # (We'll do a scatter with x=0 or 1, y=value)
        scatter_records = []
        for iso2 in dev_data_this_year:
            scatter_records.append((iso2, data.loc[iso2, best_year], "Developed"))
        for iso2 in deving_data_this_year:
            scatter_records.append((iso2, data.loc[iso2, best_year], "Developing"))
        audit_data["scatter_records"] = scatter_records

        return {
            "year": best_year,
            "developed_mean": dev_mean,
            "developing_mean": deving_mean,
            "ratio": ratio,
            "pass_status": pass_status,
            "audit_data": audit_data,
        }

    #########################################################################
    # TEST 2: GROWTH RATES CORRELATION
    #########################################################################

    def test_2_growth_rates(self, data: pd.DataFrame, min_years: int = 15) -> Dict:
        audit_data = {}
        if data.empty:
            return {"error": "No data available for Test 2", "audit_data": audit_data}

        indicator_code = self._find_original_indicator(data)
        is_inverse = self.inverse_indicators.get(indicator_code, False)

        all_years = sorted(data.columns.dropna())
        valid_years = [y for y in all_years if data[y].count() >= 10]

        if len(valid_years) < 2:
            return {
                "error": "Not enough yearly coverage for Test 2",
                "audit_data": audit_data,
            }

        start_year = valid_years[0]
        end_year = valid_years[-1]
        if (end_year - start_year) < min_years:
            return {
                "error": f"Time span < {min_years} years ({start_year}-{end_year})",
                "audit_data": audit_data,
            }

        growth_dict = {}
        for iso2 in data.index:
            v_start = (
                data.loc[iso2, start_year] if start_year in data.columns else np.nan
            )
            v_end = data.loc[iso2, end_year] if end_year in data.columns else np.nan
            if not pd.isna(v_start) and not pd.isna(v_end) and v_start != 0:
                yrs_diff = end_year - start_year
                g = (v_end / v_start) ** (1 / yrs_diff) - 1
                growth_dict[iso2] = g

        if not growth_dict:
            return {
                "error": "No countries with valid growth for Test 2",
                "audit_data": audit_data,
            }

        x_growth = pd.Series(growth_dict)

        # fetch GDP per capita for correlation
        gdp_data = self.fetch_world_bank_data("NY.GDP.PCAP.KD", start_year=1960)
        if gdp_data.empty:
            return {
                "error": "No GDP per capita data found for correlation",
                "audit_data": audit_data,
            }

        gdp_growth_dict = {}
        for iso2 in gdp_data.index:
            gs = (
                gdp_data.loc[iso2, start_year]
                if start_year in gdp_data.columns
                else np.nan
            )
            ge = (
                gdp_data.loc[iso2, end_year] if end_year in gdp_data.columns else np.nan
            )
            if not pd.isna(gs) and not pd.isna(ge) and gs != 0:
                yrs_diff = end_year - start_year
                gg = (ge / gs) ** (1 / yrs_diff) - 1
                gdp_growth_dict[iso2] = gg

        if not gdp_growth_dict:
            return {
                "error": "No valid GDP growth for correlation",
                "audit_data": audit_data,
            }

        gdp_growth = pd.Series(gdp_growth_dict)

        common = x_growth.index.intersection(gdp_growth.index)
        if len(common) < 5:
            return {
                "error": "Fewer than 5 overlapping countries for correlation",
                "audit_data": audit_data,
            }
        corr = x_growth[common].corr(gdp_growth[common])

        if is_inverse:
            corr = -corr  # Invert correlation for inverse indicators

        pass_status = self._categorize_test_result(
            corr,
            pass_threshold=0.3,
            borderline_threshold=0.1,
            higher_is_better=True,  # Always true since we inverted correlation if needed
        )
        audit_data["start_year"] = start_year
        audit_data["end_year"] = end_year
        audit_data["correlation"] = corr
        audit_data["num_countries"] = len(common)

        # Also store full x_growth & gdp_growth for these common countries
        scatter_data = []
        for ciso in common:
            scatter_data.append((ciso, x_growth[ciso], gdp_growth[ciso]))
        audit_data["scatter_records"] = scatter_data

        return {
            "correlation": corr,
            "pass_status": pass_status,
            "growth_period": f"{start_year}-{end_year}",
            "num_countries": len(common),
            "audit_data": audit_data,
        }

    #########################################################################
    # TEST 3: HISTORICAL SUCCESS vs. FAILURE
    #########################################################################

    def test_3_historical_success(
        self, data: pd.DataFrame, min_years: int = 15
    ) -> Dict:
        audit_data = {"success": [], "failure": []}
        if data.empty:
            return {"error": "No data available for Test 3", "audit_data": audit_data}

        indicator_code = self._find_original_indicator(data)
        is_inverse = self.inverse_indicators.get(indicator_code, False)

        all_years = sorted(data.columns.dropna())
        coverage_list = []
        for y in all_years:
            s_count = 0
            f_count = 0
            for s_c in self.test_cases["success_cases"]:
                iso2 = self.country_codes.get(s_c, "")
                if iso2 in data.index and not pd.isna(data.loc[iso2, y]):
                    s_count += 1
            for f_c in self.test_cases["failure_cases"]:
                iso2 = self.country_codes.get(f_c, "")
                if iso2 in data.index and not pd.isna(data.loc[iso2, y]):
                    f_count += 1
            coverage_list.append((y, s_count, f_count))

        valid_years = [y for (y, sc, fc) in coverage_list if (sc >= 2 and fc >= 2)]
        if len(valid_years) < 2:
            return {
                "error": "Not enough years with >=2 success & >=2 failure countries",
                "audit_data": audit_data,
            }

        earliest = valid_years[0]
        latest = valid_years[-1]
        if (latest - earliest) < min_years:
            return {
                "error": f"Time span < {min_years} years ({earliest}-{latest})",
                "audit_data": audit_data,
            }

        success_countries = []
        failure_countries = []

        for s_c in self.test_cases["success_cases"]:
            iso2 = self.country_codes.get(s_c, "")
            if iso2 in data.index:
                v_early = (
                    data.loc[iso2, earliest] if earliest in data.columns else np.nan
                )
                v_late = data.loc[iso2, latest] if latest in data.columns else np.nan
                if not pd.isna(v_early) and not pd.isna(v_late):
                    success_countries.append(iso2)
                    audit_data["success"].append(
                        {
                            "iso2": iso2,
                            "iso3": s_c,
                            "earliest_val": v_early,
                            "latest_val": v_late,
                        }
                    )

        for f_c in self.test_cases["failure_cases"]:
            iso2 = self.country_codes.get(f_c, "")
            if iso2 in data.index:
                v_early = (
                    data.loc[iso2, earliest] if earliest in data.columns else np.nan
                )
                v_late = data.loc[iso2, latest] if latest in data.columns else np.nan
                if not pd.isna(v_early) and not pd.isna(v_late):
                    failure_countries.append(iso2)
                    audit_data["failure"].append(
                        {
                            "iso2": iso2,
                            "iso3": f_c,
                            "earliest_val": v_early,
                            "latest_val": v_late,
                        }
                    )

        if len(success_countries) < 2 or len(failure_countries) < 2:
            return {
                "error": "Not enough success/failure overlap at earliest & latest",
                "audit_data": audit_data,
            }

        s_early = np.mean([data.loc[iso2, earliest] for iso2 in success_countries])
        s_late = np.mean([data.loc[iso2, latest] for iso2 in success_countries])
        f_early = np.mean([data.loc[iso2, earliest] for iso2 in failure_countries])
        f_late = np.mean([data.loc[iso2, latest] for iso2 in failure_countries])

        if (
            pd.isna(s_early)
            or pd.isna(s_late)
            or pd.isna(f_early)
            or pd.isna(f_late)
            or f_early == 0
        ):
            return {
                "error": "Invalid or zero data for ratio in success/failure",
                "audit_data": audit_data,
            }

        s_growth = s_late / s_early
        f_growth = f_late / f_early
        if f_growth == 0:
            return {"error": "Failure group ratio is zero", "audit_data": audit_data}

        if is_inverse:
            # For inverse indicators, we want the ratio to be smaller (improvement means decrease)
            ratio = f_growth / s_growth if s_growth else float("nan")
        else:
            # For regular indicators, we want the ratio to be larger (improvement means increase)
            ratio = s_growth / f_growth if f_growth else float("nan")

        pass_status = self._categorize_test_result(
            ratio,
            pass_threshold=1.5,
            borderline_threshold=1.2,
            higher_is_better=not is_inverse,
        )

        audit_data["earliest_year"] = earliest
        audit_data["latest_year"] = latest
        audit_data["success_ratio"] = s_growth
        audit_data["failure_ratio"] = f_growth
        audit_data["final_ratio"] = ratio

        return {
            "success_growth": s_growth,
            "failure_growth": f_growth,
            "ratio": ratio,
            "pass_status": pass_status,
            "period": f"{earliest}-{latest}",
            "audit_data": audit_data,
        }

    def _find_original_indicator(self, data: pd.DataFrame) -> str:
        return getattr(data, "_wb_indicator_code", "UNKNOWN")

    #########################################################################
    # TEST 4: REFORM PERIODS
    #########################################################################

    def test_4_reform_periods(self, data: pd.DataFrame, min_years: int = 5) -> Dict:
        if data.empty:
            return {"error": "No data available for Test 4", "audit_data": {}}

        indicator_code = self._find_original_indicator(data)
        is_inverse = self.inverse_indicators.get(indicator_code, False)

        results = {}
        audit_data_all = {}

        for country, info in self.test_cases["reform_cases"].items():
            if country not in self.country_codes:
                continue
            iso2 = self.country_codes[country]
            if iso2 not in data.index:
                continue

            pivot = info["pivot_year"]
            direction = info["expected_direction"]

            pre_years = [
                y
                for y in data.columns
                if (y < pivot and not pd.isna(data.loc[iso2, y]))
            ]
            post_years = [
                y
                for y in data.columns
                if (y > pivot and not pd.isna(data.loc[iso2, y]))
            ]

            if len(pre_years) < min_years or len(post_years) < min_years:
                audit_data_all[country] = {
                    "pivot_year": pivot,
                    "error": "Insufficient pre/post coverage",
                    "pre_years": pre_years,
                    "post_years": post_years,
                }
                continue

            pre_mean = data.loc[iso2, pre_years].mean()
            post_mean = data.loc[iso2, post_years].mean()

            if pd.isna(pre_mean) or pd.isna(post_mean) or pre_mean == 0:
                audit_data_all[country] = {
                    "pivot_year": pivot,
                    "error": "Invalid pre or post mean",
                    "pre_mean": pre_mean,
                    "post_mean": post_mean,
                }
                continue

            ratio = post_mean / pre_mean
            # Adjust interpretation based on indicator type
            if is_inverse:
                # For inverse indicators, decrease is good
                if direction == "increase":
                    pass_status = self._categorize_test_result(
                        ratio,
                        pass_threshold=1.0,
                        borderline_threshold=0.9,
                        higher_is_better=False,
                    )
                else:
                    pass_status = self._categorize_test_result(
                        ratio,
                        pass_threshold=1.0,
                        borderline_threshold=1.1,
                        higher_is_better=True,
                    )
            else:
                # Original logic for regular indicators
                if direction == "increase":
                    pass_status = self._categorize_test_result(
                        ratio,
                        pass_threshold=1.0,
                        borderline_threshold=0.9,
                        higher_is_better=True,
                    )
                else:
                    pass_status = self._categorize_test_result(
                        ratio,
                        pass_threshold=1.0,
                        borderline_threshold=1.1,
                        higher_is_better=False,
                    )

            results[country] = {
                "pivot_year": pivot,
                "pre_reform_mean": pre_mean,
                "post_reform_mean": post_mean,
                "ratio": ratio,
                "expected_direction": direction,
                "pass_status": pass_status,
            }

            audit_data_all[country] = {
                "pivot_year": pivot,
                "pre_years": pre_years,
                "post_years": post_years,
                "pre_mean": pre_mean,
                "post_mean": post_mean,
                "ratio": ratio,
                "direction": direction,
                "passed": (pass_status == "pass"),
            }

        if not results:
            return {
                "error": "No valid reform period comparisons",
                "audit_data": audit_data_all,
            }
        return {"results": results, "audit_data": audit_data_all}

    #########################################################################
    # RUN ALL ANALYSES
    #########################################################################

    def _process_one_indicator(self, code: str, name: str) -> (str, Dict):
        data = self.fetch_world_bank_data(code, start_year=1960)

        # SAVE raw data (if any)
        if not data.empty:
            csv_filename = f"{code}.csv"
            data.to_csv(os.path.join(self.outdir, csv_filename))
            self.logger.info(f"Saved raw data to {csv_filename}")
            # Keep your existing overall time-series plot
            # self._plot_indicator_data(code, name, data)
            self.plot_with_error_handling(self._plot_indicator_data, code, name, data)

        if data.empty:
            self.logger.warning(f"No data for {code}.")
            return code, {
                "name": name,
                "test1": {"error": "No data"},
                "test2": {"error": "No data"},
                "test3": {"error": "No data"},
                "test4": {"error": "No data"},
                "overall_score": 0,
                "test1_score": 0.0,
                "test2_score": 0.0,
                "test3_score": 0.0,
                "test4_score": 0.0,
            }

        # Run tests
        t1 = self.test_1_cross_sectional(data)
        t2 = self.test_2_growth_rates(data)
        t3 = self.test_3_historical_success(data)
        t4 = self.test_4_reform_periods(data)

        # Possibly do scatter/barcharts for each test
        if "error" not in t1:
            self._plot_test1_scatter(code, name, t1["audit_data"])
        if "error" not in t2:
            self._plot_test2_scatter(code, name, t2["audit_data"])
        if "error" not in t3:
            self._plot_test3_scatter(code, name, t3["audit_data"])
        if "error" not in t4:
            self._plot_test4_scatter(code, name, t4["audit_data"])

        def get_score(td: Dict) -> float:
            if "pass_status" in td:
                if td["pass_status"] == "pass":
                    return 1.0
                elif td["pass_status"] == "borderline":
                    return 0.5
            return 0.0

        t1_score = get_score(t1)
        t2_score = get_score(t2)
        t3_score = get_score(t3)
        # t4_score = 0.0
        # if isinstance(t4, dict) and "results" in t4:
        #     pass_list = [r["pass_status"] for r in t4["results"].values()]
        #     if "pass" in pass_list:
        #         t4_score = 1.0
        #     elif "borderline" in pass_list:
        #         t4_score = 0.5

        # Modified scoring for Test 4 to require majority passing
        t4_score = 0.0
        if isinstance(t4, dict) and "results" in t4:
            pass_list = [r["pass_status"] for r in t4["results"].values()]
            total_cases = len(pass_list)
            if total_cases > 0:
                passes = pass_list.count("pass")
                borderlines = pass_list.count("borderline")
                # Require more than 50% clear passes for full score
                if passes > total_cases / 2:
                    t4_score = 1.0
                # Or more than 50% combined passes and borderlines for half score
                elif (passes + borderlines) > total_cases / 2:
                    t4_score = 0.5

        total_score = t1_score + t2_score + t3_score + t4_score

        result_dict = {
            "name": name,
            "test1": t1,
            "test2": t2,
            "test3": t3,
            "test4": t4,
            "overall_score": total_score,
            "test1_score": t1_score,
            "test2_score": t2_score,
            "test3_score": t3_score,
            "test4_score": t4_score,
        }

        time.sleep(1)
        return code, result_dict

    def run_comprehensive_analysis(self, n_threads: int = 4) -> Dict:
        final_results = {}
        self._print_test_explanations()

        with concurrent.futures.ThreadPoolExecutor(max_workers=n_threads) as executor:
            futures = []
            for code, name in self.indicators.items():
                futures.append(executor.submit(self._process_one_indicator, code, name))

            for future in concurrent.futures.as_completed(futures):
                code, res = future.result()
                final_results[code] = res

        return final_results

    def _print_test_explanations(self):
        print("\n")
        print("=== Lant Pritchett's 4-Part Smell Tests for Development Importance ===")
        print(
            "1) Cross-Sectional:\n   - Developed vs. developing (pick the most recent year)."
        )
        print(
            "2) Growth Correlation:\n   - Over time, correlation with GDP per capita."
        )
        print(
            "3) Historical Success vs. Failure:\n   - 'Success' bigger increases than 'failures'."
        )
        print(
            "4) Reform Pivot:\n   - Major pivot => changes in X should reflect direction."
        )

    def _plot_test1_scatter(self, code: str, indicator_name: str, audit_data: Dict):
        """Modified Test1 scatter plot with controlled size."""
        if "scatter_records" not in audit_data or not audit_data["scatter_records"]:
            return

        fig, ax = plt.subplots(figsize=(8, 6))
        recs = audit_data["scatter_records"]

        xs, ys, labels = [], [], []
        for (iso2, val, grp) in recs:
            xval = 0 if grp == "Developed" else 1
            xs.append(xval)
            ys.append(val)
            labels.append(f"{iso2}")  # Simplified labels

        ax.scatter(xs, ys, alpha=0.7, c="tab:blue")

        # Add labels with controlled overlap
        for i, lab in enumerate(labels):
            ax.annotate(
                lab,
                (xs[i], ys[i]),
                fontsize=7,
                xytext=(5, 5),
                textcoords="offset points",
            )

        ax.set_title(
            f"{indicator_name}\nDev vs. Deving (year={audit_data.get('best_year')})"
        )
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Developed", "Developing"])
        ax.set_ylabel(indicator_name)

        outpath = os.path.join(self.outdir, f"{code}_test1.png")
        fig.tight_layout()
        fig.savefig(outpath, dpi=100, bbox_inches="tight")
        plt.close(fig)
        self.logger.info(f"Saved test1 scatter to {outpath}")

    def _plot_test2_scatter(self, code: str, indicator_name: str, audit_data: Dict):
        """Modified Test2 growth correlation scatter with controlled size."""
        if "scatter_records" not in audit_data or not audit_data["scatter_records"]:
            return

        fig, ax = plt.subplots(figsize=(8, 6))
        recs = audit_data["scatter_records"]

        xs, ys, labels = [], [], []
        for (iso2, xg, gg) in recs:
            xs.append(xg)
            ys.append(gg)
            labels.append(iso2)

        ax.scatter(xs, ys, alpha=0.7, c="tab:blue")

        # Add labels with controlled overlap using adjustText if available
        try:
            from adjustText import adjust_text

            texts = []
            for i, lab in enumerate(labels):
                texts.append(ax.annotate(lab, (xs[i], ys[i]), fontsize=7))
            adjust_text(texts, arrowprops=dict(arrowstyle="->", color="gray", lw=0.5))
        except ImportError:
            # Fallback to basic annotations if adjustText not available
            for i, lab in enumerate(labels):
                ax.annotate(
                    lab,
                    (xs[i], ys[i]),
                    fontsize=7,
                    xytext=(5, 5),
                    textcoords="offset points",
                )

        corr = audit_data.get("correlation", float("nan"))
        ax.set_title(f"{indicator_name} vs. GDP Growth\nCorrelation: {corr:.3f}")
        ax.set_xlabel(f"{indicator_name} Growth Rate")
        ax.set_ylabel("GDP per Capita Growth Rate")

        # Add gridlines for better readability
        ax.grid(True, linestyle="--", alpha=0.3)

        outpath = os.path.join(self.outdir, f"{code}_test2.png")
        fig.tight_layout()
        fig.savefig(outpath, dpi=100, bbox_inches="tight")
        plt.close(fig)
        self.logger.info(f"Saved test2 scatter to {outpath}")

    def _plot_test3_scatter(self, code: str, indicator_name: str, audit_data: Dict):
        """Modified Test3 historical success/failure scatter with controlled size."""
        success_recs = audit_data.get("success", [])
        failure_recs = audit_data.get("failure", [])

        if not success_recs and not failure_recs:
            return

        fig, ax = plt.subplots(figsize=(8, 6))

        # Plot success cases
        s_early, s_late, s_labels = [], [], []
        for srec in success_recs:
            s_early.append(srec["earliest_val"])
            s_late.append(srec["latest_val"])
            s_labels.append(f"{srec['iso2']}")

        ax.scatter(s_early, s_late, c="green", alpha=0.7, label="Success")
        for i, lab in enumerate(s_labels):
            ax.annotate(
                lab,
                (s_early[i], s_late[i]),
                fontsize=7,
                xytext=(5, 5),
                textcoords="offset points",
            )

        # Plot failure cases
        f_early, f_late, f_labels = [], [], []
        for frec in failure_recs:
            f_early.append(frec["earliest_val"])
            f_late.append(frec["latest_val"])
            f_labels.append(f"{frec['iso2']}")

        ax.scatter(f_early, f_late, c="red", alpha=0.7, label="Failure")
        for i, lab in enumerate(f_labels):
            ax.annotate(
                lab,
                (f_early[i], f_late[i]),
                fontsize=7,
                xytext=(5, 5),
                textcoords="offset points",
            )

        # Add diagonal line for reference
        if s_early and f_early:
            min_val = min(min(s_early + f_early), min(s_late + f_late))
            max_val = max(max(s_early + f_early), max(s_late + f_late))
            ax.plot([min_val, max_val], [min_val, max_val], "k--", alpha=0.3)

        eyr = audit_data.get("earliest_year")
        lyr = audit_data.get("latest_year")
        ax.set_title(f"{indicator_name}\nSuccess vs. Failure ({eyr} → {lyr})")
        ax.set_xlabel(f"Earliest Value ({eyr})")
        ax.set_ylabel(f"Latest Value ({lyr})")
        ax.legend()

        # Add gridlines
        ax.grid(True, linestyle="--", alpha=0.3)

        outpath = os.path.join(self.outdir, f"{code}_test3.png")
        fig.tight_layout()
        fig.savefig(outpath, dpi=100, bbox_inches="tight")
        plt.close(fig)
        self.logger.info(f"Saved test3 scatter to {outpath}")

    def _plot_test4_scatter(self, code: str, indicator_name: str, audit_data_all: Dict):
        """Modified Test4 reform period scatter with controlled size."""
        points = []
        for cty, info in audit_data_all.items():
            if "error" not in info:
                pm = info["pre_mean"]
                pom = info["post_mean"]
                ratio = info["ratio"]
                dir_str = "+" if info["direction"] == "increase" else "-"
                points.append((cty, pm, pom, ratio, dir_str))

        if not points:
            return

        fig, ax = plt.subplots(figsize=(8, 6))

        # Separate points by expected direction
        pos_pts = [(c, pm, pom, r) for c, pm, pom, r, d in points if d == "+"]
        neg_pts = [(c, pm, pom, r) for c, pm, pom, r, d in points if d == "-"]

        # Plot points with different colors based on expected direction
        if pos_pts:
            for c, pm, pom, r in pos_pts:
                c_color = "green" if r >= 1 else "orange"
                ax.scatter(pm, pom, c=c_color, alpha=0.7)
                ax.annotate(
                    f"{c}",
                    (pm, pom),
                    fontsize=7,
                    xytext=(5, 5),
                    textcoords="offset points",
                )

        if neg_pts:
            for c, pm, pom, r in neg_pts:
                c_color = "green" if r <= 1 else "orange"
                ax.scatter(pm, pom, c=c_color, alpha=0.7)
                ax.annotate(
                    f"{c}",
                    (pm, pom),
                    fontsize=7,
                    xytext=(5, 5),
                    textcoords="offset points",
                )

        # Add diagonal reference line
        all_vals = [
            v for _, v1, v2, _, _ in points for v in (v1, v2) if not np.isnan(v)
        ]
        if all_vals:
            min_val = min(all_vals)
            max_val = max(all_vals)
            ax.plot([min_val, max_val], [min_val, max_val], "k--", alpha=0.3)

        ax.set_title(f"{indicator_name}\nPre vs. Post Reform Values")
        ax.set_xlabel("Pre-reform Mean")
        ax.set_ylabel("Post-reform Mean")

        # Add gridlines
        ax.grid(True, linestyle="--", alpha=0.3)

        outpath = os.path.join(self.outdir, f"{code}_test4.png")
        fig.tight_layout()
        fig.savefig(outpath, dpi=100, bbox_inches="tight")
        plt.close(fig)
        self.logger.info(f"Saved test4 scatter to {outpath}")

    def _print_compact_table(
        self, headers: List[str], rows: List[List], indent: int = 4
    ):
        col_widths = [max(len(str(hdr)), 10) for hdr in headers]
        for row in rows:
            for i, val in enumerate(row):
                col_widths[i] = max(col_widths[i], len(str(val)) + 2)

        fmt = " " * indent + " | ".join(f"{{:{w}}}" for w in col_widths)

        print(fmt.format(*headers))
        print(" " * indent + "-+-".join("-" * w for w in col_widths))
        for row in rows:
            print(fmt.format(*row))

    def _plot_indicator_data(self, code: str, name: str, data: pd.DataFrame):
        """Modified time-series plot with better control over figure size and data points."""
        try:
            # Create figure with fixed size
            fig, ax = plt.subplots(figsize=(12, 8))

            # Get sorted years and handle sampling
            sorted_years = sorted(data.columns)

            # Determine sampling rate based on number of years
            n_years = len(sorted_years)
            if n_years > 30:
                step = max(1, n_years // 30)  # Show maximum 30 points
                sorted_years = sorted_years[::step]

            # Plot each country's data
            legend_handles = []
            for country in data.index:
                yvals = data.loc[country, sorted_years].values
                if not np.all(np.isnan(yvals)):
                    (line,) = ax.plot(
                        sorted_years, yvals, marker="o", markersize=4, label=country
                    )
                    legend_handles.append(line)

            # Set labels and title
            ax.set_title(f"{name} ({code})\nTime Series Analysis", pad=20)
            ax.set_xlabel("Year")
            ax.set_ylabel(name)

            # Rotate x-axis labels for better readability
            plt.xticks(rotation=45)

            # Add legend with smaller font and outside plot
            if legend_handles:
                plt.legend(
                    handles=legend_handles,
                    loc="center left",
                    bbox_to_anchor=(1.05, 0.5),
                    fontsize=8,
                )

            # Adjust layout to prevent label cutoff
            plt.tight_layout()

            # Save with controlled DPI and size
            outpath = os.path.join(self.outdir, f"{code}.png")
            plt.savefig(
                outpath,
                dpi=100,
                bbox_inches="tight",
                pad_inches=0.5,
                facecolor="white",
                edgecolor="none",
            )
            plt.close(fig)
            self.logger.info(f"Saved time-series plot to {outpath}")

        except Exception as e:
            self.logger.error(f"Error plotting time series for {code}: {str(e)}")
            plt.close("all")  # Ensure all figures are closed in case of error

    def plot_with_error_handling(self, plot_func, *args, **kwargs):
        """Enhanced error handling wrapper for plotting functions."""
        try:
            plot_func(*args, **kwargs)
        except Exception as e:
            self.logger.error(f"Error in {plot_func.__name__}: {str(e)}")
            try:
                # Create simple error notification plot
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.text(
                    0.5,
                    0.5,
                    f"Error creating plot:\n{str(e)}",
                    ha="center",
                    va="center",
                    wrap=True,
                    color="red",
                    fontsize=10,
                )
                ax.axis("off")

                # Save error plot
                error_path = os.path.join(self.outdir, f"error_{args[0]}.png")
                plt.savefig(error_path, bbox_inches="tight", dpi=100, facecolor="white")
                plt.close(fig)

            except Exception as nested_error:
                self.logger.error(f"Could not create error plot: {str(nested_error)}")
            finally:
                plt.close("all")  # Ensure cleanup

    def print_detailed_results(self, results: Dict):
        print("\n=== DETAILED RESULTS ===")
        for code, detail in results.items():
            print(f"\nIndicator: {code} - {detail['name']}")
            # Show partial scores for each test, plus total
            print(
                f"  Test1_Score={detail['test1_score']}  Test2_Score={detail['test2_score']}  "
                f"Test3_Score={detail['test3_score']}  Test4_Score={detail['test4_score']}"
            )
            print(
                f"  Overall Score: {detail['overall_score']} / 4 (0.5 for borderline)"
            )

            for i, tk in enumerate(["test1", "test2", "test3", "test4"], start=1):
                tval = detail[tk]
                print(f"\n  Test {i}:")
                if "error" in tval:
                    print(f"    * ERROR => {tval['error']}")
                    continue

                pass_status = tval.get("pass_status", "N/A")
                print(f"    * pass_status => {pass_status}")
                if tk == "test1" and "audit_data" in tval:
                    ad = tval["audit_data"]
                    if ad:
                        print("    -- Cross-Section Audit (Tabular) --")
                        hdr = ["BestYear", "DevMean", "Dev-ingMean", "Ratio"]
                        row_data = [
                            [
                                ad.get("best_year"),
                                round(ad.get("developed_mean", 0), 2),
                                round(ad.get("developing_mean", 0), 2),
                                round(ad.get("ratio", 0), 2),
                            ]
                        ]
                        self._print_compact_table(hdr, row_data, indent=6)

                elif tk == "test2" and "audit_data" in tval:
                    ad = tval["audit_data"]
                    if ad:
                        print("    -- Growth Corr Audit (Tabular) --")
                        hdr = ["StartYr", "EndYr", "Correlation", "Countries"]
                        row_data = [
                            [
                                ad.get("start_year"),
                                ad.get("end_year"),
                                round(ad.get("correlation", 0), 3),
                                ad.get("num_countries"),
                            ]
                        ]
                        self._print_compact_table(hdr, row_data, indent=6)
                        if "samples (X, GDPpc)" in ad:
                            print("      Sample data (up to 5 countries):")
                            hdr2 = ["Country", "X Growth", "GDP pc Growth"]
                            sample_rows = []
                            for k_c, v_c in ad["samples (X, GDPpc)"].items():
                                xg, gg = v_c
                                sample_rows.append([k_c, round(xg, 4), round(gg, 4)])
                            self._print_compact_table(hdr2, sample_rows, indent=8)

                elif tk == "test3" and "audit_data" in tval:
                    ad = tval["audit_data"]
                    if ad:
                        print("    -- Historical S/F Audit (Tabular) --")
                        eyr = ad.get("earliest_year")
                        lyr = ad.get("latest_year")
                        ratio_str = round(ad.get("final_ratio", 0), 2)
                        print(f"      Period: {eyr}-{lyr}  ratio={ratio_str}")
                        if "success" in ad and ad["success"]:
                            print("      Success countries:")
                            hdr_s = ["ISO2", "ISO3", "EarliestVal", "LatestVal"]
                            row_s = []
                            for sdata in ad["success"]:
                                row_s.append(
                                    [
                                        sdata["iso2"],
                                        sdata["iso3"],
                                        round(sdata["earliest_val"], 2),
                                        round(sdata["latest_val"], 2),
                                    ]
                                )
                            self._print_compact_table(hdr_s, row_s, indent=8)
                        if "failure" in ad and ad["failure"]:
                            print("      Failure countries:")
                            hdr_f = ["ISO2", "ISO3", "EarliestVal", "LatestVal"]
                            row_f = []
                            for fdata in ad["failure"]:
                                row_f.append(
                                    [
                                        fdata["iso2"],
                                        fdata["iso3"],
                                        round(fdata["earliest_val"], 2),
                                        round(fdata["latest_val"], 2),
                                    ]
                                )
                            self._print_compact_table(hdr_f, row_f, indent=8)

                elif tk == "test4":
                    if isinstance(tval, dict) and "results" in tval:
                        if "error" not in tval:
                            print(f"    * Overall Test4 Data: {tval}")
                            for cty, adt in tval["audit_data"].items():
                                if cty not in tval["results"]:
                                    print(f"    - {cty}: insufficient coverage")
                                    continue
                                sub_pass_status = tval["results"][cty]["pass_status"]
                                if sub_pass_status != "pass":
                                    print(
                                        f"    - Pivot test => {sub_pass_status} for {cty}"
                                    )
                                    hdr = [
                                        "Country",
                                        "Pivot",
                                        "PreMean",
                                        "PostMean",
                                        "Ratio",
                                        "Direction",
                                    ]
                                    row_data = [
                                        [
                                            cty,
                                            adt["pivot_year"],
                                            round(adt["pre_mean"], 2)
                                            if adt["pre_mean"]
                                            else adt["pre_mean"],
                                            round(adt["post_mean"], 2)
                                            if adt["post_mean"]
                                            else adt["post_mean"],
                                            round(adt["ratio"], 2),
                                            adt["direction"],
                                        ]
                                    ]
                                    self._print_compact_table(hdr, row_data, indent=6)
                                else:
                                    pass
                        else:
                            print(f"    ** Error: {tval.get('error')}")
                    else:
                        print(f"    * {tval}")

    def generate_ranking_report(self, results: Dict) -> pd.DataFrame:
        """
        Creates a DataFrame ranking indicators by total tests passed, including whether lower is better.
        """
        rows = []
        for code, detail in results.items():
            # Check if this is an inverse indicator (where lower is better)
            is_inverse = self.inverse_indicators.get(code, False)

            row = {
                "Indicator": code,
                "Name": detail["name"],
                "Lower is Better": "Yes" if is_inverse else "No",
                "Test1_Score": detail["test1_score"],
                "Test2_Score": detail["test2_score"],
                "Test3_Score": detail["test3_score"],
                "Test4_Score": detail["test4_score"],
                "Tests_Passed": detail["overall_score"],
            }
            rows.append(row)

        df = pd.DataFrame(rows)
        df.sort_values("Tests_Passed", ascending=False, inplace=True)
        return df

    def main_run(self, n_threads: int = 4):
        all_results = self.run_comprehensive_analysis(n_threads=n_threads)

        ranking_df = self.generate_ranking_report(all_results)
        print(
            "\n=== RANKING OF INDICATORS BY TESTS PASSED (including 0.5 for borderline) ==="
        )
        if ranking_df.empty:
            print("(No valid data to rank.)")
        else:
            print(ranking_df.to_string(index=False))

        self._print_test_explanations()
        self.print_detailed_results(all_results)

        print(
            "\n=== RANKING OF INDICATORS BY TESTS PASSED (including 0.5 for borderline) ==="
        )
        if ranking_df.empty:
            print("(No valid data to rank.)")
        else:
            print(ranking_df.to_string(index=False))


@click.group()
def cli():
    """CLI entry point for Pritchett Analysis."""
    pass


@cli.command()
@click.option(
    "--exclude-singapore",
    is_flag=True,
    default=False,
    help="Exclude Singapore from success cases",
)
@click.option(
    "--outdir",
    default="~/Documents/pritchett_tests",
    help="Output directory for data and graphs",
)
@click.option(
    "--n-threads",
    default=4,
    type=int,
    help="Number of threads for parallel data fetching",
)
def run_analysis(exclude_singapore, outdir, n_threads):
    """
    Run the Pritchett analysis, saving results to the specified OUTDIR.
    Uses n-threads to fetch data in parallel.
    """
    outdir = os.path.expanduser(outdir)
    analyzer = PritchettAnalyzer(exclude_singapore=exclude_singapore, outdir=outdir)
    analyzer.main_run(n_threads=n_threads)


if __name__ == "__main__":
    cli()
