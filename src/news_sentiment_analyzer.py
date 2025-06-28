# news_sentiment_analyzer.py

import asyncio
import logging
import time
from datetime import datetime, timedelta, timezone
from typing import Dict, Optional, Tuple, List, Any
import aiohttp
import traceback
import os
import numpy as np
import pandas as pd
import talib
import pytz # Cho investiny timezone
import ccxt

# Try importing investiny, log error if not found
try:
    import investiny
    INVESTINY_AVAILABLE = True
except ImportError:
    INVESTINY_AVAILABLE = False
    logging.error("investiny library not found. Calendar fetching via investiny will fail.")


# --- Đọc API Keys từ biến môi trường ---
# Giả định bot chính đã load .env nếu có
fmp_api_key = os.getenv("FMP_API_KEY")
av_api_key = os.getenv("AV_API_KEY")

# --- Cấu hình đã cập nhật theo yêu cầu ---
DEFAULT_SENTIMENT_CONFIG = {
    # --- Event Weights, Sensitivity, Min Fade Hours ---
    "event_weights": { # Key nên là standardized key
        "US_CPI_MOM": 0.85, "US_CORE_CPI_MOM": 0.8,
        "US_NFP": 0.75, "US_UNEMP_RATE": 0.6,
        "US_GDP_QOQ": 0.5, "US_PCE_MOM": 0.7,
        "US_CORE_PCE_MOM": 0.65, "US_PPI_MOM": 0.4,
        "US_RETAIL_MOM": 0.4, "US_ISM_MAN_PMI": 0.3,
        "FOMC_RATE": 1.0, "FOMC_STMT": 1.0,
        "Default": 0.2,"US_CORE_PPI_MOM": 0.35
    },
    "event_sensitivity": {
        "US_CPI_MOM": 1.5, "US_CORE_CPI_MOM": 1.6,
        "US_NFP": 1.3, "US_UNEMP_RATE": 1.1,
        "US_GDP_QOQ": 1.0,
        "US_PCE_MOM": 1.4, "US_CORE_PCE_MOM": 1.5,
        "US_PPI_MOM": 1.2, "US_CORE_PPI_MOM": 1.3,
        "US_RETAIL_MOM": 1.1, "US_ISM_MAN_PMI": 0.9,
        "FOMC_RATE": 1.0, "FOMC_STMT": 1.0,
        "Default": 0.8
    },
    "event_impact_direction": {
        "US_CPI_MOM": -1, "US_CORE_CPI_MOM": -1,
        "US_UNEMP_RATE": -1,
        "US_PPI_MOM": -1, "US_PCE_MOM": -1, "US_CORE_PCE_MOM": -1,
        "US_NFP": 1, "US_CORE_PPI_MOM": -1,
        "US_GDP_QOQ": 1, "US_RETAIL_MOM": 1, "US_ISM_MAN_PMI": 1,
        "FOMC_RATE": 0, # Context dependent
        "FOMC_STMT": 0, # Qualitative
        "Default": 0
    },
     "event_min_fade_hours": {
        "FOMC_STMT": 12, "FOMC_RATE": 12,
        "US_CPI_MOM": 4, "US_CORE_CPI_MOM": 4,
        "US_NFP": 3, "US_UNEMP_RATE": 3,
        "US_GDP_QOQ": 2,
        "US_PCE_MOM": 3, "US_CORE_PCE_MOM": 3,
        "US_PPI_MOM": 2, "US_CORE_PPI_MOM": 2,
        "US_RETAIL_MOM": 2, "US_ISM_MAN_PMI": 1.5,
        "Default": 1
    },
    # --- Other Configs ---
    "market_reaction_window_minutes": 15,
    "market_reaction_confirmation_threshold": 0.25,
    "market_reaction_strong_contradiction_threshold": 0.4,
    "confidence_volume_spike_ratio": 2.0,
    "confidence_score_weight_consensus": 0.5,
    "confidence_score_weight_price": 0.3,
    "confidence_score_weight_volume": 0.2,
    "fade_atr_reset_threshold": 1.15,
    "fade_atr_lookback": 20,
    "vix_extreme_threshold": 32.0,
    "calendar_fetch_interval_seconds": 3600 * 2,
    "actual_fetch_retry_delay_seconds": 5,
    "actual_fetch_max_attempts": 8,
    "api_timeout_seconds": 15,
    # --- API CONFIGURATION (Updated Providers) ---
    "use_placeholder_data": False, # *** SET TO False FOR REAL APIs ***
    # --- FMP (For Actual/Historical via /stable/economic-indicators) ---
    # "fmp_base_url": "https://financialmodelingprep.com/api/v4", # V4 or stable? Endpoint is defined directly now.
    "actual_data_provider": "fmp",
    # --- Alpha Vantage (For Historical & Market Data) ---
    "av_base_url": "https://www.alphavantage.co/query",
    "historical_data_provider": "alpha_vantage",
    "market_data_1m_provider": "alpha_vantage", # For SPX/DXY
    "vix_provider": "alpha_vantage", # Primary after CCXT check
    # --- investiny (For Calendar) ---
    "economic_calendar_provider": "investiny", # *** SET TO INVESTINY ***
    # --- Fallbacks (Removed/Set to None as per last discussion) ---
    "calendar_fallback_provider": None,
    "historical_fallback_provider": None, # AV is primary now
    # --- Mappings ---
    "event_key_standardization_map": { # Map Raw API/Scraped names -> Standard Keys
        # FMP Raw Names (from /stable/economic-indicators?) -> Standard Key
        "CPI": "US_CPI_MOM", # FMP name for /stable/? Needs verification
        "Total Nonfarm Payroll": "US_NFP",
        "Unemployment Rate": "US_UNEMP_RATE",
        "GDP": "US_GDP_QOQ", # FMP name?
        "Retail Sales": "US_RETAIL_MOM", # FMP name?
        "ISM Manufacturing PMI": "US_ISM_MAN_PMI", # FMP name?
        "Federal Funds Rate": "FOMC_RATE",
        # AV Function Names -> Standard Key
        "CPI": "US_CPI_MOM", # Also used by AV
        "NONFARM_PAYROLL": "US_NFP",
        "UNEMPLOYMENT_RATE": "US_UNEMP_RATE",
        "REAL_GDP": "US_GDP_QOQ",
        "RETAIL_SALES": "US_RETAIL_MOM",
        "MANUFACTURING_PMI": "US_ISM_MAN_PMI",
        "FEDERAL_FUNDS_RATE": "FOMC_RATE",
        # Investiny Event Names -> Standard Key (CRUCIAL MAPPING)
        "CPI MoM": "US_CPI_MOM", # Example, VERIFY actual names from investiny
        "Core CPI MoM": "US_CORE_CPI_MOM",
        "Non-Farm Employment Change": "US_NFP",
        "Unemployment Rate": "US_UNEMP_RATE", # Might be same as FMP
        "GDP QoQ": "US_GDP_QOQ", # Might be different from FMP's annualized
        "Retail Sales MoM": "US_RETAIL_MOM",
        "Core Retail Sales MoM": "US_CORE_RETAIL_MOM", # Add if needed
        "ISM Manufacturing PMI": "US_ISM_MAN_PMI",
        "PPI MoM": "US_PPI_MOM",
        "Core PPI MoM": "US_CORE_PPI_MOM",
        "Fed Interest Rate Decision": "FOMC_RATE",
        "FOMC Statement": "FOMC_STMT",
        # ... Add MANY more mappings based on investiny output ...
    },
    "fmp_indicator_name_map": { # Standard Key -> FMP Name for /stable/economic-indicators
        "US_CPI_MOM": "CPI",
        "US_NFP": "Total Nonfarm Payroll",
        "US_UNEMP_RATE": "Unemployment Rate",
        "US_CORE_PPI_MOM": "Core PPI",
        "US_GDP_QOQ": "GDP",
        "US_RETAIL_MOM": "Retail Sales",
        "US_ISM_MAN_PMI": "ISM Manufacturing PMI", # Verify this name
        "FOMC_RATE": "Federal Funds Rate",
        # ... Add others ...
    },
    "av_function_map_historical": { # Standard Key -> AV Function for Historical
         "US_CPI_MOM": "CPI",
         "US_NFP": "NONFARM_PAYROLL",
         "US_UNEMP_RATE": "UNEMPLOYMENT_RATE",
         "US_GDP_QOQ": "REAL_GDP",
         "US_CORE_PPI_MOM": "PRODUCER_PRICE_INDEX", 
         "US_RETAIL_MOM": "RETAIL_SALES",
         "US_ISM_MAN_PMI": "MANUFACTURING_PMI",
         "FOMC_RATE": "FEDERAL_FUNDS_RATE"
    },
    "av_symbol_mapping_market": { # Standard Symbol -> AV Symbol for Market Data
        "SPX": "SPY",
        "DXY": "UUP",
        "VIX": "^VIX" # Verify if AV GLOBAL_QUOTE accepts this
    }
}


def _safe_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    if value is None: return default
    if isinstance(value, (int, float)): return float(value)
    if isinstance(value, str):
        v = value.strip().replace('%','').replace('K','e3').replace('M','e6').replace('B','e9').replace(',','')
        if not v or v == '-' or v == '.': return default
        try: return float(v)
        except (ValueError, TypeError): return default
    return default


class NewsSentimentAnalyzer:
    # Nhận key AV và FMP
    def __init__(self, bot_instance, config: Dict, av_api_key: Optional[str], fmp_api_key: Optional[str]):
        self.bot = bot_instance
        self._deep_merge_config(DEFAULT_SENTIMENT_CONFIG, config.get("sentiment_config", {}))
        self.config = DEFAULT_SENTIMENT_CONFIG
        self.av_api_key = av_api_key # Store AV key
        self.fmp_api_key = fmp_api_key # Store FMP key
        self.active_event_sentiments: Dict[str, Dict[str, Any]] = {}
        self.historical_stddevs: Dict[str, float] = {}
        self.event_calendar: List[Dict] = []
        self.last_calendar_fetch_time = 0
        self.logger = logging.getLogger(__name__)
        self.http_session = None

        # Validate keys needed based on chosen providers
        if not self.config["use_placeholder_data"]:
            if self.config["economic_calendar_provider"] == "investiny" and not INVESTINY_AVAILABLE:
                 self.logger.error("investiny provider selected for calendar, but library is not installed!")
            if self.config["actual_data_provider"] == "fmp" and not self.fmp_api_key:
                 self.logger.error("FMP provider selected for actual data, but FMP API key is missing!")
            if self.config["historical_data_provider"] == "alpha_vantage" and not self.av_api_key:
                 self.logger.error("Alpha Vantage provider selected for historical data, but AV API key is missing!")
            if self.config["market_data_1m_provider"] == "alpha_vantage" and not self.av_api_key:
                 self.logger.error("Alpha Vantage provider selected for 1m market data, but AV API key is missing!")
            if self.config["vix_provider"] == "alpha_vantage" and not self.av_api_key:
                 self.logger.error("Alpha Vantage provider selected for VIX, but AV API key is missing!")

        if self.config["use_placeholder_data"]:
            self.logger.warning("Sentiment Analyzer is using PLACEHOLDER data.")

    # --- HTTP Session Management and API Request Helper ---
    async def _get_http_session(self): # Keep as before
        if self.http_session is None or self.http_session.closed:
            connector = aiohttp.TCPConnector(limit=20); self.http_session = aiohttp.ClientSession(connector=connector); self.logger.info("Initialized aiohttp ClientSession.")
        return self.http_session
    async def close_http_session(self): # Keep as before
        if self.http_session and not self.http_session.closed: await self.http_session.close(); self.http_session = None; self.logger.info("Closed aiohttp ClientSession.")

    async def _make_api_request(self, url: str, params: Optional[Dict] = None, headers: Optional[Dict] = None, method: str = 'GET') -> Optional[Any]:
        session = await self._get_http_session(); max_retries = 3; base_delay = 1.5; last_exception = None
        request_params = params.copy() if params else {}
        # Add correct API key based on URL
        if "financialmodelingprep.com" in url and self.fmp_api_key: request_params["apikey"] = self.fmp_api_key
        elif "alphavantage.co" in url and self.av_api_key: request_params["apikey"] = self.av_api_key

        for attempt in range(max_retries):
            try:
                http_request_params = {'params': request_params} if method == 'GET' else {'data': request_params}
                async with session.request(method, url, headers=headers, timeout=self.config["api_timeout_seconds"], **http_request_params) as response:
                    if response.status == 429:
                         retry_after = int(response.headers.get("Retry-After", str(base_delay * (3 ** attempt)))); self.logger.warning(f"Rate limit (429) for {url}. Waiting {retry_after}s..."); await asyncio.sleep(retry_after + np.random.uniform(0, 0.5)); continue
                    response.raise_for_status()
                    self.logger.debug(f"API Request to {url} successful (Status: {response.status})")
                    ct = response.content_type or ""
                    if 'application/json' in ct: return await response.json()
                    elif 'text/csv' in ct: return await response.text()
                    else: return await response.text()
            except (aiohttp.ClientResponseError, asyncio.TimeoutError, aiohttp.ClientConnectionError) as e:
                log_func = self.logger.error if attempt == max_retries - 1 else self.logger.warning
                log_func(f"API request failed for {url} ({type(e).__name__}, Attempt {attempt+1}/{max_retries}): {e}")
                last_exception = e
                if isinstance(e, aiohttp.ClientResponseError) and e.status < 500 and e.status != 429: break # Don't retry client errors (except 429 handled above)
                if attempt < max_retries - 1: await asyncio.sleep(base_delay * (1.5 ** attempt) + np.random.uniform(0, base_delay))
                else: break
            except Exception as e: self.logger.error(f"Unexpected API error {url}: {e}", exc_info=True); last_exception = e; break
        self.logger.error(f"API request ultimately failed for {url}. Last error: {last_exception}")
        return None


    # --- Calendar Fetching (Using investiny) ---
    async def _fetch_event_calendar(self) -> List[Dict]:
        provider = self.config["economic_calendar_provider"]
        self.logger.info(f"Fetching event calendar using provider: {provider}")
        calendar_data = None

        if provider == "investiny" and INVESTINY_AVAILABLE:
            try:
                 from_date_str = datetime.now(timezone.utc).strftime('%Y-%m-%d') # Use ISO format? Check investiny docs
                 to_date_str = (datetime.now(timezone.utc) + timedelta(days=7)).strftime('%Y-%m-%d')
                 countries = ['united states']
                 importances = ['high', 'medium']
                 loop = asyncio.get_event_loop()
                 self.logger.info(f"Running investiny.economic_calendar...")
                 # Note: investiny's economic_calendar might be async already? Check its signature. Assume it's blocking for now.
                 # If it's async: raw_data_list = await investiny.economic_calendar(...)
                 raw_data_list = await loop.run_in_executor(
                     None,
                     lambda: investiny.economic_calendar( # Use blocking call in executor
                         countries=countries, importances=importances,
                         # from_date=from_date_str, to_date=to_date_str # Check if investiny supports date ranges
                     )
                 )
                 self.logger.info(f"investiny calendar finished. Processing {len(raw_data_list) if isinstance(raw_data_list, list) else 0} events.")

                 if isinstance(raw_data_list, list) and raw_data_list:
                      # investiny might return list of dicts directly? Process it.
                      calendar_data = self._process_investiny_or_similar_calendar(raw_data_list)
                 else:
                      self.logger.warning("investiny calendar returned no data or invalid format.")
            except ImportError: self.logger.error("investiny library not installed.")
            except Exception as e: self.logger.error(f"Error fetching/processing investiny calendar: {e}", exc_info=True)

        # --- Placeholder ---
        if calendar_data is None and self.config["use_placeholder_data"]:
             self.logger.warning("Using placeholder calendar data.")
             now = datetime.now(timezone.utc); calendar_data = [{'name': 'US CPI MoM', 'event_key': 'US_CPI_MOM', 'importance': 'high', 'forecast': 0.3, 'previous': 0.4, 'release_time': (now + timedelta(minutes=np.random.randint(5,15))).replace(second=0, microsecond=0)}] # Example

        # Filter future events and sort
        now_dt = datetime.now(timezone.utc)
        future_calendar = [event for event in (calendar_data or []) if event.get('release_time') and event['release_time'] > now_dt]
        future_calendar.sort(key=lambda x: x.get('release_time', datetime.max.replace(tzinfo=timezone.utc)))
        return future_calendar

    # --- Generic Calendar Processor (Adapt based on investiny's actual output) ---
    def _process_investiny_or_similar_calendar(self, raw_data: List[Dict]) -> List[Dict]:
        """Processes a list of dicts from investiny or similar source."""
        processed = []
        std_map = self.config.get("event_key_standardization_map", {})
        self.logger.debug(f"Processing {len(raw_data)} raw calendar events.")
        for item in raw_data:
            try:
                # --- ADAPT THESE FIELD NAMES BASED ON INVESTINY OUTPUT ---
                event_name = item.get('event', 'Unknown Event').strip()
                release_time_str = item.get('date') # Or 'datetime' or 'timestamp'?
                importance_raw = item.get('importance', 'low')
                forecast_raw = item.get('forecast')
                previous_raw = item.get('previous')
                country_raw = item.get('country') # Or 'zone'
                currency_raw = item.get('currency')
                # -----------------------------------------------------

                if not release_time_str or not event_name: continue # Skip if essential info missing

                # Parse datetime (critical step, needs actual format from investiny)
                release_dt = pd.to_datetime(release_time_str, utc=True, errors='coerce') # Assume UTC if possible
                if pd.isna(release_dt):
                     # Try other formats or timezone logic if needed
                     self.logger.warning(f"Could not parse release time '{release_time_str}' for {event_name}. Skipping.")
                     continue

                # Standardize key
                event_key = std_map.get(event_name, event_name.upper().replace(" ", "_").replace("(","").replace(")","").replace(".",""))

                # Clean numeric
                forecast_val = _safe_float(forecast_raw)
                previous_val = _safe_float(previous_raw)

                # Standardize importance
                importance = importance_raw.lower() if isinstance(importance_raw, str) and importance_raw.lower() in ['low', 'medium', 'high'] else 'low'

                processed.append({
                    'name': event_name, 'event_key': event_key, 'importance': importance,
                    'forecast': forecast_val, 'previous': previous_val,
                    'release_time': release_dt.to_pydatetime(),
                    'country': country_raw, 'currency': currency_raw, 'actual': None
                })
            except Exception as e:
                 self.logger.warning(f"Could not process calendar item: {item}. Error: {e}", exc_info=True)
        self.logger.debug(f"Successfully processed {len(processed)} calendar events.")
        return processed

    # --- Actual Data Fetching (Using FMP /stable/economic-indicators) ---
    async def _fetch_actual_data(self, fmp_indicator_name: str, event_name: str) -> Optional[Any]:
        provider = self.config["actual_data_provider"]
        if provider != "fmp":
             self.logger.error(f"Actual data fetch configured for unsupported provider: {provider}. Only FMP supported.")
             return None # Only implement FMP for actual as requested

        if not self.fmp_api_key:
            self.logger.error("FMP API key missing, cannot fetch actual data.")
            return None

        self.logger.info(f"Fetching actual for '{event_name}' using FMP name '{fmp_indicator_name}'")
        url = "https://financialmodelingprep.com/stable/economic-indicators"
        today_str = datetime.now(timezone.utc).strftime('%Y-%m-%d')
        yesterday_str = (datetime.now(timezone.utc) - timedelta(days=1)).strftime('%Y-%m-%d')
        params = { "name": fmp_indicator_name, "from": yesterday_str, "to": today_str } # API key added by _make_api_request
        raw_data = await self._make_api_request(url, params=params)

        if isinstance(raw_data, list) and raw_data:
            try:
                latest_entry = raw_data[-1] # Assume sorted chronologically
                actual_value = latest_entry.get("value")
                # Try converting to float, return original if fails (for qualitative)
                float_val = _safe_float(actual_value)
                return float_val if float_val is not None else actual_value
            except (IndexError, KeyError, Exception) as e:
                 self.logger.error(f"Error processing FMP actual response for {event_name}: {e}")
        elif isinstance(raw_data, list): self.logger.warning(f"FMP actual API returned empty list for {fmp_indicator_name}.")
        else: self.logger.error(f"FMP actual fetch failed or returned invalid data for {event_name}.")
        return None

    # --- Historical Data Fetching (Using Alpha Vantage) ---
    async def _fetch_historical_data(self, indicator_key: str) -> Optional[pd.Series]:
        provider = self.config["historical_data_provider"]
        if provider != "alpha_vantage":
             self.logger.error(f"Historical data fetch configured for unsupported provider: {provider}. Only Alpha Vantage supported.")
             return None # Only implement AV as requested

        if not self.av_api_key:
            self.logger.error("Alpha Vantage API key missing, cannot fetch historical data.")
            return None

        self.logger.debug(f"Fetching historical data for {indicator_key} using Alpha Vantage")
        data_series = None
        av_function = self.config.get("av_function_map_historical", {}).get(indicator_key)
        if av_function:
             url = self.config["av_base_url"]
             params = { "function": av_function, "datatype": "json" } # API key added by helper
             if av_function == "CPI": params["interval"] = "monthly"; params["datatype"] = "csv" # CPI often better in CSV
             if av_function == "REAL_GDP": params["interval"] = "quarterly"
             # ... add other function-specific params ...

             raw_data = await self._make_api_request(url, params=params)

             # --- Parse AV Response (JSON or CSV) ---
             if params.get("datatype") == "csv" and isinstance(raw_data, str):
                  try:
                       from io import StringIO
                       df = pd.read_csv(StringIO(raw_data))
                       if 'timestamp' in df.columns and 'value' in df.columns: # Check common CSV columns
                           df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
                           df = df.dropna(subset=['timestamp']).set_index('timestamp').sort_index()
                           df['value'] = pd.to_numeric(df['value'], errors='coerce')
                           data_series = df['value'].dropna()
                       else: self.logger.error(f"Missing columns in AV CSV for {indicator_key}")
                  except Exception as e: self.logger.error(f"Error processing AV historical CSV for {indicator_key}: {e}")
             elif isinstance(raw_data, dict) and "data" in raw_data and isinstance(raw_data["data"], list):
                 if not raw_data["data"]: self.logger.warning(f"AV returned empty 'data' list for {indicator_key}")
                 else:
                     try: # Standard JSON processing
                         df = pd.DataFrame(raw_data["data"])
                         if 'date' in df.columns and 'value' in df.columns:
                             df['date'] = pd.to_datetime(df['date'], errors='coerce')
                             df = df.dropna(subset=['date']).set_index('date').sort_index()
                             df['value'] = df['value'].replace('.', np.nan) # Handle AV's '.'
                             df['value'] = pd.to_numeric(df['value'], errors='coerce')
                             data_series = df['value'].dropna()
                         else: self.logger.error(f"Missing columns in AV JSON for {indicator_key}")
                     except Exception as e: self.logger.error(f"Error processing AV historical JSON for {indicator_key}: {e}", exc_info=True)
             else: self.logger.error(f"Invalid or missing AV historical response for {indicator_key}")
        else: self.logger.warning(f"No AV function mapping for historical key: {indicator_key}")

        # --- Placeholder ---
        if data_series is None and self.config["use_placeholder_data"]:
             # ... (placeholder logic) ...
            if indicator_key not in self.historical_stddevs: self.logger.warning(f"Using placeholder historical data for {indicator_key}"); dates=pd.date_range(end=datetime.now(timezone.utc), periods=120, freq='M'); values=np.random.normal(0,1,120); data_series=pd.Series(values,index=dates)


        if data_series is not None and data_series.empty: data_series = None # Ensure None if empty after processing
        return data_series


    # --- _calculate_historical_stddev (logic unchanged) ---
    async def _calculate_historical_stddev(self, indicator_key: str) -> Optional[float]:
        # ... (Keep previous logic, calls the updated _fetch_historical_data) ...
        if indicator_key in self.historical_stddevs: return self.historical_stddevs[indicator_key]
        historical_data_series = await self._fetch_historical_data(indicator_key); std_dev = None
        if historical_data_series is not None and isinstance(historical_data_series, pd.Series) and not historical_data_series.empty:
            try:
                numeric_data = pd.to_numeric(historical_data_series, errors='coerce').dropna()
                if len(numeric_data) >= 20: std_dev = numeric_data.std()
                else: self.logger.warning(f"Not enough numeric points ({len(numeric_data)}) for StdDev {indicator_key}.")
            except Exception as e: self.logger.error(f"Error calculating StdDev for {indicator_key}: {e}")
        if std_dev is not None and pd.notna(std_dev) and std_dev > 1e-9: self.historical_stddevs[indicator_key] = std_dev; self.logger.info(f"Calculated historical StdDev for {indicator_key}: {std_dev:.4f}"); return std_dev
        else: self.logger.warning(f"Could not calculate valid historical StdDev for {indicator_key}."); return None


    # --- _fetch_market_data_1m (logic unchanged - uses ccxt/AV) ---
    async def _fetch_market_data_1m(self, symbol: str, start_ts: int, limit: int) -> Optional[pd.DataFrame]:
        # ... (Keep previous logic combining ccxt and AV based on symbol/config) ...
        provider = 'ccxt' if '/' in symbol else self.config["market_data_1m_provider"]
        df = None; ohlcv = None
        if provider == 'ccxt':
             # ... (ccxt fetch logic) ...
            if hasattr(self.bot, 'exchange') and self.bot.exchange:
                 try:
                      if symbol in self.bot.exchange.markets: ohlcv = await self.bot.exchange.fetch_ohlcv(symbol, "1m", since=start_ts, limit=limit)
                      else: self.logger.warning(f"Symbol {symbol} not found in ccxt markets.")
                 except Exception as e: self.logger.error(f"Error fetching 1m via ccxt for {symbol}: {e}")
        elif provider == 'alpha_vantage':
             # ... (AV fetch logic using TIME_SERIES_INTRADAY) ...
            if self.av_api_key:
                av_symbol = self.config.get("av_symbol_mapping_market", {}).get(symbol, symbol)
                url = self.config["av_base_url"]; params = {"function": "TIME_SERIES_INTRADAY", "symbol": av_symbol, "interval": "1min", "outputsize": "full", "datatype": "json"}
                raw_data = await self._make_api_request(url, params=params)
                if isinstance(raw_data, dict) and "Time Series (1min)" in raw_data:
                     try:
                          # ... (AV intraday parsing logic -> assigns to ohlcv list) ...
                          ts_data = raw_data["Time Series (1min)"]; ohlcv = []
                          for dt_str, values in ts_data.items():
                              try:
                                   ts = int(pd.to_datetime(dt_str).timestamp() * 1000)
                                   # Check if within desired range (optional, fetch_ohlcv usually handles 'since')
                                   # if ts >= start_ts:
                                   ohlcv.append([ts, float(values['1. open']), float(values['2. high']), float(values['3. low']), float(values['4. close']), float(values['5. volume'])])
                              except (ValueError, KeyError, TypeError): continue # Skip invalid entries
                          ohlcv.sort(key=lambda x: x[0]) # Ensure chronological order
                     except Exception as e: self.logger.error(f"Error parsing AV 1m for {symbol}: {e}")
                else: self.logger.error(f"Invalid AV 1m response for {symbol}")
            else: self.logger.error("AV API key missing for 1m market data.")

        # ... (Process ohlcv list to DataFrame - Keep as before) ...
        if ohlcv:
             try:
                  df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                  df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
                  df = df.set_index('timestamp')
                  df = df.apply(pd.to_numeric, errors='coerce').dropna()
                  if df.empty: df = None
             except Exception as e: self.logger.error(f"Error converting 1m data to DataFrame for {symbol}: {e}"); df = None
        if df is not None: self.logger.debug(f"Processed {len(df)} 1m data points for {symbol}"); return df
        else: self.logger.warning(f"Failed to fetch/process 1m data for {symbol}"); return None


    # --- _get_market_reaction (logic unchanged - calls updated fetcher) ---
    async def _get_market_reaction(self, event_release_time: datetime) -> Tuple[float, float, float, float, float]:
        # ... (Keep previous logic - it now calls the updated _fetch_market_data_1m) ...
        await asyncio.sleep(2); window_minutes=self.config["market_reaction_window_minutes"]; fetch_limit=window_minutes+10
        start_time=event_release_time - timedelta(minutes=3); end_time=event_release_time + timedelta(minutes=window_minutes)
        start_ts=int(start_time.timestamp() * 1000)
        btc_change, spx_change, dxy_change = 0.0, 0.0, 0.0; volume_ratio_btc = 1.0; correlation_score = 0.5
        try:
            tasks = { "BTC": self._fetch_market_data_1m("BTC/USDT", start_ts, fetch_limit), "SPX": self._fetch_market_data_1m("SPX", start_ts, fetch_limit), "DXY": self._fetch_market_data_1m("DXY", start_ts, fetch_limit) }
            results = await asyncio.gather(*tasks.values(), return_exceptions=True); data_dfs = dict(zip(tasks.keys(), results))
            def calculate_change(df: Optional[pd.DataFrame], st_dt: datetime, en_dt: datetime) -> float:
                if not isinstance(df, pd.DataFrame) or df.empty: return 0.0
                try:
                    pb_ser = df[df.index < st_dt]['close'].dropna(); pa_ser = df[df.index <= en_dt]['close'].dropna()
                    if pb_ser.empty or pa_ser.empty: return 0.0
                    pb_val, pa_val = pb_ser.iloc[-1], pa_ser.iloc[-1]
                    return ((pa_val - pb_val) / pb_val) * 100 if pb_val > 1e-9 else 0.0
                except Exception as calc_e: self.logger.error(f"Error calculating change: {calc_e}"); return 0.0
            btc_change = calculate_change(data_dfs.get("BTC"), event_release_time, end_time)
            spx_change = calculate_change(data_dfs.get("SPX"), event_release_time, end_time)
            dxy_change = calculate_change(data_dfs.get("DXY"), event_release_time, end_time)
            # Volume Ratio BTC
            btc_df = data_dfs.get("BTC");
            if isinstance(btc_df, pd.DataFrame) and not btc_df.empty:
                 try:
                      vol_d_df = btc_df[(btc_df.index >= event_release_time)&(btc_df.index <= end_time)]; vol_b_df = btc_df[btc_df.index < event_release_time].tail(window_minutes*2)
                      if not vol_d_df.empty and not vol_b_df.empty:
                           vol_d = vol_d_df['volume'].sum(); avg_vol_b = vol_b_df['volume'].mean()
                           if avg_vol_b > 1e-9: exp_vol_d = avg_vol_b * len(vol_d_df); volume_ratio_btc = max(0.1, vol_d / exp_vol_d) if exp_vol_d > 1e-9 else 1.0
                 except Exception as vol_e: self.logger.error(f"Error calculating vol ratio: {vol_e}")
            # Correlation Score
            if btc_change != 0 and spx_change != 0 and dxy_change != 0:
                 btc_s, spx_s, dxy_s = np.sign(btc_change), np.sign(spx_change), np.sign(dxy_change)
                 if (btc_s == 1 and spx_s == 1 and dxy_s == -1) or (btc_s == -1 and spx_s == -1 and dxy_s == 1): correlation_score = 0.85
                 else: correlation_score = 0.15
            self.logger.debug(f"Market Reaction ({window_minutes}m): BTC={btc_change:.2f}%, SPX={spx_change:.2f}%, DXY={dxy_change:.2f}%, VolR={volume_ratio_btc:.2f}, Corr={correlation_score:.2f}")
            return btc_change, spx_change, dxy_change, volume_ratio_btc, correlation_score
        except Exception as e: self.logger.error(f"Error getting market reaction: {e}", exc_info=True); return 0.0, 0.0, 0.0, 1.0, 0.5


    # --- _calculate_confidence_score (logic unchanged) ---
    def _calculate_confidence_score(self, btc_change: float, spx_change: float, dxy_change: float, volume_ratio_btc: float, corr_score: float) -> float:
        # ... (Keep previous improved logic) ...
        try:
            w_con, w_pri, w_vol = self.config["confidence_score_weight_consensus"], self.config["confidence_score_weight_price"], self.config["confidence_score_weight_volume"]
            tot_w = w_con + w_pri + w_vol; tot_w = 1.0 if tot_w <= 1e-9 else tot_w
            pri_mf = abs(btc_change) / self.config["market_reaction_confirmation_threshold"]; pri_s = np.tanh(pri_mf)
            vol_f = (volume_ratio_btc - 1.0) / max(1e-9, self.config["confidence_volume_spike_ratio"] - 1.0); vol_s = np.tanh(vol_f)
            con_s = np.clip((corr_score - 0.15) / (0.85 - 0.15), 0.0, 1.0)
            raw_c = (con_s * w_con + pri_s * w_pri + vol_s * w_vol) / tot_w; conf_s = np.clip(raw_c, 0.1, 1.0)
            self.logger.debug(f"Confidence Calc: Pri={pri_s:.2f}, Vol={vol_s:.2f}, Con={con_s:.2f} => Conf={conf_s:.2f}")
            return conf_s
        except Exception as e: self.logger.error(f"Error calculating confidence: {e}"); return 0.5


    # --- calculate_sentiment_score (logic unchanged) ---
    async def calculate_sentiment_score(self, event_details: Dict) -> Tuple[Optional[float], Optional[float]]:
        # ... (Keep previous logic - handles qualitative, calls updated _get_market_reaction, _calculate_confidence_score, _get_vix_value) ...
        name=event_details.get("name"); key=event_details.get("event_key"); forecast=event_details.get("forecast"); actual=event_details.get("actual"); release_time=event_details.get("release_time")
        is_qualitative = False; qualitative_score_input = 0.0
        if key in ["FOMC_STMT"] or (actual is None and forecast is None and key != "FOMC_RATE"):
            is_qualitative = True
            sentiment_label = actual.get("sentiment_label") if isinstance(actual, dict) else None

            if sentiment_label == "hawkish":
                qualitative_score_input = -0.7
            elif sentiment_label == "dovish":
                qualitative_score_input = 0.7

            self.logger.info(
                f"Handling qualitative event: {name} ({key}) - Label='{sentiment_label}'"
            )

        elif (
            actual is None
            or forecast is None
            or key is None
            or name is None
            or release_time is None
        ):
            self.logger.warning(f"Skipping quant sentiment for {name}: Missing data.")
            return None, None

        else:
            forecast_f = _safe_float(forecast)
            actual_f = _safe_float(actual)

            if forecast_f is None or actual_f is None:
                self.logger.error(f"Could not convert actual/forecast for {name}")
                return None, None
        try:
            event_weight = self.config["event_weights"].get(key, self.config["event_weights"]["Default"]); confidence_score = 0.5
            if is_qualitative:
                weighted_score = qualitative_score_input * event_weight
                confidence_score = 0.4  # Lower default confidence

            else:
                std_dev = await self._calculate_historical_stddev(key)

                if std_dev is None or std_dev <= 1e-9:
                    return None, None

                deviation = actual_f - forecast_f
                normalized_deviation = deviation / std_dev

                sensitivity = self.config["event_sensitivity"].get(
                    key, self.config["event_sensitivity"]["Default"]
                )

                base_score_raw = np.tanh(normalized_deviation * sensitivity)

                impact_direction = self.config["event_impact_direction"].get(
                    key, self.config["event_impact_direction"]["Default"]
                )

                base_sentiment_score = base_score_raw * impact_direction if impact_direction != 0 else 0.0
                weighted_score = base_sentiment_score * event_weight
            # Market Reaction
            btc_change, spx_change, dxy_change, vol_ratio, corr_score = await self._get_market_reaction(release_time); confidence_score = self._calculate_confidence_score(btc_change, spx_change, dxy_change, vol_ratio, corr_score)
            # Adjust Score based on reaction
            final_sentiment_score = weighted_score; expected_good = weighted_score > 0.05; expected_bad = weighted_score < -0.05; market_on = (corr_score > 0.6 and abs(btc_change) > self.config["market_reaction_confirmation_threshold"] * 0.5); market_off = (corr_score < 0.4 and abs(btc_change) > self.config["market_reaction_confirmation_threshold"] * 0.5)
            if (expected_good and market_off) or (expected_bad and market_on):
                contr_str_f = abs(btc_change) / self.config["market_reaction_strong_contradiction_threshold"]
                self.logger.warning(f"Market reaction contradicts for {name}. InitScore={weighted_score:.3f}.")

                if confidence_score > 0.65 and contr_str_f >= 1.0:
                    self.logger.warning(f"Strong contradiction! Reversing sign for {name}.")
                    final_sentiment_score = -weighted_score * 0.6
                else:
                    self.logger.warning(f"Weak/moderate contradiction. Reducing impact for {name}.")
                    final_sentiment_score *= 0.15

            elif (expected_good and market_on) or (expected_bad and market_off):
                self.logger.info(f"Market reaction confirms for {name}. InitScore={weighted_score:.3f}.")
                final_sentiment_score *= (1.0 + 0.1 * confidence_score)

            else:
                self.logger.info(f"Market reaction unclear for {name}. InitScore={weighted_score:.3f}. Scaling by confidence.")
                final_sentiment_score *= confidence_score
            # VIX Filter
            vix = await self._get_vix_value();
            if vix is not None and vix > self.config["vix_extreme_threshold"]: self.logger.warning(f"VIX high ({vix:.2f}). Reducing final impact for {name}."); final_sentiment_score *= 0.25; confidence_score *= 0.6
            # Final Clip
            final_sentiment_score = np.clip(final_sentiment_score, -event_weight, event_weight)
            self.logger.info(f"Calculated Sentiment for {name} ({key}): Final Score={final_sentiment_score:.3f}, Confidence={confidence_score:.2f}")
            return final_sentiment_score, confidence_score
        except Exception as e: self.logger.error(f"Error calculating sentiment score for {name}: {e}", exc_info=True); return None, None


    # --- _get_vix_value (logic unchanged - tries ccxt then AV) ---
    async def _get_vix_value(self) -> Optional[float]:
        vix_value = None
        # 1. Try CCXT first (if bot has exchange and symbol exists)
        if hasattr(self.bot, 'exchange') and self.bot.exchange:
            # Common VIX symbols (check what your exchange uses)
            vix_symbols_to_try = ['VOL-VIX/USD', 'VIX/USD', '.VIX', 'VIX'] # Add more possibilities
            for vix_sym in vix_symbols_to_try:
                 if vix_sym in self.bot.exchange.markets:
                      try:
                           ticker = await self.bot.exchange.fetch_ticker(vix_sym)
                           if ticker and ticker.get('last') is not None:
                                vix_value = _safe_float(ticker['last'])
                                if vix_value is not None:
                                     self.logger.debug(f"Fetched VIX via ccxt ({vix_sym}): {vix_value:.2f}")
                                     return vix_value
                      except ccxt.NetworkError: break # Stop trying ccxt if network error
                      except Exception as e: self.logger.warning(f"Error fetching VIX '{vix_sym}' via ccxt: {e}")
                 # else: self.logger.debug(f"VIX symbol {vix_sym} not in markets.")

        # 2. If CCXT failed or not configured, try configured provider (e.g., Alpha Vantage)
        if vix_value is None:
             provider = self.config["vix_provider"]
             self.logger.debug(f"Fetching VIX value using provider: {provider}")
             if provider == 'alpha_vantage' and self.av_api_key:
                  # ... (AV fetch logic using GLOBAL_QUOTE as before) ...
                 av_vix_symbol = self.config.get("av_symbol_mapping_market",{}).get("VIX", "^VIX")
                 url=self.config["av_base_url"]; params={"function":"GLOBAL_QUOTE","symbol":av_vix_symbol}
                 raw_data = await self._make_api_request(url, params=params)
                 if isinstance(raw_data, dict) and "Global Quote" in raw_data and raw_data["Global Quote"]:
                      try: price_str = raw_data["Global Quote"].get("05. price"); vix_value = _safe_float(price_str)
                      except Exception as e: self.logger.error(f"Error parsing AV VIX: {e}")
                 else: self.logger.error(f"Invalid AV VIX response for {av_vix_symbol}")
             # --- Add FMP VIX logic here if configured as provider ---
             # elif provider == 'fmp' and self.fmp_api_key: ...
             elif provider == "placeholder" or self.config["use_placeholder_data"]:
                  vix_value = np.random.uniform(15.0, 35.0)
             else: self.logger.error(f"Unsupported or misconfigured VIX provider: {provider}")

        return vix_value

    def _update_sentiment_fadeout(self, current_time: datetime):
        events_to_remove = []
        now_aware = current_time.astimezone(timezone.utc)
        for event_key in list(self.active_event_sentiments.keys()):
            if event_key not in self.active_event_sentiments: continue
            data = self.active_event_sentiments[event_key]; expired = False; min_fade_time = data.get("min_fade_time"); release_time = data.get("release_time")
            if not isinstance(min_fade_time, datetime) or not isinstance(release_time, datetime): events_to_remove.append(event_key); continue
            min_fade_time_aware = min_fade_time.astimezone(timezone.utc); release_time_aware = release_time.astimezone(timezone.utc)
            if now_aware > min_fade_time_aware:
                try:
                    symbol = "BTC/USDT"; tf = "15m"
                    if self.bot and hasattr(self.bot, 'data') and symbol in self.bot.data and tf in self.bot.data[symbol]:
                        df_tf = self.bot.data[symbol][tf]
                        if isinstance(df_tf, pd.DataFrame) and not df_tf.empty and 'ATR' in df_tf.columns and len(df_tf) > self.config['fade_atr_lookback'] + 1:
                            atr_current = df_tf['ATR'].iloc[-1]; df_before_event = df_tf[df_tf.index < release_time_aware]
                            if len(df_before_event) >= self.config['fade_atr_lookback']:
                                 atr_mean_before = df_before_event['ATR'].tail(self.config['fade_atr_lookback']).mean()
                                 if pd.notna(atr_current) and pd.notna(atr_mean_before) and atr_mean_before > 1e-9:
                                     atr_ratio = atr_current / atr_mean_before
                                     if atr_ratio < self.config['fade_atr_reset_threshold']: self.logger.info(f"Sentiment {event_key} faded: ATR ratio ({atr_ratio:.2f})"); expired = True
                                     # else: self.logger.debug(f"Sentiment {event_key} ATR ratio high: {atr_ratio:.2f}")
                                 # else: self.logger.debug(f"Cannot check ATR reset {event_key}: Invalid ATR values.")
                            # else: self.logger.debug(f"Cannot check ATR reset {event_key}: Not enough data before event.")
                        # else: self.logger.debug(f"Cannot check ATR reset {event_key}: Missing ATR/data {symbol} {tf}.")
                    # else: self.logger.debug(f"Cannot check ATR reset {event_key}: Bot market data missing.")
                except Exception as e: self.logger.error(f"Error checking ATR fadeout for {event_key}: {e}")
            if expired: events_to_remove.append(event_key)
        for key in events_to_remove:
            if key in self.active_event_sentiments: del self.active_event_sentiments[key]; self.logger.info(f"Removed expired/faded sentiment: {key}")


    # --- get_active_sentiment_score, get_detailed_active_sentiment (logic unchanged) ---
    def get_active_sentiment_score(self, current_time: datetime) -> float:
        detailed = self.get_detailed_active_sentiment(current_time); return detailed.get("total_score_adj_confidence", 0.0)
    def get_detailed_active_sentiment(self, current_time: datetime) -> Dict:
        self._update_sentiment_fadeout(current_time); total_score_raw=0.0; total_score_adj_confidence=0.0; sum_abs_score_times_confidence=0.0; sum_abs_score=0.0; active_events={}
        for key in list(self.active_event_sentiments.keys()):
             if key not in self.active_event_sentiments: continue; data=self.active_event_sentiments[key]; score=data.get('score',0.0); confidence=data.get('confidence',0.5); weight=data.get('event_weight',0.0); abs_score=abs(score)
             total_score_raw+=score; score_adj_conf=score*confidence; total_score_adj_confidence+=score_adj_conf; sum_abs_score_times_confidence+=abs_score*confidence; sum_abs_score+=abs_score; active_events[key]={'score':score,'confidence':confidence,'weight':weight,'score_adj_conf':score_adj_conf}
        average_confidence=(sum_abs_score_times_confidence/sum_abs_score) if sum_abs_score > 1e-9 else 0.5; average_confidence=np.clip(average_confidence,0.1,1.0); max_abs_raw_score=1.5; max_abs_adj_score=1.0
        final_total_score_raw=np.clip(total_score_raw,-max_abs_raw_score,max_abs_raw_score); final_total_score_adj_confidence=np.clip(total_score_adj_confidence,-max_abs_adj_score,max_abs_adj_score)
        if abs(final_total_score_adj_confidence)>0.05: self.logger.debug(f"Detailed Sentiment: Raw={final_total_score_raw:.3f}, AdjConf={final_total_score_adj_confidence:.3f}, AvgConf={average_confidence:.2f}, Count={len(active_events)}")
        return {"total_score_raw":final_total_score_raw, "total_score_adj_confidence":final_total_score_adj_confidence, "average_confidence":average_confidence, "active_events_count":len(active_events), "active_events_details":active_events}


    # --- process_event_release (logic adjusted for FMP actual source) ---
    async def process_event_release(self, event_detail: Dict):
        release_time = event_detail.get("release_time"); event_key = event_detail.get("event_key"); event_name = event_detail.get("name")
        # Ensure release_time is aware UTC
        if isinstance(release_time, datetime):
             if release_time.tzinfo is None: release_time = release_time.replace(tzinfo=timezone.utc)
             event_detail["release_time"] = release_time
        else: self.logger.error(f"Invalid release_time for {event_name}"); return
        if not all([event_key, event_name]): self.logger.error(f"Invalid event detail (key/name): {event_detail}"); return

        # --- Determine FMP indicator name for fetching Actual ---
        fmp_indicator_name = event_detail.get("fmp_indicator_name") # Check if provided by calendar source
        if not fmp_indicator_name: # If not provided by calendar, try mapping
            fmp_name_map = self.config.get("fmp_indicator_name_map", {})
            fmp_indicator_name = fmp_name_map.get(event_key)

        if not fmp_indicator_name: # Still no name for FMP API
             self.logger.error(f"Cannot fetch actual for {event_name} ({event_key}): Missing FMP indicator name mapping.")
             # Handle qualitative events differently - they might not need FMP name
             if event_key in ["FOMC_STMT"]: # Or other known qualitative keys
                  self.logger.info(f"Proceeding with qualitative handling for {event_name}")
                  # Allow calculation to proceed, it will handle None actual/forecast
             else:
                  return # Cannot proceed for quantitative without FMP name


        # 1. Fetch Actual Data (using FMP name)
        actual_value = None; max_attempts = self.config["actual_fetch_max_attempts"]; base_delay = self.config["actual_fetch_retry_delay_seconds"]
        # Only fetch if FMP name is known OR it's a known qualitative event we handle differently
        if fmp_indicator_name or event_key in ["FOMC_STMT"]:
             for attempt in range(max_attempts):
                  fetch_key = fmp_indicator_name if fmp_indicator_name else event_key # Use FMP name if available
                  self.logger.debug(f"Fetching actual for '{event_name}' using API ID '{fetch_key}' (Attempt {attempt+1})")
                  actual_value = await self._fetch_actual_data(fetch_key, event_name) # _fetch_actual_data now expects FMP name
                  if actual_value is not None: self.logger.info(f"Fetched actual for {event_name}: {actual_value}"); break
                  wait_time = base_delay*(1.5**attempt)+np.random.uniform(0,base_delay*0.5); self.logger.warning(f"Attempt {attempt+1}/{max_attempts} failed for {event_name}. Retrying in {wait_time:.1f}s..."); await asyncio.sleep(wait_time)
             else: self.logger.error(f"Failed to fetch actual data for {event_name} after {max_attempts} attempts."); return
        else: # Should not happen if qualitative check above is correct, but safety check
             self.logger.error(f"Skipping actual fetch for {event_name} - missing FMP indicator name.")
             return


        event_detail["actual"] = actual_value

        # 2. Calculate Sentiment Score (using standardized event_key)
        sentiment_score, confidence_score = await self.calculate_sentiment_score(event_detail)

        # 3. Store Active Sentiment (using standardized event_key)
        if sentiment_score is not None and confidence_score is not None:
            now = datetime.now(timezone.utc); min_fade_hours = self.config["event_min_fade_hours"].get(event_key, self.config["event_min_fade_hours"]["Default"]); min_fade_time = release_time + timedelta(hours=min_fade_hours)
            self.active_event_sentiments[event_key] = { "name": event_name, "score": sentiment_score, "confidence": confidence_score, "release_time": release_time, "min_fade_time": min_fade_time, "event_weight": self.config["event_weights"].get(event_key, self.config["event_weights"]["Default"]), "last_update": now, "actual": actual_value, "forecast": event_detail.get("forecast"), "fmp_indicator_name": fmp_indicator_name }
            self.logger.info(f"Stored active sentiment for {event_name} ({event_key}): Score={sentiment_score:.3f}, Conf={confidence_score:.2f}")
        else: self.logger.warning(f"Could not calculate/store sentiment for {event_name}.")


    # --- run_periodic_tasks (logic unchanged) ---
    async def run_periodic_tasks(self):
        now_ts=time.time(); now_dt=datetime.now(timezone.utc)
        try: # Fetch Calendar
             if now_ts - self.last_calendar_fetch_time > self.config["calendar_fetch_interval_seconds"]:
                  self.logger.info("Fetching updated economic calendar..."); new_cal = await self._fetch_event_calendar()
                  if isinstance(new_cal, list): self.event_calendar = new_cal; self.last_calendar_fetch_time = now_ts; self.logger.info(f"Fetched {len(self.event_calendar)} upcoming events.")
                  else: self.logger.error("Fetched event calendar invalid.")
        except Exception as e: self.logger.error(f"Failed to fetch/update event calendar: {e}")
        # Check Releases
        events_to_process = []; remaining_calendar = []; processed_keys = set()
        cal_copy = list(self.event_calendar)
        for event in cal_copy:
            try:
                 rt = event.get("release_time"); ek = event.get("event_key"); en = event.get("name", "N/A")
                 if not ek or not isinstance(rt, datetime): continue
                 if rt.tzinfo is None: rt = rt.replace(tzinfo=timezone.utc)
                 time_diff_min = (now_dt - rt).total_seconds() / 60
                 if 0 <= time_diff_min < 5: # Process window
                      if ek not in self.active_event_sentiments and ek not in processed_keys: self.logger.info(f"Event release time occurred: {en} ({ek})"); events_to_process.append(event); processed_keys.add(ek)
                 elif time_diff_min >= 5: pass # Old, remove implicitly
                 else: remaining_calendar.append(event) # Future
            except Exception as e: self.logger.error(f"Error checking calendar event: {event}. Error: {e}")
        self.event_calendar = remaining_calendar
        # Process concurrently
        if events_to_process:
             self.logger.info(f"Creating tasks for {len(events_to_process)} events...")
             tasks = [self.process_event_release(evt) for evt in events_to_process]; results = await asyncio.gather(*tasks, return_exceptions=True)
             for i, res in enumerate(results):
                  if isinstance(res, Exception): self.logger.error(f"Error processing event '{events_to_process[i].get('name','?')}': {res}", exc_info=isinstance(res, BaseException))


    # --- Helper for deep merging configs ---
    def _deep_merge_config(self, default: Dict, custom: Dict):
        for key, value in custom.items():
            if isinstance(value, dict) and isinstance(default.get(key), dict): self._deep_merge_config(default[key], value)
            else: default[key] = value