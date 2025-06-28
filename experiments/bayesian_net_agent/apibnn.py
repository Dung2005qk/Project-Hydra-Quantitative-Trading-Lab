# api.py (Version with BNN + NSGA-II Integration + Improvements)

import ccxt.async_support as ccxt
import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from sklearn.covariance import LedoitWolf # <<< IMPROVEMENT IMPORT >>>
from imblearn.over_sampling import RandomOverSampler
import joblib
import logging
from datetime import datetime, timedelta, timezone
import pytz
from typing import List, Dict, Optional, Tuple, Any
import talib
import matplotlib.pyplot as plt
import os
import asyncio
import sys
import aiohttp
from stable_baselines3 import DQN, SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_util import make_vec_env
from gymnasium import Env
from gymnasium.spaces import Discrete, Box
from transformers import pipeline
from scipy.optimize import minimize as scipy_minimize
from scipy.signal import find_peaks
from hmmlearn.hmm import GaussianHMM
from bayes_opt import BayesianOptimization
import torch
import torch.nn as nn
import torch.nn.functional as F
import pyro
import pyro.distributions as dist
from pyro.infer import Predictive

# Pymoo imports
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import ElementwiseProblem
from pymoo.optimize import minimize as pymoo_minimize
from pymoo.termination import get_termination
from pymoo.operators.sampling.lhs import LHS

import time
import json
import copy
import optuna
import traceback
from pathlib import Path

# --- Torch Geometric Handling ---
try:
    from torch_geometric.nn import GCNConv
    _torch_geometric_available = True
except ImportError:
    logging.error("torch_geometric not available. GraphNeuralNet will not function fully.")
    class GCNConv:
        def __init__(self, *args, **kwargs): pass
        def __call__(self, *args, **kwargs): return args[0]
    _torch_geometric_available = False

# --- Import/Define Enhanced BNN ---
# IMPORTANT: Ensure this class definition matches EXACTLY the one used for training
#            Load actual best parameters instead of using placeholders.
try:
    # >>> TODO: Load best BNN params (hidden_dim, num_layers, dropout) from Optuna study <<<
    _BETA_BNN_PARAMS = {"input_dim": 10, "hidden_dim": 64, "output_dim": 1, "num_hidden_layers": 2, "dropout_rate": 0.1} # ** REPLACE THIS PLACEHOLDER **

    class EnhancedBayesianNeuralNetwork(nn.Module):
        # ... (Keep the full class definition from previous response) ...
        def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_hidden_layers: int = 1, dropout_rate: float = 0.0):
            super().__init__()
            self.input_dim = input_dim; self.output_dim = output_dim; self.num_hidden_layers = max(1, num_hidden_layers); self.dropout_rate = dropout_rate
            layers = []; in_features = input_dim
            for _ in range(self.num_hidden_layers):
                layers.append(nn.Linear(in_features, hidden_dim)); layers.append(nn.ReLU())
                if self.dropout_rate > 0: layers.append(nn.Dropout(self.dropout_rate))
                in_features = hidden_dim
            layers.append(nn.Linear(in_features, output_dim)); self.net = nn.Sequential(*layers)
        def forward(self, x): return self.net(x)
        def model(self, x, y=None):
            priors = {}; module_idx = 0
            for module in self.net:
                 if isinstance(module, nn.Linear):
                      priors[f'net.{module_idx}.weight'] = dist.Normal(0., 1.).expand(module.weight.shape).to_event(module.weight.dim())
                      priors[f'net.{module_idx}.bias'] = dist.Normal(0., 1.).expand(module.bias.shape).to_event(module.bias.dim())
                 module_idx += 1
            lifted_module = pyro.random_module("module", self, priors); lifted_reg_model = lifted_module()
            prediction_mean = lifted_reg_model(x)
            sigma = pyro.param("likelihood_sigma", torch.tensor(0.1), constraint=dist.constraints.positive)
            with pyro.plate("data", x.shape[0]): pyro.sample("obs", dist.Normal(prediction_mean, sigma).to_event(1), obs=y)
            return prediction_mean
        def guide(self, x, y=None):
             guide_params = {}; module_idx = 0
             for module in self.net:
                  if isinstance(module, nn.Linear):
                       weight_shape = module.weight.shape; bias_shape = module.bias.shape
                       w_mu = pyro.param(f'guide_W_mu_{module_idx}', torch.randn(weight_shape) * 0.1); w_sigma = pyro.param(f'guide_W_sigma_{module_idx}', torch.randn(weight_shape) * 0.1)
                       guide_params[f'net.{module_idx}.weight'] = dist.Normal(w_mu, F.softplus(w_sigma)).to_event(module.weight.dim())
                       b_mu = pyro.param(f'guide_b_mu_{module_idx}', torch.randn(bias_shape) * 0.1); b_sigma = pyro.param(f'guide_b_sigma_{module_idx}', torch.randn(bias_shape) * 0.1)
                       guide_params[f'net.{module_idx}.bias'] = dist.Normal(b_mu, F.softplus(b_sigma)).to_event(module.bias.dim())
                  module_idx += 1
             lifted_module = pyro.random_module("module", self, guide_params); return lifted_module()

    _BNN_CLASS_AVAILABLE = True
except Exception as bnn_def_e:
     logging.error(f"Error defining EnhancedBayesianNeuralNetwork class: {bnn_def_e}")
     EnhancedBayesianNeuralNetwork = None
     _BNN_CLASS_AVAILABLE = False

# --- Configuration & Logging ---
SCRIPT_DIR = Path(__file__).parent
LOG_FILE_PATH = SCRIPT_DIR / "enhanced_bot_v_bnn_nsga.log" # New log file

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s",
    handlers=[logging.FileHandler(LOG_FILE_PATH, encoding="utf-8"), logging.StreamHandler()]
)

# --- CONFIG ---
# Assume CONFIG dict is defined here or loaded, including new parameters:
# decision_cycle_interval_seconds, bnn_model_dir, bnn_input_features,
# bnn_future_return_period, bnn_num_prediction_samples, nsga2_pop_size,
# nsga2_n_gen, portfolio_risk_aversion, optimizer_min_weight,
# target_portfolio_risk (e.g., 0.015 for 1.5% target std dev)
CONFIG = {
    "symbols": ["BTC/USDT", "ETH/USDT"], "timeframes": ["1m", "5m", "15m", "1h", "4h"],
    "days_to_fetch": 365, "initial_balance": 10000, "leverage": 5, "risk_per_trade": 0.02,
    "fee_rate": 0.0005, "max_trades_per_day": 20, "max_account_risk": 0.07,
    "min_position_size": 0.01, "max_position_size": 0.5, "prob_threshold": 0.65,
    "sentiment_window": 24, "min_samples": 100, "lookback_window": 20,
    "hmm_warmup_min": 50, "hmm_warmup_max": 200, "vah_bin_range": (20, 60),
    "fib_levels": [0, 0.236, 0.382, 0.5, 0.618, 1], "max_gap_storage": 3,
    "circuit_breaker_threshold": 4.0, "circuit_breaker_cooldown": 60, "max_exposure": 0.75,
    "volume_spike_multiplier": 4.0, "volume_sustained_multiplier": 1.5,
    "adx_thresholds": {"1h": 22, "4h": 25, "15m": 25, "default": 20},
    "volume_ma20_threshold": 0.8,
    "api_key": os.getenv("BINANCE_API_KEY", "YOUR_API_KEY"), "api_secret": os.getenv("BINANCE_API_SECRET", "YOUR_API_SECRET"),
    "enable_rate_limit": True, "rate_limit_delay": 0.2, "websocket_url": "wss://stream.binance.com:9443/ws",
    "xgb_long_threshold": 0.65, "xgb_short_threshold": 0.35, "min_xgb_feature_dim": 20,
    # --- New/Updated Params ---
    "decision_cycle_interval_seconds": 60,
    "bnn_model_dir": str(SCRIPT_DIR / "trained_models_bnn_v3"),
    "bnn_input_features": ["entry_price_norm", "atr_norm", "volatility", "correlation_proxy", "liquidity_proxy", "duration_proxy", "RSI", "MACD_hist", "BB_width", "ADX"],
    "bnn_future_return_period": 10,
    "bnn_num_prediction_samples": 100,
    "nsga2_pop_size": 50, "nsga2_n_gen": 50,
    "portfolio_risk_aversion": 0.5, # Not directly used in Sharpe selection, but could be
    "optimizer_min_weight": 0.05,
    "target_portfolio_risk": 0.015, # Target portfolio std dev (e.g., 1.5%)
    "covariance_lookback_days": 90, # Days for historical cov calculation
    "covariance_min_history": 50,   # Min data points for cov calculation
}
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
logging.info(f"Using device: {DEVICE}")
TIMEFRANE = "15m" # Assuming BNN uses 15m data for features/target

# --- Placeholder Classes (Keep names, replace body with pass) ---
class RiskFeatureEncoder: pass
class LSTMDepthPredictor(nn.Module): pass
class RealTimeDataFeed: pass # Assume implementation exists
class LOBTransformer(nn.Module): pass
class PriceActionTransformer(nn.Module): pass
class MarketEnvironment(Env): pass # Assume implementation exists
class EntryPointOptimizer: pass # Assume implementation exists
# class BayesianNeuralNetwork(nn.Module): pass # Old BNN - Removed/Replaced
# class PnLOptimizer: pass # Old Optimizer - Removed/Replaced
class MCSimulator: pass
# class PortfolioOptimizationProblem(ElementwiseProblem): pass # Old NSGA Problem - Removed/Replaced
class TransformerForecaster(nn.Module): pass
class GraphNeuralNet(nn.Module): pass
class IntelligentStopSystem: pass # Assume implementation exists
class FederatedAveraging: pass
class SmartPositionSizer: pass # Assume implementation exists
class HierarchicalAttention: pass
class MultiTimeframeOptimizer: pass
class TradingEnv(Env): pass # Assume implementation exists

# --- NSGA-II Problem Definition ---
class PortfolioOptimizationProblemBNN(ElementwiseProblem):
    # ... (Keep the class definition from the previous response) ...
    def __init__(self, n_vars: int, mu_vector: np.ndarray, variance_vector: np.ndarray, covariance_matrix: np.ndarray):
        if not (mu_vector.shape[0] == n_vars and variance_vector.shape[0] == n_vars and covariance_matrix.shape == (n_vars, n_vars)): raise ValueError("Dimension mismatch")
        super().__init__(n_var=n_vars, n_obj=2, n_constr=1, xl=0.0, xu=1.0)
        self.mu = mu_vector; self.variances = variance_vector; self.Sigma = covariance_matrix
    def _evaluate(self, w, out, *args, **kwargs):
        expected_return = np.sum(w * self.mu); portfolio_variance = np.dot(w.T, np.dot(self.Sigma, w))
        out["F"] = [-expected_return, portfolio_variance]; out["G"] = [np.sum(w) - 1.0]


# --- Main Trading Bot Class ---
class EnhancedTradingBot:
    # ... (Keep XGB constants and ADAPTIVE_WEIGHTS) ...
    XGB_STRONG_PROB = 0.75; XGB_MEDIUM_PROB = 0.60; XGB_WEAK_PROB = 0.55
    ADAPTIVE_WEIGHTS = {'high_vol': {'sac': 0.7, 'xgb': 0.3}, 'trending': {'sac': 0.4, 'xgb': 0.6}, 'default': {'sac': 0.5, 'xgb': 0.5}}

    def __init__(self):
        # --- Existing initializations ---
        self.exchange = None
        self.balance = CONFIG.get("initial_balance", 10000)
        self.equity_peak = self.balance
        self.trade_history = []
        self.scalers = {} # XGB Scalers
        self.models = {} # XGB Models
        self.drl_model = None # DQN Model
        self.entry_optimizer = None # SAC Model
        self.stop_system = IntelligentStopSystem() # Assumed implemented
        self.regime_model = GaussianHMM(n_components=2, covariance_type="diag", n_iter=100, random_state=42)
        self.gap_levels = {}; self.walk_forward_windows = []
        self.temporal_features = CONFIG.get('temporal_features', ['hour', 'day_of_week', 'is_weekend', 'time_since_last_high', 'time_since_last_gap', 'volatility_regime_change'])
        self.circuit_breaker_triggered = False; self.last_trade_time = None
        self.exposure_per_symbol = {symbol: 0.0 for symbol in CONFIG.get("symbols", [])}
        self.open_positions = {}; self.data = {}
        self.rollback_state = None; self.data_feeds = {}; self.indicator_cache = {}
        self._setup_social_media(); self._validate_config()
        try: self.position_sizer = SmartPositionSizer(self) # Assumed implemented
        except Exception as e: logging.critical(f"Failed SP Sizer init: {e}", exc_info=True); raise
        self._cleanup_complete = False
        self.trading_env_scaler = None # DQN Scaler

        # --- BNN+NSGA-II related initializations ---
        self.bnn_model = None; self.bnn_guide = None; self.bnn_scaler = None
        self.bnn_input_dim = len(CONFIG.get("bnn_input_features", []))
        self.last_decision_time = time.time()

        if _BNN_CLASS_AVAILABLE and self.bnn_input_dim > 0: self._load_bnn_model_and_scaler()
        else: logging.warning("BNN class/features not available. BNN+NSGA-II disabled.")

        self._load_drl_models() # Load DQN/SAC models

    # <<< NEW/MODIFIED METHOD >>>
    def _load_bnn_model_and_scaler(self):
        """Loads the trained BNN model, guide parameters, and scaler."""
        bnn_dir = Path(CONFIG.get("bnn_model_dir", str(SCRIPT_DIR / "trained_models_bnn_v3")))
        guide_path = bnn_dir / "bnn_pnl_guide_final_optuna.pth"
        scaler_path = bnn_dir / "bnn_pnl_scaler.pkl"

        # 1. Load Scaler
        if not scaler_path.exists(): logging.error(f"BNN Scaler not found: {scaler_path}. BNN disabled."); return
        try:
            self.bnn_scaler = joblib.load(scaler_path)
            logging.info(f"BNN Scaler loaded from {scaler_path}")
            if not hasattr(self.bnn_scaler, 'n_features_in_') or self.bnn_scaler.n_features_in_ != self.bnn_input_dim:
                logging.error(f"BNN Scaler dim mismatch! Expected {self.bnn_input_dim}, got {getattr(self.bnn_scaler, 'n_features_in_', 'N/A')}. BNN disabled.")
                self.bnn_scaler = None; return
        except Exception as e: logging.error(f"Failed to load BNN scaler: {e}", exc_info=True); self.bnn_scaler = None; return

        # 2. Instantiate BNN Model
        try:
            # >>> TODO: Load actual best params from Optuna study for BNN <<<
            best_bnn_params = _BETA_BNN_PARAMS # ** REPLACE THIS PLACEHOLDER **
            if best_bnn_params["input_dim"] != self.bnn_input_dim:
                 logging.warning(f"Mismatch BNN input_dim: Params({best_bnn_params['input_dim']}) vs Config({self.bnn_input_dim}). Using Config.")
                 best_bnn_params["input_dim"] = self.bnn_input_dim

            self.bnn_model = EnhancedBayesianNeuralNetwork(**best_bnn_params).to(DEVICE).eval()
            self.bnn_guide = self.bnn_model.guide
            logging.info(f"BNN Model instantiated with params: {best_bnn_params}")
        except Exception as e: logging.error(f"Failed BNN model instantiation: {e}", exc_info=True); self.bnn_model = None; self.bnn_guide = None; return

        # 3. Load Guide Parameters
        if not guide_path.exists(): logging.error(f"BNN Guide params not found: {guide_path}. BNN disabled."); self.bnn_model=None; self.bnn_guide=None; return
        if self.bnn_model and self.bnn_guide:
             try:
                  pyro.clear_param_store()
                  pyro.get_param_store().load(str(guide_path), map_location=DEVICE)
                  logging.info(f"BNN Guide parameters loaded from {guide_path}")
             except Exception as e: logging.error(f"Failed BNN guide load: {e}", exc_info=True); self.bnn_model=None; self.bnn_guide=None
        else: logging.error("BNN Model/Guide not ready for param loading. BNN disabled."); self.bnn_model=None; self.bnn_guide=None

    def _load_drl_models(self):
        """Loads DRL (DQN) model and associated scaler."""
        # ... (Keep implementation from previous response) ...
        pass # Placeholder


    async def initialize(self):
        """Initializes exchange, data, and components."""
        # ... (Keep implementation from previous response, ensure self.data[symbol][tf].attrs are set) ...
        pass # Placeholder

    # --- Data Fetching and Processing ---
    async def fetch_multi_timeframe_data(self, symbol: str) -> Dict[str, Optional[pd.DataFrame]]:
        # ... (Keep implementation - ensure calculate_advanced_indicators and attrs setting happens) ...
        pass # Placeholder

    async def _fetch_ohlcv(self, symbol: str, timeframe: str) -> Optional[pd.DataFrame]: pass # Placeholder
    def _process_ohlcv(self, ohlcv: List) -> pd.DataFrame: pass # Placeholder

    def calculate_advanced_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
         # ... (Keep implementation - ensure BNN features and proxies are calculated) ...
         pass # Placeholder

    # --- Other indicator methods ---
    def detect_swing_points(self, df: pd.DataFrame, lookback: int = 5, prominence_factor: float = 0.001) -> pd.DataFrame: pass # Placeholder
    def detect_regime_change(self, df: pd.DataFrame) -> pd.DataFrame: pass # Placeholder
    # ... other methods like detect_divergence, detect_price_gaps, etc. ...

    # --- ML/DRL Training Methods ---
    def prepare_ml_data(self, df: pd.DataFrame, symbol: str) -> Optional[Tuple[np.ndarray, np.ndarray]]: pass
    def train_model(self, symbol: str, X: np.ndarray, y: np.ndarray): pass
    def walk_forward_validation(self, data: Dict[str, pd.DataFrame], symbol: str): pass
    def evaluate_model(self, model, X_test, y_test) -> dict: pass
    async def train_drl(self, data: Dict[str, Dict[str, pd.DataFrame]]): pass
    async def train_entry_optimizer(self, data: Dict[str, Dict[str, pd.DataFrame]]): pass

    # --- Signal Generation (Candidate) ---
    async def generate_signal(self, symbol: str, data: dict) -> Optional[dict]:
        # ... (Keep implementation generating candidate signal dict) ...
        pass # Placeholder

    # --- BNN Feature Prep and Prediction ---
    def _prepare_bnn_features_for_signal(self, signal: dict, last_row: pd.Series) -> Optional[np.ndarray]:
        # ... (Keep implementation from previous response) ...
        pass # Placeholder

    def _predict_bnn_return_distribution(self, features_tensor: torch.Tensor) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        # ... (Keep implementation from previous response) ...
        pass # Placeholder

    # <<< IMPROVED METHOD >>>
    def _estimate_covariance_matrix(self, signals: List[dict], mus: np.ndarray, sigmas: np.ndarray) -> np.ndarray:
        """ Estimates covariance matrix using historical returns and Ledoit-Wolf shrinkage. """
        num_signals = len(signals)
        if num_signals <= 1:
            return np.array([[sigmas[0]**2]]) if num_signals == 1 and sigmas is not None and len(sigmas) == 1 else np.array([[0.01**2]])

        historical_returns = {}
        min_history_length = CONFIG.get("covariance_min_history", 50)
        lookback_days = CONFIG.get("covariance_lookback_days", 90)
        latest_signal_ts = max(s['timestamp'] for s in signals) if signals else pd.Timestamp.now(tz='UTC') # Use timezone aware
        end_date = latest_signal_ts
        start_date = end_date - pd.Timedelta(days=lookback_days)

        valid_symbols_for_cov = []
        symbols_in_signals = [s['symbol'] for s in signals] # List of symbols in order

        for symbol in set(symbols_in_signals): # Get unique symbols first
            if symbol in self.data and TIMEFRANE in self.data[symbol]:
                df = self.data[symbol][TIMEFRANE]
                # Ensure index is datetime and timezone aware if possible
                if not isinstance(df.index, pd.DatetimeIndex): continue # Skip if index invalid
                if df.index.tz is None: df.index = df.index.tz_localize('UTC') # Assume UTC if naive

                df_slice = df[(df.index >= start_date) & (df.index <= end_date)]
                if len(df_slice) >= min_history_length:
                     returns = df_slice['close'].pct_change().dropna()
                     if len(returns) >= min_history_length - 1:
                          historical_returns[symbol] = returns
                     # else: logging.debug(f"Not enough returns for {symbol}")
                # else: logging.debug(f"Not enough data points for {symbol}")
            # else: logging.debug(f"No base data for {symbol}")

        # Align returns based on the order in 'signals'
        aligned_returns_list = []
        valid_signal_indices_for_cov = [] # Indices in the original 'signals' list that have valid returns
        for i, symbol in enumerate(symbols_in_signals):
            if symbol in historical_returns:
                aligned_returns_list.append(historical_returns[symbol])
                valid_signal_indices_for_cov.append(i)
            # else: logging.debug(f"Symbol {symbol} from signal {i} has no historical returns.")

        num_valid_cov_signals = len(aligned_returns_list)

        if num_valid_cov_signals <= 1:
            logging.warning(f"Covariance: <=1 symbol with history ({num_valid_cov_signals}). Returning diagonal.")
            variances = sigmas**2 if sigmas is not None and len(sigmas) == num_signals else np.full(num_signals, 0.01**2)
            return np.diag(variances)

        # Create aligned DataFrame
        returns_df = pd.concat(aligned_returns_list, axis=1, keys=[signals[i]['symbol'] for i in valid_signal_indices_for_cov]).dropna()

        if len(returns_df) < min_history_length / 2:
            logging.warning(f"Covariance: Not enough overlapping data ({len(returns_df)}). Returning diagonal.")
            variances = sigmas**2 if sigmas is not None and len(sigmas) == num_signals else np.full(num_signals, 0.01**2)
            return np.diag(variances)

        # Calculate Covariance using Ledoit-Wolf
        try:
            estimator = LedoitWolf()
            # Fit on the available valid symbols' returns
            estimator.fit(returns_df.values)
            cov_matrix_valid = estimator.covariance_

            # --- Expand to the original number of signals, filling non-valid with 0 cov and estimated variance ---
            full_cov_matrix = np.zeros((num_signals, num_signals))
            # Place estimated variances on the diagonal
            diag_variances = sigmas**2 if sigmas is not None and len(sigmas) == num_signals else np.full(num_signals, 0.01**2)
            np.fill_diagonal(full_cov_matrix, diag_variances)

            # Fill in the calculated covariances for valid signals
            valid_indices_mesh = np.ix_(valid_signal_indices_for_cov, valid_signal_indices_for_cov)
            full_cov_matrix[valid_indices_mesh] = cov_matrix_valid

            # Ensure Positive Semidefinite
            min_eig = np.min(np.linalg.eigvalsh(full_cov_matrix))
            if min_eig < 1e-9:
                logging.warning(f"Covariance: Full matrix not PSD (min_eig={min_eig:.2e}). Adding diagonal noise.")
                full_cov_matrix += np.eye(num_signals) * 1e-9

            logging.info(f"Estimated covariance matrix using Ledoit-Wolf for {num_valid_cov_signals}/{num_signals} signals.")
            return full_cov_matrix

        except Exception as e:
            logging.error(f"Error estimating covariance matrix: {e}", exc_info=True)
            logging.warning("Falling back to diagonal covariance matrix.")
            variances = sigmas**2 if sigmas is not None and len(sigmas) == num_signals else np.full(num_signals, 0.01**2)
            return np.diag(variances)


    # --- Solution Selection ---
    def _select_solution_from_pareto(self, results: Any, mus: np.ndarray, variances: np.ndarray, Sigma: np.ndarray) -> Optional[np.ndarray]:
        # ... (Keep implementation from previous response - select by Sharpe) ...
        pass # Placeholder

    # --- NSGA-II Optimization ---
    def _optimize_portfolio_nsga2(self, mus: np.ndarray, sigmas: np.ndarray, Sigma: np.ndarray) -> Optional[np.ndarray]:
        # ... (Keep implementation from previous response) ...
        pass # Placeholder

    # --- Main Loop --- <<< MODIFIED >>>
    async def run(self):
        """Main bot loop: init, train, periodic decision cycle."""
        try:
            await self.initialize()
            logging.info("Bot initialized.")

            # --- Initial Training (Assume this happens correctly) ---
            # ... (Existing training logic for XGB, DRL, SAC) ...
            logging.info("Initial model training assumed complete.") # Placeholder message

            # --- Setup Real-Time Feeds ---
            # ... (Existing feed setup logic) ...
            logging.info("Real-time feeds assumed setup.") # Placeholder message

            # --- Main Decision Loop ---
            logging.info("Starting main portfolio decision loop...")
            decision_interval = CONFIG.get("decision_cycle_interval_seconds", 60)
            self.last_decision_time = time.time() - decision_interval # Allow first cycle

            while True:
                if self._cleanup_complete: break
                loop_start_time = time.time()

                # 1. Update Data Feeds (Essential for fresh data)
                update_tasks = [self._process_symbol_data_update(symbol) for symbol in self.data_feeds.keys()]
                if update_tasks: await asyncio.gather(*update_tasks, return_exceptions=True)

                # 2. Portfolio Decision Cycle
                current_time = time.time()
                if current_time - self.last_decision_time >= decision_interval:
                    logging.info(f"--- Starting Decision Cycle ---")
                    self.last_decision_time = current_time

                    # Check global circuit breaker
                    # ... (Existing CB check logic) ...
                    global_cb_active = False # Placeholder

                    if global_cb_active:
                        logging.warning("Global circuit breaker active. Skipping decision cycle.")
                    elif self.bnn_model is None or self.bnn_guide is None or self.bnn_scaler is None:
                         logging.warning("BNN Model/Guide/Scaler not available. Skipping portfolio optimization cycle.")
                    else:
                        # --- Portfolio Optimization Logic ---
                        candidate_signals = await self._collect_candidate_signals()

                        if candidate_signals:
                            logging.info(f"Collected {len(candidate_signals)} candidate signals.")
                            bnn_results = await self._evaluate_signals_with_bnn(candidate_signals)

                            if bnn_results:
                                mus_np, sigmas_np, valid_indices, valid_signals_list, features_list = bnn_results
                                if len(valid_signals_list) > 0:
                                    # Estimate Covariance (using improved method)
                                    Sigma = self._estimate_covariance_matrix(valid_signals_list, mus_np, sigmas_np) # Pass mus/sigmas

                                    # Optimize Portfolio
                                    optimal_weights_raw = self._optimize_portfolio_nsga2(mus_np, sigmas_np, Sigma) # NSGA returns weights for valid signals

                                    # --- Map optimal_weights_raw back to original signal list ---
                                    final_optimal_weights = np.zeros(len(candidate_signals)) # Full weight vector
                                    if optimal_weights_raw is not None and len(optimal_weights_raw) == len(valid_indices):
                                         final_optimal_weights[valid_indices] = optimal_weights_raw
                                    else:
                                         logging.warning("NSGA-II optimization failed or returned invalid weights. No trades this cycle.")

                                    # Execute Trades based on final weights (pass BNN results too)
                                    # Need mus/sigmas corresponding to the *original* candidate_signals order
                                    full_mus = np.zeros(len(candidate_signals))
                                    full_sigmas = np.ones(len(candidate_signals)) * 1.0 # Default large sigma
                                    if len(valid_indices) > 0:
                                         full_mus[valid_indices] = mus_np
                                         full_sigmas[valid_indices] = sigmas_np

                                    # Pass the full weight vector and BNN results
                                    await self._execute_optimized_trades_v3(
                                        candidate_signals,      # Original list
                                        final_optimal_weights, # Full weight vector (with zeros)
                                        features_list,          # Features (order might need adjustment if using) - Check if needed
                                        full_mus,               # Full mu vector
                                        full_sigmas,            # Full sigma vector
                                        Sigma                   # Covariance matrix (order matching valid_signals) - Need careful handling
                                    )
                                else:
                                    logging.info("No signals passed BNN evaluation.")
                            else:
                                logging.info("BNN evaluation failed for all candidates.")
                        else:
                            logging.info("No candidate signals generated in this cycle.")
                    logging.info(f"--- Finished Decision Cycle ---")

                # 3. Manage Open Positions
                await self._manage_open_positions()

                # 4. Sleep
                loop_duration = time.time() - loop_start_time
                next_decision_in = max(0.1, (self.last_decision_time + decision_interval) - time.time())
                sleep_time = max(0.1, min(1.0 - loop_duration, next_decision_in))
                await asyncio.sleep(sleep_time)

        # ... (Keep existing exception handling and finally block) ...
        except asyncio.CancelledError: logging.info("Bot run loop cancelled.")
        except Exception as outer_e: logging.critical(f"Critical error: {outer_e}", exc_info=True); logging.critical(traceback.format_exc())
        finally:
            if not self._cleanup_complete: await self._cleanup()

    # --- Candidate Signal Collection ---
    async def _collect_candidate_signals(self) -> List[dict]:
        # ... (Keep implementation from previous response) ...
        pass # Placeholder

    # --- BNN Signal Evaluation ---
    async def _evaluate_signals_with_bnn(self, signals: List[dict]) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, List[dict], List[np.ndarray]]]:
        # ... (Keep implementation from previous response) ...
        pass # Placeholder

    # <<< IMPROVED METHOD v3 >>>
    async def _execute_optimized_trades_v3(self,
                                        all_signals: List[dict],     # All original candidate signals
                                        optimal_weights: np.ndarray, # Full weight vector (incl. zeros)
                                        features_list: List[np.ndarray], # Corresponding features (order check needed)
                                        all_mus: np.ndarray,         # Mu for all original signals
                                        all_sigmas: np.ndarray,      # Sigma for all original signals
                                        valid_cov_matrix: np.ndarray): # Cov matrix only for *valid* signals from BNN eval
        """
        Calculates final size and places trades based on NSGA-II weights,
        incorporating portfolio risk (sigma_p) and improved risk scaling.
        Receives results aligned with the original 'all_signals' list.
        """
        if not (len(all_signals) == len(optimal_weights) == len(all_mus) == len(all_sigmas)):
            logging.error("Mismatch in inputs for execute_optimized_trades_v3. Aborting.")
            return

        min_weight_threshold = CONFIG.get("optimizer_min_weight", 0.05)
        target_portfolio_risk = CONFIG.get("target_portfolio_risk", 0.015) # Target std dev %

        # --- Identify signals to actually trade (weight > threshold) ---
        trade_indices = np.where(optimal_weights >= min_weight_threshold)[0]
        if len(trade_indices) == 0:
            logging.info("No signals selected for trading by optimizer.")
            return

        num_trade_signals = len(trade_indices)
        logging.info(f"--- Executing Trades based on Portfolio Optimization ({num_trade_signals} signals) ---")

        # --- Recalculate Portfolio Risk ONLY for selected trades ---
        #    We need the weights and covariance matrix corresponding ONLY to the selected trades.
        selected_weights_raw = optimal_weights[trade_indices]
        selected_mus = all_mus[trade_indices]
        selected_sigmas = all_sigmas[trade_indices]

        # Extract the sub-covariance matrix for the selected signals
        # This requires knowing the mapping from trade_indices back to the indices used for valid_cov_matrix
        # Let's assume valid_cov_matrix corresponds to the signals that *passed BNN eval*
        # We need the original indices of those that passed BNN eval AND were selected by NSGA
        # This mapping is becoming complex. Let's simplify for now: Calculate sigma_p using *all* weights > 0
        # This might slightly overestimate the scaling factor if some signals passed BNN but got 0 weight.

        weights_for_risk_calc = optimal_weights # Use the full vector returned by _optimize_portfolio_nsga2
        valid_indices_for_cov = np.where(all_sigmas < 0.99)[0] # Indices of signals that passed BNN eval (sigma != 1.0 default)
        if len(valid_indices_for_cov) != valid_cov_matrix.shape[0]:
             logging.warning("Mismatch between covariance matrix size and signals passing BNN. Using diagonal fallback for risk scaling.")
             portfolio_risk_sigma_p = None
        else:
             # Extract weights corresponding to the valid covariance matrix
             weights_subset_for_cov = weights_for_risk_calc[valid_indices_for_cov]
             # Normalize these subset weights for sigma_p calculation relative to selected group
             if weights_subset_for_cov.sum() > 1e-6:
                  norm_weights_for_cov = weights_subset_for_cov / weights_subset_for_cov.sum()
                  try:
                      portfolio_variance = norm_weights_for_cov.T @ valid_cov_matrix @ norm_weights_for_cov
                      portfolio_risk_sigma_p = np.sqrt(max(portfolio_variance, 1e-12))
                      logging.info(f"Predicted Portfolio Risk (sigma_p) for evaluated assets: {portfolio_risk_sigma_p:.6f}")
                  except Exception as e:
                      logging.error(f"Error calculating portfolio risk sigma_p: {e}", exc_info=True)
                      portfolio_risk_sigma_p = None
             else:
                  portfolio_risk_sigma_p = None # No valid weights in the cov matrix subset


        # --- Determine Risk Scaling Factor ---
        risk_scaling_factor = 1.0
        if portfolio_risk_sigma_p is not None and portfolio_risk_sigma_p > 1e-6:
            risk_scaling_factor = target_portfolio_risk / portfolio_risk_sigma_p
            risk_scaling_factor = np.clip(risk_scaling_factor, 0.5, 1.5)
            logging.info(f"Portfolio Risk Scaling Factor (sigma_p={portfolio_risk_sigma_p:.4f} vs target={target_portfolio_risk:.4f}): {risk_scaling_factor:.3f}")
        else:
            logging.warning("Could not calculate portfolio risk (sigma_p). Using scaling factor 1.0.")

        # --- Determine Capital Allocation ---
        available_exposure = CONFIG.get("max_exposure", 0.75) - sum(self.exposure_per_symbol.values())
        allocation_percentage = min(0.5 * CONFIG.get("max_exposure", 0.75), max(0, available_exposure))
        allocated_capital = self.balance * allocation_percentage
        logging.info(f"Allocating capital for this cycle: ${allocated_capital:.2f} ({allocation_percentage:.2%})")

        if allocated_capital <= 10: logging.info("Insufficient capital."); return

        # --- Normalize selected weights to sum to 1 for capital distribution ---
        selected_trade_weights = optimal_weights[trade_indices]
        if selected_trade_weights.sum() < 1e-6: logging.info("Zero total weight for selected trades."); return
        normalized_trade_weights = selected_trade_weights / selected_trade_weights.sum()


        # --- Loop through selected signals for execution ---
        for i, original_idx in enumerate(trade_indices):
            signal = all_signals[original_idx]
            weight = normalized_trade_weights[i] # Use normalized weight for capital allocation
            symbol = signal['symbol']
            signal_direction = signal['direction']
            bnn_mu = all_mus[original_idx]
            bnn_sigma = all_sigmas[original_idx]

            if symbol in self.open_positions: continue # Double check

            # 1. Calculate Capital Allocation
            capital_i = allocated_capital * weight
            if capital_i < 1: continue

            # 2. Calculate Initial Risk % for Allocated Capital
            base_risk = 0.0; quality = signal.get("signal_quality", "WEAK")
            if quality == 'STRONG': base_risk = CONFIG.get("risk_per_trade", 0.02) * 1.2
            elif quality == 'MEDIUM': base_risk = CONFIG.get("risk_per_trade", 0.02)
            else: base_risk = CONFIG.get("risk_per_trade", 0.02) * 0.8
            expected_sharpe_i = bnn_mu / bnn_sigma if bnn_sigma > 1e-6 else bnn_mu * 100
            sharpe_factor = np.clip(1 + (expected_sharpe_i - 0.5) * 0.2, 0.8, 1.5)
            initial_risk_for_alloc_capital = np.clip(base_risk * sharpe_factor, 0.005, 0.1)

            # 3. Apply Portfolio Risk Scaling
            risk_per_allocated_capital = initial_risk_for_alloc_capital * risk_scaling_factor
            risk_per_allocated_capital = np.clip(risk_per_allocated_capital, 0.002, 0.15)

            logging.info(f"Processing {symbol} {signal_direction}: NormWeight={weight:.3f}, Capital=${capital_i:.2f}, "
                         f"InitialRisk={initial_risk_for_alloc_capital:.3%}, PortScale={risk_scaling_factor:.2f} -> "
                         f"RiskOfAlloc={risk_per_allocated_capital:.3%}")

            # 4. Calculate TP/SL
            try:
                tp_levels, sl_price = self.dynamic_stop_management(signal)
                if (signal_direction == 'LONG' and sl_price >= signal['entry_price']) or \
                   (signal_direction == 'SHORT' and sl_price <= signal['entry_price']):
                     logging.error(f"Invalid SL for {symbol}. Skipping."); continue
            except Exception as e: logging.error(f"Error TP/SL {symbol}: {e}"); continue

            # 5. Calculate Position Size
            try:
                sl_distance = abs(signal['entry_price'] - sl_price)
                if sl_distance < 1e-9: raise ValueError("SL distance too small")
                max_loss_amount_usd = capital_i * risk_per_allocated_capital
                position_size = max_loss_amount_usd / sl_distance

                min_qty = self._safe_get_min_qty(symbol)
                max_qty = self._safe_get_max_qty(symbol) # Assume this exists
                position_size = np.clip(position_size, min_qty, max_qty)

                amount_prec = self.data[symbol][TIMEFRANE].attrs.get('amount_precision', 8)
                if hasattr(self.exchange, 'amount_to_precision'):
                     size_str = self.exchange.amount_to_precision(symbol, position_size)
                     position_size = float(size_str)
                else: position_size = round(position_size, amount_prec)

                if position_size < min_qty: position_size = 0.0
            except Exception as e: logging.error(f"Error Calc Size {symbol}: {e}"); position_size = 0.0

            # 6. Final Checks and Execution
            if position_size <= 0: continue
            if not self.check_position_exposure(position_size, signal["entry_price"], symbol):
                logging.warning(f"Skipping {symbol} {signal_direction}: Exposure limit."); continue

            # --- Execute Simulated Order ---
            log_prefix = "!!! SIMULATED ORDER (Optimized v3) !!!"
            price_prec = self.data[symbol][TIMEFRANE].attrs.get('price_precision', 4)
            logging.info(f"{log_prefix} Opening {signal_direction} {symbol}")
            logging.info(f"{log_prefix} Size: {position_size:.{amount_prec}f} | Entry: ~{signal['entry_price']:.{price_prec}f}")
            logging.info(f"{log_prefix} TPs: {[f'{p:.{price_prec}f}' for p in tp_levels]} | SL: {sl_price:.{price_prec}f}")
            logging.info(f"{log_prefix} Context: NormWeight={weight:.3f}, AllocCapital=${capital_i:.2f}, RiskOfAlloc={risk_per_allocated_capital:.3%}, PortRiskScale={risk_scaling_factor:.2f}")

            position_opened = self._add_open_position(signal, position_size, sl_price, tp_levels)
            if position_opened: logging.info(f"{log_prefix} Position state updated for {symbol}.")
            else: logging.warning(f"{log_prefix} Failed to update open position state for {symbol}.")

        logging.info(f"--- Finished Executing Optimized Trades ---")


    # --- Placeholder for other methods ---
    async def _process_symbol_data_update(self, symbol: str): pass
    async def _update_symbol_timeframe(self, symbol: str, tf: str) -> bool: pass
    def _get_observation_for_dqn(self, df_15m: pd.DataFrame) -> Optional[np.ndarray]: pass
    def _get_dynamic_adx_threshold(self, symbol: str, timeframe: str) -> int: pass
    def _get_higher_timeframe_trend(self, symbol: str, timeframe: str) -> Optional[str]: pass
    async def _manage_open_positions(self): pass
    def _rollback_to_safe_state(self): pass
    def plot_performance(self): pass
    def check_circuit_breaker(self, df: pd.DataFrame) -> bool: pass
    def check_position_exposure(self, position_size: float, entry_price: float, symbol: str) -> bool: pass
    # calculate_position_size is not directly called anymore for sizing
    # def calculate_position_size(self, signal: dict, stop_loss_price: float, final_risk_factor: float) -> float: pass
    def dynamic_stop_management(self, signal: dict) -> Tuple[List[float], float]: pass
    def _add_open_position(self, signal: dict, position_size: float, sl_price: float, tp_levels: List[float]): pass
    def _update_balance(self, pnl: float, position_size: float, entry_price: float, exit_price: float, symbol: str, direction: str, timestamp: Any): pass
    def _safe_get_min_qty(self, symbol: str) -> float: pass
    def _safe_get_max_qty(self, symbol: str) -> float: # Add this helper
         # Similar logic to _safe_get_min_qty but for 'max' limit
         try:
              market_info = self._get_market_info(symbol)
              return float(market_info['limits']['amount']['max']) if market_info and market_info.get('limits',{}).get('amount',{}).get('max') is not None else 1e9
         except: return 1e9
    def _get_market_info(self, symbol: str) -> Optional[dict]: pass
    def _setup_social_media(self): pass
    def _validate_config(self): pass
    def _init_vah_optimizer(self, initial_data: pd.DataFrame): pass
    async def fetch_order_book(self, symbol: str, limit: int = 20) -> Optional[dict]: pass
    async def _initialize_exchange(self): pass # Add this placeholder
    def get_adaptive_warmup_period(self, df: pd.DataFrame) -> int: # Add placeholder
         return CONFIG.get("hmm_warmup_max", 200)
    def apply_slippage(self, price: float, direction: str, liquidity: float) -> float: pass # Add placeholder
    def validate_volume(self, df: pd.DataFrame) -> bool: pass # Add placeholder
    def check_volume_anomaly(self, df: pd.DataFrame, symbol: str, signal: dict) -> bool: pass # Add placeholder
    def _time_since_last_event(self, df: pd.DataFrame, col: str, compare_max: bool = False, window: int = 10) -> pd.Series: pass # Add placeholder
    def _detect_volatility_regime(self, df: pd.DataFrame) -> pd.Series: pass # Add placeholder
    def detect_price_gaps(self, df: pd.DataFrame) -> pd.DataFrame: pass # Add placeholder
    def _calculate_fib_levels(self, low: float, high: float) -> dict: pass # Add placeholder
    def calculate_volume_profile(self, df: pd.DataFrame) -> Dict[str, Any]: pass # Add placeholder
    def _get_vah_state(self, df: pd.DataFrame) -> np.ndarray: pass # Add placeholder
    def create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame: pass # Add placeholder
    def detect_rejection_candle(self, df: pd.DataFrame) -> pd.Series: pass # Add placeholder
    def detect_divergence(self, df: pd.DataFrame, swing_lookback: int = 5, max_bar_diff: int = 5) -> pd.Series: pass # Add placeholder
    async def _simulate_trade(self, data: Dict[str, pd.DataFrame], current_step: int, signal: dict, tp_levels: List[float], sl: float, position_size: float) -> Tuple[float, float]: pass # Add placeholder
    def _simulate_trade_sync(self, data: Dict[str, pd.DataFrame], current_step: int, signal: dict, tp_levels: List[float], sl: float, position_size: float) -> Tuple[float, float]: pass # Add placeholder
    # --- SAC related methods ---
    def _get_sac_state_unscaled(self, symbol: str, current_step_index: int = -1) -> Optional[np.ndarray]: pass # Placeholder
    def _scale_sac_state(self, state_unscaled: Optional[np.ndarray]) -> Optional[np.ndarray]: pass # Placeholder
    def generate_signal_sync(self, symbol: str, data: dict) -> Optional[dict]: pass # Placeholder

# --- Main Execution ---
async def main():
    # ... (Keep existing main function logic) ...
    bot = None
    try:
        bot = EnhancedTradingBot()
        await bot.run()
        if bot: bot.plot_performance()
    except ValueError as config_err: logging.critical(f"Config Error: {config_err}")
    except ccxt.AuthenticationError: logging.critical("CCXT Auth Error. Check API keys.")
    except Exception as e: logging.critical(f"Critical error in main: {e}", exc_info=True); logging.critical(traceback.format_exc())
    finally:
        if bot and not bot._cleanup_complete: await bot._cleanup()
        logging.info("Main execution finished.")

if __name__ == "__main__":
    if sys.platform == "win32": asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    try: asyncio.run(main())
    except KeyboardInterrupt: logging.info("KeyboardInterrupt received. Exiting.")