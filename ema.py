import streamlit as st
import pandas as pd
import requests
import time
import logging
import os
from datetime import datetime, timedelta
import json
from typing import Dict, List, Optional
from dataclasses import dataclass
import plotly.graph_objects as go
import plotly.express as px
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import queue
import gc

# Configure page
st.set_page_config(
    page_title="EMA Crossover Monitor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class StockConfig:
    symbol: str
    security_id: str
    exchange_segment: str
    instrument: str
    expiry_code: Optional[int] = 0
    fast_ema: int = 9
    slow_ema: int = 21

@dataclass
class EMAAlert:
    symbol: str
    security_id: str
    cross_type: str
    price: float
    fast_ema: float
    slow_ema: float
    timestamp: datetime
    volume: int

class DhanAPI:
    def __init__(self, access_token: str):
        self.base_url = "https://api.dhan.co/v2"
        self.headers = {
            "Content-Type": "application/json",
            "access-token": access_token
        }
    
    def fetch_historical_data(self, stock_config: StockConfig, days: int = 25) -> Optional[pd.DataFrame]:
        """Fetch historical data for a single stock"""
        url = f"{self.base_url}/charts/historical"
        
        to_date = datetime.now()
        from_date = to_date - timedelta(days=days)
        
        payload = {
            "securityId": str(stock_config.security_id),
            "exchangeSegment": stock_config.exchange_segment,
            "instrument": stock_config.instrument,
            "fromDate": from_date.strftime("%Y-%m-%d"),
            "toDate": to_date.strftime("%Y-%m-%d")
        }
        
        if stock_config.instrument in ["FUTIDX", "FUTSTK", "FUTCOM"] and stock_config.expiry_code:
            payload["expiryCode"] = stock_config.expiry_code
        
        try:
            response = requests.post(url, json=payload, headers=self.headers, timeout=20)
            
            if response.status_code != 200:
                logger.error(f"API Error for {stock_config.symbol}: {response.status_code}")
                return None
                
            data = response.json()
            
            if not data or "timestamp" not in data:
                return None
            
            df = pd.DataFrame({
                "timestamp": data["timestamp"],
                "open": data["open"],
                "high": data["high"],
                "low": data["low"],
                "close": data["close"],
                "volume": data["volume"]
            })
            
            df["datetime"] = pd.to_datetime(df["timestamp"], unit="s")
            df = df.sort_values("datetime").reset_index(drop=True)
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching data for {stock_config.symbol}: {str(e)}")
            return None

class EMAAnalyzer:
    @staticmethod
    def calculate_ema(df: pd.DataFrame, span: int) -> pd.Series:
        return df["close"].ewm(span=span, adjust=False).mean()
    
    @staticmethod
    def detect_crossover(df: pd.DataFrame, fast_span: int, slow_span: int) -> Optional[EMAAlert]:
        if len(df) < max(fast_span, slow_span) + 1:
            return None
        
        df["ema_fast"] = EMAAnalyzer.calculate_ema(df, fast_span)
        df["ema_slow"] = EMAAnalyzer.calculate_ema(df, slow_span)
        
        if len(df) < 2:
            return None
            
        curr_fast = df["ema_fast"].iloc[-1]
        curr_slow = df["ema_slow"].iloc[-1]
        curr_price = df["close"].iloc[-1]
        curr_volume = df["volume"].iloc[-1]
        
        prev_fast = df["ema_fast"].iloc[-2]
        prev_slow = df["ema_slow"].iloc[-2]
        
        cross_type = None
        
        if prev_fast <= prev_slow and curr_fast > curr_slow:
            cross_type = "BULLISH"
        elif prev_fast >= prev_slow and curr_fast < curr_slow:
            cross_type = "BEARISH"
        
        if cross_type:
            return EMAAlert(
                symbol="",
                security_id="",
                cross_type=cross_type,
                price=curr_price,
                fast_ema=curr_fast,
                slow_ema=curr_slow,
                timestamp=datetime.now(),
                volume=curr_volume
            )
        
        return None

class TelegramBot:
    def __init__(self, bot_token: str, chat_id: str):
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.base_url = f"https://api.telegram.org/bot{bot_token}"
    
    def send_message(self, message: str) -> bool:
        url = f"{self.base_url}/sendMessage"
        
        payload = {
            "chat_id": self.chat_id,
            "text": message,
            "parse_mode": "HTML"
        }
        
        try:
            response = requests.post(url, json=payload, timeout=10)
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Error sending Telegram message: {str(e)}")
            return False
    
    def format_alert_message(self, alert: EMAAlert, symbol: str) -> str:
        emoji = "🟢" if alert.cross_type == "BULLISH" else "🔴"
        direction = "📈" if alert.cross_type == "BULLISH" else "📉"
        
        message = f"""
{emoji} <b>EMA CROSSOVER ALERT</b> {direction}

<b>Symbol:</b> {symbol}
<b>Cross Type:</b> {alert.cross_type}

<b>Price:</b> ₹{alert.price:.2f}
<b>Fast EMA:</b> ₹{alert.fast_ema:.2f}
<b>Slow EMA:</b> ₹{alert.slow_ema:.2f}
<b>Volume:</b> {alert.volume:,}

<b>Time:</b> {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}
        """
        return message.strip()

def load_stocks_from_github(github_url: str) -> List[StockConfig]:
    """Load stocks from GitHub raw CSV URL"""
    try:
        df = pd.read_csv(github_url)
        stocks = []
        
        for _, row in df.iterrows():
            stock = StockConfig(
                symbol=row['symbol'],
                security_id=str(row['security_id']),
                exchange_segment=row['exchange_segment'],
                instrument=row['instrument'],
                expiry_code=int(row.get('expiry_code', 0)),
                fast_ema=int(row.get('fast_ema', 9)),
                slow_ema=int(row.get('slow_ema', 21))
            )
            stocks.append(stock)
        
        return stocks
    except Exception as e:
        st.error(f"Error loading stocks from GitHub: {e}")
        return []

def process_single_stock(stock_config: StockConfig, dhan_api: DhanAPI) -> Optional[EMAAlert]:
    """Process a single stock for EMA crossover"""
    try:
        df = dhan_api.fetch_historical_data(stock_config)
        if df is None or df.empty:
            return None
        
        alert = EMAAnalyzer.detect_crossover(df, stock_config.fast_ema, stock_config.slow_ema)
        if alert:
            alert.symbol = stock_config.symbol
            alert.security_id = stock_config.security_id
            return alert
        
        return None
    except Exception as e:
        logger.error(f"Error processing {stock_config.symbol}: {str(e)}")
        return None

def scan_stocks_parallel(stocks: List[StockConfig], dhan_api: DhanAPI, max_workers: int = 8) -> List[EMAAlert]:
    """Scan stocks in parallel and return alerts"""
    alerts = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_stock = {
            executor.submit(process_single_stock, stock, dhan_api): stock 
            for stock in stocks
        }
        
        for future in as_completed(future_to_stock):
            try:
                alert = future.result(timeout=45)
                if alert:
                    alerts.append(alert)
            except Exception as e:
                logger.error(f"Error in parallel processing: {e}")
    
    return alerts

# Initialize session state
if 'last_scan' not in st.session_state:
    st.session_state.last_scan = None
if 'scan_count' not in st.session_state:
    st.session_state.scan_count = 0
if 'alerts_sent' not in st.session_state:
    st.session_state.alerts_sent = 0
if 'last_alerts' not in st.session_state:
    st.session_state.last_alerts = {}
if 'monitoring_active' not in st.session_state:
    st.session_state.monitoring_active = False
if 'scan_in_progress' not in st.session_state:
    st.session_state.scan_in_progress = False

def is_market_hours() -> bool:
    """Check if current time is within market hours (9:15 AM - 3:30 PM IST)"""
    now = datetime.now()
    market_start = now.replace(hour=9, minute=15, second=0, microsecond=0)
    market_end = now.replace(hour=15, minute=30, second=0, microsecond=0)
    return market_start <= now <= market_end

def main():
    # Custom CSS for better UI
    st.markdown("""
    <style>
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #e6e9ef;
    }
    .alert-success {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 0.75rem;
        border-radius: 0.25rem;
        margin: 0.5rem 0;
    }
    .alert-danger {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
        padding: 0.75rem;
        border-radius: 0.25rem;
        margin: 0.5rem 0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.title("📈 EMA Crossover Monitor")
    st.markdown("*Real-time EMA crossover detection with Telegram alerts*")
    
    # Get configuration from Streamlit secrets or environment
    try:
        dhan_token = st.secrets.get("DHAN_TOKEN", os.getenv('DHAN_TOKEN', ''))
        telegram_token = st.secrets.get("TELEGRAM_BOT_TOKEN", os.getenv('TELEGRAM_BOT_TOKEN', ''))
        telegram_chat_id = st.secrets.get("TELEGRAM_CHAT_ID", os.getenv('TELEGRAM_CHAT_ID', ''))
        github_csv_url = st.secrets.get("GITHUB_CSV_URL", os.getenv('GITHUB_CSV_URL', ''))
        scan_interval = int(st.secrets.get("SCAN_INTERVAL_MINUTES", os.getenv('SCAN_INTERVAL_MINUTES', 15)))
        debug_mode = st.secrets.get("DEBUG_MODE", os.getenv('DEBUG_MODE', 'false')).lower() == 'true'
    except Exception as e:
        st.error(f"Configuration error: {e}")
        st.stop()
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Configuration status
        config_status = []
        if dhan_token: config_status.append("✅ DhanHQ Token")
        else: config_status.append("❌ DhanHQ Token")
        
        if telegram_token: config_status.append("✅ Telegram Bot")
        else: config_status.append("❌ Telegram Bot")
        
        if telegram_chat_id: config_status.append("✅ Chat ID")
        else: config_status.append("❌ Chat ID")
        
        if github_csv_url: config_status.append("✅ Stocks CSV")
        else: config_status.append("❌ Stocks CSV")
        
        for status in config_status:
            st.markdown(status)
        
        st.markdown("---")
        
        # Market status
        market_open = is_market_hours()
        market_status = "🟢 Market Open" if market_open else "🔴 Market Closed"
        st.markdown(f"**Market Status:** {market_status}")
        
        st.markdown(f"**Scan Interval:** {scan_interval} minutes")
        st.markdown(f"**Next Auto-Scan:** {scan_interval - (datetime.now().minute % scan_interval)} min")
        
        st.markdown("---")
        
        # Manual controls
        manual_scan_disabled = st.session_state.scan_in_progress or not all([dhan_token, telegram_token, telegram_chat_id])
        
        if st.button("🔍 Manual Scan", type="primary", disabled=manual_scan_disabled):
            if not all([dhan_token, telegram_token, telegram_chat_id, github_csv_url]):
                st.error("Please configure all required settings in Streamlit secrets")
            else:
                perform_scan(dhan_token, telegram_token, telegram_chat_id, github_csv_url)
        
        # Auto-monitoring info
        st.info(f"💡 **Auto-monitoring via UptimeRobot**\n\nPing this URL every {scan_interval} minutes:\n`{st.experimental_get_query_params().get('health', [''])[0] and 'Your app URL'}?health=check`")
    
    # Main dashboard
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Scans", 
            st.session_state.scan_count,
            delta=1 if st.session_state.scan_in_progress else None
        )
    
    with col2:
        st.metric(
            "Alerts Sent", 
            st.session_state.alerts_sent,
            delta=None
        )
    
    with col3:
        if st.session_state.last_scan:
            time_diff = datetime.now() - st.session_state.last_scan
            minutes_ago = int(time_diff.total_seconds() / 60)
            st.metric("Last Scan", f"{minutes_ago}m ago")
        else:
            st.metric("Last Scan", "Never")
    
    with col4:
        if st.session_state.scan_in_progress:
            status = "🟡 Scanning..."
        elif all([dhan_token, telegram_token, telegram_chat_id]):
            status = "🟢 Ready"
        else:
            status = "🔴 Not Configured"
        st.metric("Status", status)
    
    # Health check endpoint for UptimeRobot
    query_params = st.experimental_get_query_params()
    
    if query_params.get("health") == ["check"]:
        st.success("✅ EMA Monitor Health Check Passed")
        
        # Auto-trigger scan if conditions are met
        if (not st.session_state.scan_in_progress and 
            all([dhan_token, telegram_token, telegram_chat_id, github_csv_url]) and
            (st.session_state.last_scan is None or 
             datetime.now() - st.session_state.last_scan >= timedelta(minutes=scan_interval-1))):
            
            # Trigger scan in background
            perform_scan(dhan_token, telegram_token, telegram_chat_id, github_csv_url, auto_triggered=True)
        
        # Return health status JSON
        health_data = {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "last_scan": st.session_state.last_scan.isoformat() if st.session_state.last_scan else None,
            "scan_count": st.session_state.scan_count,
            "alerts_sent": st.session_state.alerts_sent,
            "scan_in_progress": st.session_state.scan_in_progress,
            "market_hours": is_market_hours()
        }
        st.json(health_data)
        
        # Don't show rest of UI for health checks
        return
    
    # Test endpoints
    if query_params.get("test") == ["telegram"]:
        if telegram_token and telegram_chat_id:
            bot = TelegramBot(telegram_token, telegram_chat_id)
            success = bot.send_message("🧪 Test message from EMA Monitor!")
            if success:
                st.success("✅ Telegram test message sent successfully!")
            else:
                st.error("❌ Failed to send Telegram test message")
        else:
            st.error("Telegram not configured")
        return
    
    # Display recent alerts
    st.markdown("---")
    st.subheader("📊 Recent Activity")
    
    if st.session_state.last_alerts:
        alert_data = []
        for key, timestamp in list(st.session_state.last_alerts.items())[-10:]:  # Last 10 alerts
            symbol, cross_type = key.split('_', 1)
            alert_data.append({
                'Symbol': symbol,
                'Cross Type': cross_type,
                'Time': timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                'Minutes Ago': int((datetime.now() - timestamp).total_seconds() / 60)
            })
        
        if alert_data:
            df = pd.DataFrame(alert_data)
            st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.info("No recent alerts. System will automatically scan when triggered by UptimeRobot.")
    
    # Debug information
    if debug_mode:
        with st.expander("🐛 Debug Information"):
            st.json({
                "session_state": {
                    "last_scan": str(st.session_state.last_scan),
                    "scan_count": st.session_state.scan_count,
                    "alerts_sent": st.session_state.alerts_sent,
                    "scan_in_progress": st.session_state.scan_in_progress,
                    "alert_count": len(st.session_state.last_alerts)
                },
                "config": {
                    "dhan_configured": bool(dhan_token),
                    "telegram_configured": bool(telegram_token and telegram_chat_id),
                    "github_csv_configured": bool(github_csv_url),
                    "scan_interval": scan_interval
                },
                "system": {
                    "market_hours": is_market_hours(),
                    "current_time": datetime.now().isoformat(),
                    "query_params": dict(query_params)
                }
            })

def perform_scan(dhan_token: str, telegram_token: str, telegram_chat_id: str, github_csv_url: str, auto_triggered: bool = False):
    """Perform a complete EMA crossover scan"""
    if st.session_state.scan_in_progress:
        if not auto_triggered:
            st.warning("Scan already in progress. Please wait...")
        return
    
    st.session_state.scan_in_progress = True
    
    try:
        # Load stocks
        if github_csv_url:
            stocks = load_stocks_from_github(github_csv_url)
        else:
            # Fallback to default stocks
            stocks = [
                StockConfig("TCS", "11536", "NSE_EQ", "EQUITY", 0, 9, 21),
                StockConfig("RELIANCE", "2885", "NSE_EQ", "EQUITY", 0, 9, 21),
                StockConfig("INFY", "1594", "NSE_EQ", "EQUITY", 0, 12, 26),
            ]
        
        if not stocks:
            st.error("No stocks loaded for scanning")
            return
        
        scan_start_time = datetime.now()
        
        if not auto_triggered:
            st.info(f"🔍 Scanning {len(stocks)} stocks for EMA crossovers...")
            progress_bar = st.progress(0)
            status_text = st.empty()
        
        # Initialize APIs
        dhan_api = DhanAPI(dhan_token)
        telegram_bot = TelegramBot(telegram_token, telegram_chat_id)
        
        # Scan stocks in batches for better progress tracking
        alerts = []
        processed = 0
        failed = 0
        batch_size = 20
        
        for i in range(0, len(stocks), batch_size):
            batch = stocks[i:i + batch_size]
            
            # Process batch
            batch_alerts = scan_stocks_parallel(batch, dhan_api, max_workers=6)
            alerts.extend(batch_alerts)
            
            # Count failed stocks (for logging)
            successful_in_batch = len([a for a in batch_alerts if a])
            failed += len(batch) - successful_in_batch
            processed += len(batch)
            
            # Update progress
            if not auto_triggered:
                progress = min(processed / len(stocks), 1.0)
                progress_bar.progress(progress)
                status_text.text(f"Processed {processed}/{len(stocks)} stocks... Found {len(alerts)} crossovers")
            
            # Small delay between batches
            time.sleep(0.5)
        
        # Filter out recent duplicates
        new_alerts = []
        for alert in alerts:
            alert_key = f"{alert.symbol}_{alert.cross_type}"
            if (alert_key not in st.session_state.last_alerts or 
                datetime.now() - st.session_state.last_alerts[alert_key] > timedelta(hours=1)):
                new_alerts.append(alert)
                st.session_state.last_alerts[alert_key] = datetime.now()
        
        # Send alerts
        alerts_sent = 0
        telegram_errors = 0
        
        if new_alerts:
            for alert in new_alerts:
                message = telegram_bot.format_alert_message(alert, alert.symbol)
                if telegram_bot.send_message(message):
                    alerts_sent += 1
                    if not auto_triggered:
                        st.success(f"✅ {alert.cross_type} alert sent for {alert.symbol}")
                else:
                    telegram_errors += 1
                    if not auto_triggered:
                        st.error(f"❌ Failed to send alert for {alert.symbol}")
                
                time.sleep(1)  # Rate limiting
        
        # Send summary for auto-triggered scans
        if auto_triggered and new_alerts:
            scan_duration = (datetime.now() - scan_start_time).total_seconds()
            summary = f"""
📊 <b>Auto-Scan Summary</b>

🔍 Stocks Scanned: {len(stocks)}
⏱️ Scan Duration: {scan_duration:.1f}s
🚨 New Alerts: {len(new_alerts)}
📈 Bullish: {len([a for a in new_alerts if a.cross_type == 'BULLISH'])}
📉 Bearish: {len([a for a in new_alerts if a.cross_type == 'BEARISH'])}

🕒 Time: {datetime.now().strftime('%H:%M:%S')}
            """
            telegram_bot.send_message(summary.strip())
        
        # Update session state
        st.session_state.last_scan = datetime.now()
        st.session_state.scan_count += 1
        st.session_state.alerts_sent += alerts_sent
        
        # Clean up progress indicators
        if not auto_triggered:
            progress_bar.empty()
            status_text.empty()
            
            # Show results
            scan_duration = (datetime.now() - scan_start_time).total_seconds()
            
            if new_alerts:
                st.success(f"""
                ✅ **Scan Complete!**
                - Duration: {scan_duration:.1f} seconds
                - Stocks scanned: {len(stocks)} 
                - New crossovers found: {len(new_alerts)}
                - Alerts sent: {alerts_sent}
                - Failed requests: {failed}
                """)
            else:
                st.info(f"""
                ℹ️ **Scan Complete**
                - Duration: {scan_duration:.1f} seconds  
                - Stocks scanned: {len(stocks)}
                - No new crossovers detected
                - Failed requests: {failed}
                """)
            
            # Show breakdown of alerts
            if new_alerts:
                bullish = [a for a in new_alerts if a.cross_type == 'BULLISH']
                bearish = [a for a in new_alerts if a.cross_type == 'BEARISH']
                
                col1, col2 = st.columns(2)
                if bullish:
                    with col1:
                        st.success(f"🟢 **Bullish Crossovers ({len(bullish)})**")
                        for alert in bullish:
                            st.write(f"• {alert.symbol} - ₹{alert.price:.2f}")
                
                if bearish:
                    with col2:
                        st.error(f"🔴 **Bearish Crossovers ({len(bearish)})**")
                        for alert in bearish:
                            st.write(f"• {alert.symbol} - ₹{alert.price:.2f}")
        
        # Force garbage collection to free memory
        gc.collect()
        
    except Exception as e:
        error_msg = f"Error during scan: {str(e)}"
        logger.error(error_msg)
        
        if not auto_triggered:
            st.error(error_msg)
        else:
            # Send error notification for auto scans
            try:
                telegram_bot = TelegramBot(telegram_token, telegram_chat_id)
                telegram_bot.send_message(f"❌ <b>EMA Monitor Error</b>\n\n{error_msg}")
            except:
                pass
    
    finally:
        st.session_state.scan_in_progress = False

if __name__ == "__main__":
    main()import streamlit as st
import pandas as pd
import requests
import time
import logging
import os
from datetime import datetime, timedelta
import json
from typing import Dict, List, Optional
from dataclasses import dataclass
import plotly.graph_objects as go
import plotly.express as px
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import queue

# Configure page
st.set_page_config(
    page_title="EMA Crossover Monitor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class StockConfig:
    symbol: str
    security_id: str
    exchange_segment: str
    instrument: str
    expiry_code: Optional[int] = 0
    fast_ema: int = 9
    slow_ema: int = 21

@dataclass
class EMAAlert:
    symbol: str
    security_id: str
    cross_type: str
    price: float
    fast_ema: float
    slow_ema: float
    timestamp: datetime
    volume: int

class DhanAPI:
    def __init__(self, access_token: str):
        self.base_url = "https://api.dhan.co/v2"
        self.headers = {
            "Content-Type": "application/json",
            "access-token": access_token
        }
    
    def fetch_historical_data(self, stock_config: StockConfig, days: int = 30) -> Optional[pd.DataFrame]:
        """Fetch historical data for a single stock"""
        url = f"{self.base_url}/charts/historical"
        
        to_date = datetime.now()
        from_date = to_date - timedelta(days=days)
        
        payload = {
            "securityId": str(stock_config.security_id),
            "exchangeSegment": stock_config.exchange_segment,
            "instrument": stock_config.instrument,
            "fromDate": from_date.strftime("%Y-%m-%d"),
            "toDate": to_date.strftime("%Y-%m-%d")
        }
        
        if stock_config.instrument in ["FUTIDX", "FUTSTK", "FUTCOM"] and stock_config.expiry_code:
            payload["expiryCode"] = stock_config.expiry_code
        
        try:
            response = requests.post(url, json=payload, headers=self.headers, timeout=30)
            
            if response.status_code != 200:
                logger.error(f"API Error for {stock_config.symbol}: {response.status_code}")
                return None
                
            data = response.json()
            
            if not data or "timestamp" not in data:
                return None
            
            df = pd.DataFrame({
                "timestamp": data["timestamp"],
                "open": data["open"],
                "high": data["high"],
                "low": data["low"],
                "close": data["close"],
                "volume": data["volume"]
            })
            
            df["datetime"] = pd.to_datetime(df["timestamp"], unit="s")
            df = df.sort_values("datetime").reset_index(drop=True)
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching data for {stock_config.symbol}: {str(e)}")
            return None

class EMAAnalyzer:
    @staticmethod
    def calculate_ema(df: pd.DataFrame, span: int) -> pd.Series:
        return df["close"].ewm(span=span, adjust=False).mean()
    
    @staticmethod
    def detect_crossover(df: pd.DataFrame, fast_span: int, slow_span: int) -> Optional[EMAAlert]:
        if len(df) < max(fast_span, slow_span) + 1:
            return None
        
        df["ema_fast"] = EMAAnalyzer.calculate_ema(df, fast_span)
        df["ema_slow"] = EMAAnalyzer.calculate_ema(df, slow_span)
        
        if len(df) < 2:
            return None
            
        curr_fast = df["ema_fast"].iloc[-1]
        curr_slow = df["ema_slow"].iloc[-1]
        curr_price = df["close"].iloc[-1]
        curr_volume = df["volume"].iloc[-1]
        
        prev_fast = df["ema_fast"].iloc[-2]
        prev_slow = df["ema_slow"].iloc[-2]
        
        cross_type = None
        
        if prev_fast <= prev_slow and curr_fast > curr_slow:
            cross_type = "BULLISH"
        elif prev_fast >= prev_slow and curr_fast < curr_slow:
            cross_type = "BEARISH"
        
        if cross_type:
            return EMAAlert(
                symbol="",
                security_id="",
                cross_type=cross_type,
                price=curr_price,
                fast_ema=curr_fast,
                slow_ema=curr_slow,
                timestamp=datetime.now(),
                volume=curr_volume
            )
        
        return None

class TelegramBot:
    def __init__(self, bot_token: str, chat_id: str):
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.base_url = f"https://api.telegram.org/bot{bot_token}"
    
    def send_message(self, message: str) -> bool:
        url = f"{self.base_url}/sendMessage"
        
        payload = {
            "chat_id": self.chat_id,
            "text": message,
            "parse_mode": "HTML"
        }
        
        try:
            response = requests.post(url, json=payload, timeout=10)
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Error sending Telegram message: {str(e)}")
            return False
    
    def format_alert_message(self, alert: EMAAlert, symbol: str) -> str:
        emoji = "🟢" if alert.cross_type == "BULLISH" else "🔴"
        direction = "📈" if alert.cross_type == "BULLISH" else "📉"
        
        message = f"""
{emoji} <b>EMA CROSSOVER ALERT</b> {direction}

<b>Symbol:</b> {symbol}
<b>Security ID:</b> {alert.security_id}
<b>Cross Type:</b> {alert.cross_type}

<b>Price:</b> ₹{alert.price:.2f}
<b>Fast EMA:</b> ₹{alert.fast_ema:.2f}
<b>Slow EMA:</b> ₹{alert.slow_ema:.2f}
<b>Volume:</b> {alert.volume:,}

<b>Time:</b> {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}
        """
        return message.strip()

def load_stocks_from_github(github_url: str) -> List[StockConfig]:
    """Load stocks from GitHub raw CSV URL"""
    try:
        df = pd.read_csv(github_url)
        stocks = []
        
        for _, row in df.iterrows():
            stock = StockConfig(
                symbol=row['symbol'],
                security_id=str(row['security_id']),
                exchange_segment=row['exchange_segment'],
                instrument=row['instrument'],
                expiry_code=int(row.get('expiry_code', 0)),
                fast_ema=int(row.get('fast_ema', 9)),
                slow_ema=int(row.get('slow_ema', 21))
            )
            stocks.append(stock)
        
        return stocks
    except Exception as e:
        st.error(f"Error loading stocks from GitHub: {e}")
        return []

def process_single_stock(stock_config: StockConfig, dhan_api: DhanAPI) -> Optional[EMAAlert]:
    """Process a single stock for EMA crossover"""
    try:
        df = dhan_api.fetch_historical_data(stock_config)
        if df is None or df.empty:
            return None
        
        alert = EMAAnalyzer.detect_crossover(df, stock_config.fast_ema, stock_config.slow_ema)
        if alert:
            alert.symbol = stock_config.symbol
            alert.security_id = stock_config.security_id
            return alert
        
        return None
    except Exception as e:
        logger.error(f"Error processing {stock_config.symbol}: {str(e)}")
        return None

def scan_stocks_parallel(stocks: List[StockConfig], dhan_api: DhanAPI, max_workers: int = 10) -> List[EMAAlert]:
    """Scan stocks in parallel and return alerts"""
    alerts = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_stock = {
            executor.submit(process_single_stock, stock, dhan_api): stock 
            for stock in stocks
        }
        
        for future in as_completed(future_to_stock):
            try:
                alert = future.result(timeout=60)
                if alert:
                    alerts.append(alert)
            except Exception as e:
                logger.error(f"Error in parallel processing: {e}")
    
    return alerts

# Initialize session state
if 'last_scan' not in st.session_state:
    st.session_state.last_scan = None
if 'scan_count' not in st.session_state:
    st.session_state.scan_count = 0
if 'alerts_sent' not in st.session_state:
    st.session_state.alerts_sent = 0
if 'last_alerts' not in st.session_state:
    st.session_state.last_alerts = {}
if 'monitoring_active' not in st.session_state:
    st.session_state.monitoring_active = False

def main():
    st.title("📈 EMA Crossover Monitor")
    st.markdown("Real-time EMA crossover detection with Telegram alerts")
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Environment variables or manual input
        dhan_token = st.text_input(
            "DhanHQ API Token", 
            value=os.getenv('DHAN_TOKEN', ''),
            type="password",
            help="Your DhanHQ API access token"
        )
        
        telegram_token = st.text_input(
            "Telegram Bot Token", 
            value=os.getenv('TELEGRAM_BOT_TOKEN', ''),
            type="password",
            help="Your Telegram bot token"
        )
        
        telegram_chat_id = st.text_input(
            "Telegram Chat ID", 
            value=os.getenv('TELEGRAM_CHAT_ID', ''),
            help="Your Telegram chat ID"
        )
        
        github_csv_url = st.text_input(
            "GitHub CSV URL",
            value=os.getenv('GITHUB_CSV_URL', ''),
            help="Raw GitHub URL to your stocks CSV file"
        )
        
        scan_interval = st.slider(
            "Scan Interval (minutes)",
            min_value=1,
            max_value=60,
            value=int(os.getenv('SCAN_INTERVAL_MINUTES', 15))
        )
        
        st.markdown("---")
        
        # Manual scan button
        if st.button("🔍 Manual Scan", type="primary"):
            if not all([dhan_token, telegram_token, telegram_chat_id]):
                st.error("Please provide all required tokens")
            else:
                perform_scan(dhan_token, telegram_token, telegram_chat_id, github_csv_url)
        
        # Auto-monitoring toggle
        auto_monitor = st.checkbox(
            "🔄 Auto Monitor", 
            value=st.session_state.monitoring_active,
            help=f"Automatically scan every {scan_interval} minutes"
        )
        
        if auto_monitor != st.session_state.monitoring_active:
            st.session_state.monitoring_active = auto_monitor
            if auto_monitor:
                st.success("Auto-monitoring enabled!")
            else:
                st.info("Auto-monitoring disabled")
    
    # Main content area
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Scans", 
            st.session_state.scan_count,
            delta=None
        )
    
    with col2:
        st.metric(
            "Alerts Sent", 
            st.session_state.alerts_sent,
            delta=None
        )
    
    with col3:
        if st.session_state.last_scan:
            time_diff = datetime.now() - st.session_state.last_scan
            st.metric(
                "Last Scan", 
                f"{int(time_diff.total_seconds() / 60)} min ago"
            )
        else:
            st.metric("Last Scan", "Never")
    
    with col4:
        status = "🟢 Active" if st.session_state.monitoring_active else "⚪ Inactive"
        st.metric("Status", status)
    
    # Auto-refresh logic for continuous monitoring
    if st.session_state.monitoring_active and all([dhan_token, telegram_token, telegram_chat_id]):
        if (st.session_state.last_scan is None or 
            datetime.now() - st.session_state.last_scan >= timedelta(minutes=scan_interval)):
            
            with st.spinner("Auto-scanning stocks..."):
                perform_scan(dhan_token, telegram_token, telegram_chat_id, github_csv_url)
    
    # Display recent activity
    st.markdown("---")
    st.subheader("📊 Recent Activity")
    
    if st.session_state.last_alerts:
        alert_df = []
        for key, timestamp in st.session_state.last_alerts.items():
            symbol, cross_type = key.split('_', 1)
            alert_df.append({
                'Symbol': symbol,
                'Cross Type': cross_type,
                'Time': timestamp.strftime('%Y-%m-%d %H:%M:%S')
            })
        
        if alert_df:
            df = pd.DataFrame(alert_df)
            st.dataframe(df, use_container_width=True)
    else:
        st.info("No recent alerts")
    
    # Health check endpoint for UptimeRobot
    if st.query_params.get("health") == "check":
        st.success("✅ EMA Monitor is healthy and running")
        st.json({
            "status": "healthy",
            "last_scan": st.session_state.last_scan.isoformat() if st.session_state.last_scan else None,
            "scan_count": st.session_state.scan_count,
            "alerts_sent": st.session_state.alerts_sent,
            "monitoring_active": st.session_state.monitoring_active
        })
    
    # Auto-refresh for monitoring
    if st.session_state.monitoring_active:
        time.sleep(2)
        st.rerun()

def perform_scan(dhan_token: str, telegram_token: str, telegram_chat_id: str, github_csv_url: str):
    """Perform a complete EMA crossover scan"""
    try:
        # Load stocks
        if github_csv_url:
            stocks = load_stocks_from_github(github_csv_url)
        else:
            # Default sample stocks if no GitHub URL
            stocks = [
                StockConfig("TCS", "11536", "NSE_EQ", "EQUITY", 0, 9, 21),
                StockConfig("RELIANCE", "2885", "NSE_EQ", "EQUITY", 0, 9, 21),
                StockConfig("INFY", "1594", "NSE_EQ", "EQUITY", 0, 12, 26),
            ]
        
        if not stocks:
            st.error("No stocks loaded for scanning")
            return
        
        st.info(f"Scanning {len(stocks)} stocks for EMA crossovers...")
        
        # Initialize APIs
        dhan_api = DhanAPI(dhan_token)
        telegram_bot = TelegramBot(telegram_token, telegram_chat_id)
        
        # Progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Scan stocks
        alerts = []
        processed = 0
        
        # Process in smaller batches to show progress
        batch_size = 10
        for i in range(0, len(stocks), batch_size):
            batch = stocks[i:i + batch_size]
            batch_alerts = scan_stocks_parallel(batch, dhan_api, max_workers=5)
            alerts.extend(batch_alerts)
            
            processed += len(batch)
            progress = min(processed / len(stocks), 1.0)
            progress_bar.progress(progress)
            status_text.text(f"Processed {processed}/{len(stocks)} stocks...")
        
        # Filter out recent duplicates
        new_alerts = []
        for alert in alerts:
            alert_key = f"{alert.symbol}_{alert.cross_type}"
            if alert_key not in st.session_state.last_alerts or \
               datetime.now() - st.session_state.last_alerts[alert_key] > timedelta(hours=1):
                new_alerts.append(alert)
                st.session_state.last_alerts[alert_key] = datetime.now()
        
        # Send alerts
        alerts_sent = 0
        if new_alerts:
            for alert in new_alerts:
                message = telegram_bot.format_alert_message(alert, alert.symbol)
                if telegram_bot.send_message(message):
                    alerts_sent += 1
                    st.success(f"✅ {alert.cross_type} alert sent for {alert.symbol}")
                else:
                    st.error(f"❌ Failed to send alert for {alert.symbol}")
                
                time.sleep(1)  # Rate limiting
        
        # Update session state
        st.session_state.last_scan = datetime.now()
        st.session_state.scan_count += 1
        st.session_state.alerts_sent += alerts_sent
        
        # Summary
        if new_alerts:
            st.success(f"Scan complete! Found {len(new_alerts)} new crossovers, sent {alerts_sent} alerts")
        else:
            st.info("Scan complete! No new EMA crossovers detected")
            
        progress_bar.empty()
        status_text.empty()
        
    except Exception as e:
        st.error(f"Error during scan: {str(e)}")
        logger.error(f"Scan error: {e}")

if __name__ == "__main__":
    main()
