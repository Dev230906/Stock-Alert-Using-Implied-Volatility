# %%

import yfinance as yf
import pandas as pd
import numpy as np
import datetime as dt
import math
from scipy.stats import norm
from scipy.optimize import brentq
import smtplib
import os
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
from dotenv import load_dotenv


RISK_FREE_RATE = 0.03  




def bsm_put_price(S, K, r, sigma, T):
    """Black-Scholes European put price (no dividends)."""
    if T <= 0:
        return max(K - S, 0.0)
    sqrtT = math.sqrt(T)
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT

    call = S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
    put = call - S + K * math.exp(-r * T)
    return put


def bsm_vega(S, K, r, sigma, T):
    """Vega of option (derivative wrt volatility)."""
    if T <= 0:
        return 0.0
    sqrtT = math.sqrt(T)
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrtT)
    return S * norm.pdf(d1) * sqrtT


def implied_vol_put_nr_bisect(market_price, S, K, r, T,
                              sigma0=0.25, tol=1e-6, max_iter=100):

    if market_price is None or market_price <= 0 or T <= 0:
        return np.nan

    intrinsic = max(K - S, 0.0)
    if market_price < intrinsic:
        market_price = intrinsic

    # Newton-Raphson
    sigma = float(sigma0)
    for _ in range(max_iter):
        price = bsm_put_price(S, K, r, sigma, T)
        diff = price - market_price

        if abs(diff) < tol:
            return max(sigma, 0.0)

        vega = bsm_vega(S, K, r, sigma, T)
        if vega < 1e-8:
            break

        sigma = sigma - diff / vega
        sigma = min(max(sigma, 1e-6), 5.0)

    # brentq fallback
    def f(s):
        return bsm_put_price(S, K, r, s, T) - market_price

    low, high = 1e-6, 5.0
    try:
        if f(low) * f(high) > 0:
            high = 10.0
            if f(low) * f(high) > 0:
                return np.nan
        return brentq(f, low, high, xtol=tol, maxiter=200)
    except Exception:
        return np.nan




def get_otm_put_iv(ticker_symbol, r=RISK_FREE_RATE):
    ticker = yf.Ticker(ticker_symbol)
    ticker._download_options()

    today = dt.datetime.today().date()

    # convert expirations
    expirations = [
        dt.datetime.strptime(d, "%Y-%m-%d").date()
        for d in ticker._expirations
    ]

    # nearest expirations within 7 days. change as per your needs
    nearby = [d for d in expirations if 0 < (d - today).days < 7]
    if len(nearby) == 0:
        return None

    
    puts_list = []
    for exp in nearby:
        oc = ticker.option_chain(exp.strftime("%Y-%m-%d"))
        df_puts = oc.puts.copy()
        df_puts["expiration"] = exp
        puts_list.append(df_puts)

    all_puts = pd.concat(puts_list, ignore_index=True)

    
    spot = ticker.history(period="1d")["Close"].iloc[-1]

    otm_puts = all_puts[all_puts["strike"] < spot]
    if otm_puts.empty:
        return None

    
    otm_puts = otm_puts.sort_values(
        by=["openInterest", "volume"],
        ascending=False
    )
    row = otm_puts.iloc[0]


    bid = row.get("bid", np.nan)
    ask = row.get("ask", np.nan)
    last = row.get("lastPrice", np.nan)

    if (not np.isnan(bid)) and (not np.isnan(ask)) and ask > 0:
        mid = 0.5 * (bid + ask)
    elif (not np.isnan(last)) and last > 0:
        mid = last
    else:
        return None

    K = float(row["strike"])
    days = (row["expiration"] - today).days
    T = max(days / 252.0, 1e-6)

    iv = implied_vol_put_nr_bisect(
        market_price=mid,
        S=spot,
        K=K,
        r=r,
        T=T,
        sigma0=0.25
    )

    return {
        "ticker": ticker_symbol,
        "spot": spot,
        "strike": K,
        "expiry": row["expiration"],
        "iv": iv,
        "mid": mid,
        "volume": row.get("volume", np.nan),
        "openInterest": row.get("openInterest", np.nan),
    }


# %%


def realized_vol_21d(ticker_symbol):
    """
    Compute 21-day annualized realized volatility.
    """
    df = yf.download(ticker_symbol, period="2mo", interval="1d", progress=False)
    
    if df.shape[0] < 22:
        return np.nan

    df["ret"] = df["Close"].pct_change()
    rv = df["ret"].std() * np.sqrt(252)   
    return rv


stocks = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA",
          "NVDA", "META", "JPM", "UNH", "V",
          "DIS", "MA", "NFLX", "PG", "BAC",
          "KO", "PFE", "WMT", "XOM", "HD"]


data = []
for s in stocks:
    iv_dict = get_otm_put_iv(s)
    if iv_dict is not None:
        data.append(iv_dict)

df_iv = pd.DataFrame(data)


df_iv["rv_21d"] = df_iv["ticker"].apply(realized_vol_21d)

# Compute IV / RV ratio
df_iv["iv_rv_ratio"] = df_iv["iv"] / df_iv["rv_21d"]

print(df_iv)


# %%
df_iv

# %%

df_iv["iv_percentile"] = df_iv["iv"].rank(pct=True)


iv_rv_min = df_iv["iv_rv_ratio"].min()
iv_rv_max = df_iv["iv_rv_ratio"].max()
df_iv["iv_rv_scaled"] = (df_iv["iv_rv_ratio"] - iv_rv_min) / (iv_rv_max - iv_rv_min)

print(df_iv)


# %%
#Compute Put-Call Skew using calculated call IV

def bsm_call_price(S, K, r, sigma, T):
    """Black-Scholes European call price (no dividends)."""
    if T <= 0:
        return max(S - K, 0.0)
    sqrtT = math.sqrt(T)
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT
    return S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)

def implied_vol_call_nr_bisect(market_price, S, K, r, T,
                               sigma0=0.25, tol=1e-6, max_iter=100):
    """Compute call IV using Newton-Raphson + brentq fallback."""
    if market_price is None or market_price <= 0 or T <= 0:
        return np.nan

    intrinsic = max(S - K, 0.0)
    if market_price < intrinsic:
        market_price = intrinsic

    # Newton-Raphson
    sigma = float(sigma0)
    for _ in range(max_iter):
        price = bsm_call_price(S, K, r, sigma, T)
        diff = price - market_price
        vega = bsm_vega(S, K, r, sigma, T)
        if abs(diff) < tol:
            return max(sigma, 0.0)
        if vega < 1e-8:
            break
        sigma = min(max(sigma - diff / vega, 1e-6), 5.0)

    # Brentq fallback
    def f(s): return bsm_call_price(S, K, r, s, T) - market_price
    low, high = 1e-6, 5.0
    try:
        if f(low) * f(high) > 0:
            high = 10.0
            if f(low) * f(high) > 0:
                return np.nan
        return brentq(f, low, high, xtol=tol, maxiter=200)
    except Exception:
        return np.nan

def get_otm_call_iv(ticker_symbol, r=RISK_FREE_RATE):
    """Fetch highest liquidity OTM call and compute IV."""
    ticker = yf.Ticker(ticker_symbol)
    ticker._download_options()
    today = dt.datetime.today().date()
    expirations = [dt.datetime.strptime(d, "%Y-%m-%d").date() for d in ticker._expirations]
    nearby = [d for d in expirations if 0 < (d - today).days < 7]
    if len(nearby) == 0:
        return None

    calls_list = []
    for exp in nearby:
        oc = ticker.option_chain(exp.strftime("%Y-%m-%d"))
        df_calls = oc.calls.copy()
        df_calls["expiration"] = exp
        calls_list.append(df_calls)
    all_calls = pd.concat(calls_list, ignore_index=True)
    spot = ticker.history(period="1d")["Close"].iloc[-1]
    otm_calls = all_calls[all_calls["strike"] > spot]
    if otm_calls.empty:
        return None

    otm_calls = otm_calls.sort_values(by=["openInterest", "volume"], ascending=False)
    row = otm_calls.iloc[0]

    bid = row.get("bid", np.nan)
    ask = row.get("ask", np.nan)
    last = row.get("lastPrice", np.nan)

    if (not np.isnan(bid)) and (not np.isnan(ask)) and ask > 0:
        mid = 0.5 * (bid + ask)
    elif (not np.isnan(last)) and last > 0:
        mid = last
    else:
        return None

    K = float(row["strike"])
    days = (row["expiration"] - today).days
    T = max(days / 252.0, 1e-6)

    iv = implied_vol_call_nr_bisect(market_price=mid, S=spot, K=K, r=r, T=T)
    return iv

# Compute skew using calculated put & call IV
put_call_skew = []
for s in stocks:
    put_iv = df_iv.loc[df_iv["ticker"] == s, "iv"].values
    call_iv = get_otm_call_iv(s)
    if len(put_iv) == 0 or call_iv is None or np.isnan(call_iv) or call_iv == 0:
        put_call_skew.append(np.nan)
    else:
        put_call_skew.append(put_iv[0] / call_iv)

df_iv["put_call_skew"] = put_call_skew

print(df_iv)


# %%

# Weight parameters (just for demonstration, adjust as needed)
weight_iv_percentile = 0.4
weight_iv_rv_scaled = 0.4
weight_put_call_skew = 0.2

# Scale put-call skew to 0-1 for consistency
skew_min = np.nanmin(df_iv["put_call_skew"])
skew_max = np.nanmax(df_iv["put_call_skew"])
df_iv["skew_scaled"] = (df_iv["put_call_skew"] - skew_min) / (skew_max - skew_min)

# Compute final score
df_iv["score"] = (
    weight_iv_percentile * df_iv["iv_percentile"] +
    weight_iv_rv_scaled * df_iv["iv_rv_scaled"] +
    weight_put_call_skew * df_iv["skew_scaled"]
)

# Generate binary signal based on threshold
threshold = 0.75
df_iv["signal"] = df_iv["score"].apply(lambda x: 1 if x >= threshold else 0)

print(df_iv[["ticker", "score", "signal"]])


# %%
# Separate stocks with indicator = 1
df_flagged = df_iv[df_iv["signal"] == 1].copy()


df_flagged.reset_index(drop=True, inplace=True)

print("Stocks with signal = 1:")
print(df_flagged)


# %%
#load_dotenv(r"C:your_path/iv.env")

# %%
#smtp_server = "smtp.gmail.com"
#smtp_port = 587
#email_id = os.getenv("email_id")
#email_password = os.getenv("email_password")

#(Use a dotenv file to store sensitive information)

# %%
#from tabulate import tabulate



# %%
#table_str = tabulate(df_flagged, headers="keys", tablefmt="pretty", showindex=False)

# %%
#def send_email(subject, body, recipient_email):
    #try:
        #msg = MIMEMultipart()
        #msg['From'] = email_id
        #msg['To'] = recipient_email
        #msg['Subject'] = subject

    # Attach the body text
        #msg.attach(MIMEText(body, 'plain'))

    # Send the email
        #with smtplib.SMTP(smtp_server, smtp_port) as server:
          #server.starttls()
          #server.login(email_id, email_password)
          #server.sendmail(email_id, recipient_email, msg.as_string())

        #print("Email sent successfully")
    #except Exception as e:  
        #print(f"Failed to send email: {e}")

#if __name__ == "__main__":
    #subject = "Flagged Stocks with Signal = 1"
    #body = (
    #"Hello,\n\n"
    #"Here is your daily alert for stocks that triggered **Signal = 1**.\n"
    #"These stocks may be showing high risk based on IV score.\n\n"
    #"-----------------------------------------\n"
    #"SUMMARY\n"
    #f"Total flagged stocks: {len(df_flagged)}\n"
    #"-----------------------------------------\n\n"
    #"📊 FLAGGED STOCK DETAILS:\n"
    #f"{table_str}\n\n"
    #"-----------------------------------------\n"
    #"This is an automated alert. Monitor these tickers closely.\n"
    #"Regards,\n"
    #"Your bot")

    #recipient_email = "putyourmail"



# %%
#send_email(subject, body, recipient_email)

# %%


# %%



