import os, schedule, time, json, feedparser, threading, asyncio
import yfinance as yf
import pandas as pd
import ta
from nsepython import nse_eq_symbols, equity_history
from telegram import Update, Bot
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes
from openai import OpenAI
from datetime import datetime, timedelta
from transformers import pipeline

BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
CHAT_ID = int(os.getenv('TELEGRAM_CHAT_ID'))
LLM = OpenAI()
bot = Bot(token=BOT_TOKEN)

# Load FinBERT once for speed
try:
    sentiment_pipe = pipeline("sentiment-analysis", model="ProsusAI/finbert")
except:
    sentiment_pipe = None # fallback if low RAM

def get_universe():
    try:
        symbols = nse_eq_symbols()
        return [s for s in symbols if s.isalpha()][:250] # 250 to save CPU/RAM
    except:
        return ['RELIANCE','TCS','HDFCBANK','INFY','ICICIBANK','SBIN','ITC','LT','AXISBANK','MARUTI']

def get_delivery_pct(symbol):
    try:
        to_date = datetime.now()
        from_date = to_date - timedelta(days=5)
        df = equity_history(symbol, "EQ", from_date.strftime('%d-%m-%Y'), to_date.strftime('%d-%m-%Y'))
        if not df.empty:
            return float(df.iloc[-1]['DeliveryPercent'])
    except: return 0
    return 0

def get_news_sentiment(symbol):
    try:
        url = f"https://news.google.com/rss/search?q={symbol}+NSE+stock&hl=en-IN&gl=IN&ceid=IN:en"
        feed = feedparser.parse(url)
        headlines = [e.title for e in feed.entries[:3]]
        if not headlines: return 0, []

        if sentiment_pipe:
            scores = [s['score'] if s['label']=='positive' else -s['score'] if s['label']=='negative' else 0
                      for s in sentiment_pipe(headlines)]
            avg_sent = sum(scores) / len(scores)
        else: # fallback keyword method for low RAM
            headlines_l = [h.lower() for h in headlines]
            bad_words = ['scam','penalty','fraud','ban','default','probe','loss']
            avg_sent = -0.5 if any(w in h for h in headlines_l for w in bad_words) else 0.2
        return round(avg_sent, 2), headlines
    except: return 0, []

def analyze_stock(symbol):
    try:
        ticker = symbol + '.NS'
        df = yf.download(ticker, period='3mo', interval='1d', progress=False)
        if len(df) < 50: return None

        df['20EMA'] = ta.trend.ema_indicator(df['Close'], 20)
        df['50EMA'] = ta.trend.ema_indicator(df['Close'], 50)
        df['RSI'] = ta.momentum.rsi(df['Close'], 14)
        df['ATR'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'], 14)
        df['20DayHigh'] = df['High'].rolling(20).max()
        df['AvgVol'] = df['Volume'].rolling(20).mean()

        l = df.iloc[-1]
        prev = df.iloc[-2]

        breakout = l['Close'] > prev['20DayHigh'] and l['Volume'] > 1.8 * l['AvgVol']
        pullback = l['Low'] <= l['20EMA'] * 1.02 and l['Close'] > l['20EMA'] and 45 < l['RSI'] < 65
        trend_ok = l['Close'] > l['50EMA'] and l['20EMA'] > l['50EMA']

        if not ((breakout or pullback) and trend_ok and l['Close'] > 50): return None

        del_pct = get_delivery_pct(symbol)
        news_score, headlines = get_news_sentiment(symbol)

        if del_pct < 40 or news_score < -0.4: return None

        return {
            'ticker': symbol,
            'close': round(l['Close'], 2),
            'setup': 'Breakout' if breakout else 'Pullback to 20EMA',
            'rsi': round(l['RSI'], 1),
            'vol_x': round(l['Volume']/l['AvgVol'], 1),
            'atr': round(l['ATR'], 2),
            'del_pct': round(del_pct, 1),
            'news_score': news_score,
            'headlines': headlines,
            'stop': round(l['Close'] - 1.5*l['ATR'], 2),
            'target': round(l['Close'] + 3*l['ATR'], 2),
            'entry': round(l['Close'], 2)
        }
    except: return None

def llm_rank_picks(candidates):
    if not candidates: return []
    prompt = f"""You are a swing trading analyst for Indian stocks. Holding 3-10 days.
Pick TOP 5 from these pre-screened stocks. Consider: delivery %, news sentiment, volume spike, R:R.
For each: entry, SL, target, 1-line reason using the data.

Data: {json.dumps(candidates[:15], indent=2)}

Return JSON: {{"stocks":[{{"ticker":"","entry":0,"sl":0,"target":0,"reason":""}}]}}"""

    res = LLM.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role":"user","content":prompt}],
        response_format={"type": "json_object"}
    )
    return json.loads(res.choices[0].message.content).get('stocks', [])

async def run_scan(send_update=True):
    if send_update: await bot.send_message(CHAT_ID, f"🔍 Manual scan started... {datetime.now().strftime('%d-%b %H:%M')}")
    universe = get_universe()
    candidates = []

    for i, symbol in enumerate(universe):
        if i % 25 == 0: print(f"Scanned {i}/{len(universe)}")
        res = analyze_stock(symbol)
        if res: candidates.append(res)
        time.sleep(0.3)

    candidates = sorted(candidates, key=lambda x: (x['vol_x'], x['del_pct']), reverse=True)
    top5 = llm_rank_picks(candidates)

    if not top5:
        await bot.send_message(CHAT_ID, "No setups passed delivery + news filter today. No trade = good trade.")
        return

    msg = "*📈 Swing Picks - NSE*\n"
    msg += f"_Scan: {datetime.now().strftime('%d %b %Y %H:%M')}_\n\n"
    for s in top5:
        rr = round((s['target']-s['entry'])/(s['entry']-s['sl']),1)
        msg += f"*{s['ticker']}* | {s['setup']} | Del: {s['del_pct']}% | News: {s['news_score']}\n"
        msg += f"Entry: ₹{s['entry']} | SL: ₹{s['sl']} | TGT: ₹{s['target']} | R:R 1:{rr}\n"
        msg += f"Reason: {s['reason']}\n\n"
    msg += "_Educational only. Not SEBI advice._"

    await bot.send_message(CHAT_ID, msg, parse_mode='Markdown')

# Telegram Commands
async def scan_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_chat.id!= CHAT_ID: return
    await update.message.reply_text("Running scan... Takes 2-3 mins. Will post results here.")
    await run_scan(send_update=False)

async def status_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_chat.id!= CHAT_ID: return
    await update.message.reply_text(f"Bot online ✅\nTime: {datetime.now().strftime('%d %b %H:%M IST')}\nNext auto scan: Weekdays 8:45 AM")

def schedule_runner():
    def job():
        asyncio.run(run_scan())
    schedule.every().monday.at("08:45").do(job)
    schedule.every().tuesday.at("08:45").do(job)
    schedule.every().wednesday.at("08:45").do(job)
    schedule.every().thursday.at("08:45").do(job)
    schedule.every().friday.at("08:45").do(job)
    while True:
        schedule.run_pending()
        time.sleep(60)

if __name__ == "__main__":
    # Start scheduler in background thread
    threading.Thread(target=schedule_runner, daemon=True).start()

    # Start Telegram bot
    app = ApplicationBuilder().token(BOT_TOKEN).build()
    app.add_handler(CommandHandler("scan", scan_command))
    app.add_handler(CommandHandler("status", status_command))
    print("Bot started. Send /scan in Telegram to test.")
    app.run_polling()