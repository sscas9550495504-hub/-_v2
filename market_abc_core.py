#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""台股 ABC 反彈全市場篩選與無未來函數的事件研究。

運行：
  pip install pandas numpy requests
  python b_wave_breakout.py --asof 2026-10-20 --workers 10
  python b_wave_breakout.py --data downloaded_daily.csv --asof 2026-10-20

行情：https://github.com/chenyicheng2008/TWSTOCK (來源與品質由第三方提供)
此程式在有網路的電腦執行才會下載行情；不應將未執行結果當作回測績效。
"""
import argparse
import concurrent.futures
import gzip
import io
import os
import re
import time
from pathlib import Path
from urllib.parse import quote

import numpy as np
import pandas as pd
import requests

REPO = 'https://raw.githubusercontent.com/chenyicheng2008/TWSTOCK/main'
BULK = 'https://github.com/chenyicheng2008/TWSTOCK/releases/latest/download/tw_all_daily_1yr.csv.gz'
TREE = 'https://api.github.com/repos/chenyicheng2008/TWSTOCK/git/trees/main?recursive=1'
COST_BUY = 0.001425 + 0.001   # 交易手續費 + 單邊滑價；未打折
COST_SELL = 0.001425 + 0.003 + 0.001  # 手續費 + 普通股證交稅 + 單邊滑價
HORIZONS = (5, 10, 20)
NAMES = {'3090':'日電貿','2472':'立隆電','2375':'凱美','6175':'立敦',
         '6449':'鈺邦','2492':'華新科','3042':'晶技','6284':'佳邦'}


def normalize(df):
    df = df.copy()
    df.columns = [str(k).strip().lower().lstrip('\ufeff') for k in df.columns]
    renamed = {'stock_id':'symbol','stockid':'symbol','ticker':'symbol','code':'symbol',
               'stock':'symbol','datetime':'date','trading_date':'date',
               'trade_volume':'volume', 'tradevolume':'volume', 'adj close':'close'}
    df = df.rename(columns={k:v for k,v in renamed.items() if k in df.columns})
    if 'symbol' not in df: raise ValueError('找不到 symbol/stock_id/ticker 欄位')
    required = ['symbol','date','open','high','low','close','volume']
    if not set(required).issubset(df.columns):
        raise ValueError(f'行情欄位不足，需：{required}；實際：{list(df.columns)}')
    df['symbol'] = df['symbol'].astype(str).str.strip().str.replace(r'\.0$', '', regex=True)
    df = df[df['symbol'].str.fullmatch(r'[1-9][0-9]{3}')].copy()  # 普通股為主；剔除ETF/權證
    df['date'] = pd.to_datetime(df['date'], errors='coerce').dt.normalize()
    for k in ('open','high','low','close','volume'):
        df[k] = pd.to_numeric(df[k], errors='coerce')
    df = df.dropna(subset=required).query('open > 0 and high > 0 and low > 0 and close > 0')
    return df.drop_duplicates(['symbol','date'], keep='last').sort_values(['symbol','date'])


def get_data(args):
    if args.data:
        path = Path(args.data)
        if path.suffix == '.gz':
            with gzip.open(path,'rt',encoding='utf-8-sig') as fp: return normalize(pd.read_csv(fp,dtype={'symbol':str}))
        return normalize(pd.read_csv(path, dtype={'symbol':str}))
    print('[資料] 嘗試 GitHub Release 合併日線 ...', flush=True)
    session = requests.Session()
    session.headers.update({'User-Agent':'Mozilla/5.0 TW-ABC-Research/1.0'})
    try:
        response = session.get(BULK, timeout=75)
        response.raise_for_status()
        raw = response.content
        data = gzip.decompress(raw) if raw[:2] == b'\x1f\x8b' else raw
        df = normalize(pd.read_csv(io.BytesIO(data), dtype={'symbol':str}, low_memory=False))
        if len(df) < 5000: raise ValueError('合併資料過少')
        print(f'[資料] 合併檔：{df.symbol.nunique()} 檔股票、{len(df):,} 筆', flush=True)
        return df
    except Exception as e:
        print(f'[備援] 合併檔不可用：{str(e)[:125]}，改抓個別日線', flush=True)
    tree = session.get(TREE, timeout=45).json()
    codes = sorted(set(m.group(1) for x in tree.get('tree',[]) if x.get('type')=='blob'
                       for m in [re.match(r'^daily/([1-9][0-9]{3})\.csv$', x.get('path',''))] if m))
    codes = [c for c in codes if c]
    if not codes:
        raise RuntimeError('無法取得股票代碼清單。請手動下載全市場資料並用 --data 載入')
    if args.max_symbols: codes = codes[:args.max_symbols]
    print(f'[資料] 個股抓取：{len(codes)} 檔',flush=True)
    def fetch(code):
        url = f'{REPO}/daily/{quote(code)}.csv'
        for attempt in range(3):
            try:
                r = requests.get(url, timeout=20, headers={'User-Agent':'TW-ABC-Research'})
                r.raise_for_status()
                temp = pd.read_csv(io.BytesIO(r.content))
                temp['symbol'] = code
                return temp
            except Exception:
                time.sleep((attempt + 1)*0.5)
        return None
    chunks = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        for i, x in enumerate(pool.map(fetch,codes),1):
            if x is not None: chunks.append(x)
            if i % 250 == 0: print(f'  {i}/{len(codes)} 已處理',flush=True)
    if not chunks: raise RuntimeError('所有下載都失敗。檢查網路，或用 --data 匯入 CSV')
    return normalize(pd.concat(chunks, ignore_index=True))


def enrich(df):
    q = df.copy().reset_index(drop=True)
    c = q['close']
    q['ma5'] = c.rolling(5).mean()
    q['ma20'] = c.rolling(20).mean()
    q['v20'] = q['volume'].shift(1).rolling(20).mean()  # 當日量不能被納入自己的基準
    q['vol_ratio'] = q['volume'] / q['v20'].replace(0,np.nan)
    delta = c.diff()
    gain = delta.clip(lower=0).ewm(alpha=1/14,adjust=False,min_periods=14).mean()
    loss = -delta.clip(upper=0).ewm(alpha=1/14,adjust=False,min_periods=14).mean()
    rs = gain / loss.replace(0,np.nan)
    q['rsi'] = 100 - 100/(1+rs)
    ema12, ema26 = c.ewm(span=12,adjust=False).mean(),c.ewm(span=26,adjust=False).mean()
    q['macd'] = ema12-ema26
    q['macd_signal'] = q['macd'].ewm(span=9,adjust=False).mean()
    q['macd_hist'] = q['macd']-q['macd_signal']
    q['hist_rising'] = q['macd_hist'] > q['macd_hist'].shift(1)
    q['ma5_cross'] = (c>q['ma5']) & (c.shift(1)<=q['ma5'].shift(1))
    return q


def probe_events(code, df, summer_start, summer_end):
    """日資料有限狀態機：先確認跌幅，再確認 A，再等待 B，最後觀察 C。
       所有狀態只依當天及過去資料，嚴禁使用右側確認未來K棒。
    """
    n=len(df)
    if n<90: return [], None
    events=[]
    state='SEARCH'
    p0=ph=a_high=b_low=None
    p0_idx=peak_idx=a_idx=b_idx=None
    detected_at=None
    b_fired=c_fired=False
    current=None
    dte=df['date'].to_numpy()
    high=df['high'].to_numpy(float)
    low=df['low'].to_numpy(float)
    close=df['close'].to_numpy(float)
    volume=df['volume'].to_numpy(float)
    avgvol=df['v20'].to_numpy(float)
    rsi=df['rsi'].to_numpy(float)
    ratio=df['vol_ratio'].to_numpy(float)
    hist=df['macd_hist'].to_numpy(float)
    hist_rise=df['hist_rising'].to_numpy(bool)
    crosses=df['ma5_cross'].to_numpy(bool)
    ma5=df['ma5'].to_numpy(float)
    def reset():
        nonlocal state,p0,ph,a_high,b_low,p0_idx,peak_idx,a_idx,b_idx,b_fired,c_fired,detected_at
        state='SEARCH';p0=ph=a_high=b_low=None
        p0_idx=peak_idx=a_idx=b_idx=None
        b_fired=c_fired=False; detected_at=None
    def signal(kind,i):
        retrace = (a_high-b_low)/(a_high-p0) if a_high>p0 else np.nan
        events.append(dict(symbol=code, name=NAMES.get(code,''),type=kind,
          signal_date=pd.Timestamp(dte[i]).date().isoformat(), signal_index=i,
          crash_peak=ph, bottom=p0, a_high=a_high,b_low=b_low,
          drop_pct=(p0/ph-1)*100, a_gain_pct=(a_high/p0-1)*100,
          b_retrace=retrace,close=close[i], rsi=rsi[i],macd_hist=hist[i],
          hist_rising=bool(hist_rise[i]),vol_ratio=ratio[i],
          days_from_bottom=i-p0_idx, bottom_date=pd.Timestamp(dte[p0_idx]).date().isoformat(),
          in_summer=summer_start<=pd.Timestamp(dte[p0_idx])<=summer_end))

    for i in range(65,n):
        if volume[i] <= 0:  # 停牌日不得觸發或被當作可交易價格
            continue
        if state=='SEARCH':
            prior = high[i-65:i]
            ph0=np.max(prior)
            peak_i=(i-65)+int(np.argmax(prior))
            if ph0>0 and (close[i]/ph0<=0.70) and (i-peak_i <= 48):
                ph=ph0; peak_idx=peak_i
                p0=low[i]; p0_idx=i; state='BOTTOM'
        elif state=='BOTTOM':
            if low[i] < p0:
                p0=low[i];p0_idx=i
            if i-p0_idx>40 or i-peak_idx>105:
                reset(); continue
            if close[i] >= p0*1.18 and i-p0_idx>=2:
                state='A';a_high=high[i];a_idx=i
        elif state=='A':
            if high[i]>a_high:
                a_high=high[i];a_idx=i
            if low[i] <= a_high-0.25*(a_high-p0) and i-a_idx>=2 and a_high>=p0*1.25:
                state='B';b_low=low[i];b_idx=i
            if i-p0_idx>100: reset()
        elif state=='B':
            if low[i]<b_low:
                b_low=low[i];b_idx=i
            retrace=(a_high-b_low)/(a_high-p0)
            if b_low<=p0*1.01 or retrace>=0.96 or i-p0_idx>105:
                reset();continue
            if (not b_fired and i-b_idx>=2 and close[i]>=b_low*1.04
                and (crosses[i] or (close[i]>ma5[i] and close[i-1]<=ma5[i-1]))
                and 35<=rsi[i]<=68 and hist_rise[i] and ratio[i]>=0.75):
                signal('B_提前轉強',i);b_fired=True
            if (close[i]>=a_high*1.01 and close[i-1]<a_high*1.01
                and ratio[i]>=1.30 and rsi[i]>=50 and hist[i]>0):
                signal('C_放量突破',i);c_fired=True;state='C'
        elif state=='C':
            if low[i]<=p0*1.01 or i-p0_idx>110:
                reset(); continue
            # 若跌破 B 低點，已不再是健康的 C 波
            if close[i]<b_low:
                state='FAILED'
        elif state=='FAILED':
            if i-p0_idx>110 or low[i]<=p0:
                reset()

    if state in {'A','B','C','FAILED'} and (n-1-p0_idx)<=110:
        i=n-1
        retrace=(a_high-b_low)/(a_high-p0) if b_low is not None else np.nan
        if state=='B': label='B波整理／等待轉強'
        elif state=='C':
            label='C波突破後拉回' if close[i]<a_high else 'C波突破區'
        elif state=='FAILED': label='C波失敗／跌破B低'
        else: label='A波進行中／尚未確認B'
        current=dict(symbol=code,name=NAMES.get(code,''),date=pd.Timestamp(dte[i]).date().isoformat(),
           close=close[i],stage=label,crash_peak=ph,bottom=p0,a_high=a_high,b_low=b_low,
           b_retrace=retrace,drop_pct=100*(p0/ph-1),rsi=rsi[i],
           macd_hist=hist[i],vol_ratio=ratio[i],ma5=ma5[i],
           since_bottom=i-p0_idx,bottom_date=pd.Timestamp(dte[p0_idx]).date().isoformat(),
           in_summer=summer_start<=pd.Timestamp(dte[p0_idx])<=summer_end)
    return events,current


def evaluate_trade(df, event, hold, stop_pct=0.08, target_pct=0.25):
    t=int(event['signal_index']);n=len(df)
    entry_i=t+1
    exit_i=entry_i+hold-1
    if exit_i>=n: return None   # 未到期持有不可混入已實現勝率
    if df.iloc[entry_i]['volume']<=0: return None
    entry=float(df.iloc[entry_i]['open'])
    if entry<=0: return None
    stop=entry*(1-stop_pct)
    target=entry*(1+target_pct)
    exited=False
    for j in range(entry_i,exit_i+1):
        row=df.iloc[j]
        if row['volume']<=0: continue
        hi,lo,o=float(row['high']),float(row['low']),float(row['open'])
        # 隔日跳空跨過停損/停利時，以較保守、可能成交的開盤價格計
        if o<=stop: sale=o;reason='跳空停損';end=j;exited=True;break
        if o>=target: sale=o;reason='跳空停利';end=j;exited=True;break
        # 當日先後順序未知：同時觸及停損停利，採停損保守原則
        if lo<=stop: sale=stop;reason='停損';end=j;exited=True;break
        if hi>=target: sale=target;reason='停利';end=j;exited=True;break
    if not exited:
        # 若最後一日停牌，該筆結果暫不列為可出場成交
        if df.iloc[exit_i]['volume']<=0: return None
        end=exit_i;sale=float(df.iloc[end]['close']);reason='到期'
    gross=sale/entry-1
    net=sale*(1-COST_SELL)/(entry*(1+COST_BUY))-1
    sub=df.iloc[entry_i:end+1]
    mae=float(sub['low'].min()/entry-1)
    mfe=float(sub['high'].max()/entry-1)
    d=event.copy()
    d.pop('signal_index',None)
    d.update(hold=hold,entry_date=df.iloc[entry_i]['date'].date().isoformat(),
      entry=entry,exit_date=df.iloc[end]['date'].date().isoformat(),exit=sale,
      bars_held=end-entry_i+1,gross_pct=100*gross,net_pct=100*net,
      mae_pct=100*mae,mfe_pct=100*mfe,exit_reason=reason)
    return d


def agg(group):
    s=group['net_pct']
    winners=s[s>0];losers=s[s<0]
    return pd.Series(dict(n=len(s), win_rate_pct=100*(s>0).mean(),
      mean_net_pct=s.mean(),median_net_pct=s.median(),
      p10_pct=s.quantile(.1),p90_pct=s.quantile(.9),
      mean_mae_pct=group['mae_pct'].mean(),
      profit_factor=(winners.sum()/(-losers.sum()) if len(losers) else np.nan)))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--data',help='外部已有的日線合併 CSV 或 CSV.GZ')
    parser.add_argument('--asof',default='2026-10-20')
    parser.add_argument('--out',default='abc_results')
    parser.add_argument('--workers',type=int,default=10)
    parser.add_argument('--max-symbols',type=int,default=0,help='測試用：限制抓取前N檔')
    parser.add_argument('--summer-from',default='2026-06-01')
    parser.add_argument('--summer-to',default='2026-08-31')
    parser.add_argument('--min-turnover',type=float,default=1.0e7,help='近20日日均成交金額估算下限：新台幣元')
    args=parser.parse_args()
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    data=get_data(args)
    data=data[data.date<=pd.Timestamp(args.asof)]
    print(f'[分析] {len(data):,} 筆，{data.symbol.nunique()} 檔，最新日 {data.date.max().date()}')
    summer_from,summer_to=pd.Timestamp(args.summer_from),pd.Timestamp(args.summer_to)
    events=[];screen=[];all_trades=[];analyzed=0
    for code,grp in data.groupby('symbol'):
        d=enrich(grp.sort_values('date'))
        # 避免隨便把低流動性標的列入排名；歷史信號另依訊號日前20日成交額篩選
        d['avg_turnover20']=(d['close']*d['volume']).shift(1).rolling(20).mean()
        e, cur=probe_events(code,d,summer_from,summer_to)
        if cur is not None:
            cur['avg_turnover20']=float(d.iloc[-1]['avg_turnover20'])
            if (cur['in_summer'] and cur['avg_turnover20']>=args.min_turnover and
                cur['date']>=str(pd.Timestamp(args.asof)-pd.Timedelta(days=8))):
                screen.append(cur)
        for event in e:
            i=int(event['signal_index'])
            if (i>=20 and d.iloc[i]['avg_turnover20']>=args.min_turnover):
                events.append({k:v for k,v in event.items() if k!='signal_index'})
                for hold in HORIZONS:
                    tr=evaluate_trade(d,event,hold)
                    if tr is not None:all_trades.append(tr)
        analyzed+=1
    screen_df=pd.DataFrame(screen)
    events_df=pd.DataFrame(events)
    trades_df=pd.DataFrame(all_trades)
    if not screen_df.empty:
        screen_df['priority']=((screen_df['stage'].str.contains('C'))*2
          +screen_df['rsi'].between(45,65).astype(int)
          +(screen_df['vol_ratio']>=1.2).astype(int)
          +(screen_df['macd_hist']>0).astype(int))
        screen_df=screen_df.sort_values(['priority','avg_turnover20'],ascending=[False,False])
    screen_df.to_csv(out/'current_B_C_candidates.csv',index=False,encoding='utf-8-sig')
    events_df.to_csv(out/'historical_signal_events.csv',index=False,encoding='utf-8-sig')
    trades_df.to_csv(out/'all_historical_trades.csv',index=False,encoding='utf-8-sig')
    if trades_df.empty:
        print('[回測] 沒有成熟樣本。調整門檻/加長資料期間；不可虛構績效。')
        return
    summary=pd.DataFrame([dict(type=kind,hold=hold,**agg(sub).to_dict())
        for (kind,hold),sub in trades_df.groupby(['type','hold'],dropna=False)])
    summary.to_csv(out/'performance_B_vs_C.csv',index=False,encoding='utf-8-sig')
    feat=trades_df[trades_df['hold']==20].copy()
    feat['rsi_bin']=pd.cut(feat.rsi,[0,40,50,60,70,100],include_lowest=True)
    feat['vol_bin']=pd.cut(feat.vol_ratio,[0,1,1.5,2.5,100],include_lowest=True)
    feat['retrace_bin']=pd.cut(feat.b_retrace,[0,.382,.618,.786,1],include_lowest=True)
    groups=[]
    for col in ('rsi_bin','vol_bin','retrace_bin','hist_rising'):
        for (kind,b),sub in feat.groupby(['type',col],observed=True):
            if len(sub)<5: continue
            row=agg(sub).to_dict();row.update(type=kind,feature=col,bucket=str(b))
            row['credible_sample_n15']=len(sub)>=15
            groups.append(row)
    feature_df=pd.DataFrame(groups)
    if not feature_df.empty:
        feature_df=feature_df.sort_values(['type','mean_net_pct'],ascending=[True,False])
    feature_df.to_csv(out/'feature_performance_20d.csv',index=False,encoding='utf-8-sig')
    with (out/'METHODOLOGY.txt').open('w',encoding='utf-8') as f:
        f.write('ABC Wave Backtest - no lookahead indicators. Entry = NEXT trading day open.\n'
                'A peak/B bottom defined chronologically; signals after each day close.\n'
                'Min fall=30%, A rebound>=25%, B pullback=25%-96% of A distance; B RSI=35..68 and MACD histogram rising; C close>=A*1.01, volume>1.3 x prior20, MACD histogram>0.\n'
                'Return window 5/10/20 trading days, stop -8%, profit +25%. Stop priority if stop and profit both touched intraday.\n'
                f'Costs: buy {COST_BUY*100:.4f}%, sell {COST_SELL*100:.4f}%; fees, tax & slippage.\n'
                'Results by signal are event studies, NOT market-neutral portfolio returns.\n'
                'Caveats: third-party adjusted-price data, current constituents/survivorship bias, one-year history, no limit-up execution model, no cross-stock clustering adjustment. Do not optimize thresholds using the same period used to report significance.\n')
    print('\n[完成] 篩選候選數:',len(screen_df),' 歷史信號數:',len(events_df),' 成熟交易結果:',len(trades_df))
    print(summary.to_string(index=False,float_format=lambda x:f'{x:.2f}'))
    print(f'\n結果位址：{out.resolve()}')

if __name__=='__main__':
    main()
