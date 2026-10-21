#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""台股 B 上緣突破 -> C 段 -> 回升趨勢研究 v2.

本程式的當日信號不引用未來K線。所有績效是歷史事件回測，不是投資建議。
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from market_abc_core import get_data, enrich, COST_BUY, COST_SELL

HORIZONS = (5, 10, 20, 40, 60)
STAGES = {
    'NO_PATTERN':'未符合型態', 'A_FORM':'A反彈尚未形成B',
    'WATCH_B':'B盤整中', 'NEAR_B':'接近B突破壓力',
    'B_UNCONFIRMED':'站上B壓力但量價未確認',
    'B_BREAK':'B剛突破 / C起步', 'B_RETEST':'突破後回測B上緣',
    'C_RUN':'C波續強', 'C_MATURE':'C波較成熟／不宜視為新買點',
    'A_BREAK':'剛突破A波高點', 'A_RUN':'站上A高／延伸趨勢',
    'A_PULLBACK':'A高突破後回落', 'FAILED':'型態失效／跌破B低點',
}
BUY_STAGES = {'B_BREAK'}
WATCH_STAGES = {'NEAR_B','B_UNCONFIRMED','B_RETEST'}
TREND_STAGES = {'C_RUN','C_MATURE','A_BREAK','A_RUN','A_PULLBACK'}


def num(v):
    try:
        r=float(v)
        return r if np.isfinite(r) else np.nan
    except (TypeError,ValueError):
        return np.nan


def event_payload(code, d, i, kind, peak, bottom, a_high, b_low,
                  bottom_i, a_i, b_start, ref, avg_turnover):
    bar=d.iloc[i];prev=d.iloc[i-1]
    retrace=(a_high-b_low)/(a_high-bottom) if a_high>bottom else np.nan
    b_mean=num(d.iloc[max(b_start,i-9):i+1].volume.mean())
    # B期均量 vs A反彈期均量：兩個樣本窗口完全截至今日。
    a_mean=num(d.iloc[bottom_i:a_i+1].volume.mean())
    close_loc=(bar.close-bar.low)/(bar.high-bar.low) if bar.high>bar.low else .5
    rsi=num(bar.rsi);v=num(bar.vol_ratio);macd=num(bar.macd_hist)
    slope=num(bar.ma20/d.iloc[i-5].ma20-1) if i>=5 else np.nan
    gap=num(bar.open/prev.close-1) if prev.close else np.nan
    # 注意：優質條件是研究假設，不是依據本次資料最佳化後的參數。
    quality=bool(kind=='B_上緣突破' and .382<=retrace<=.75 and
       50<=rsi<=70 and v>=1.5 and close_loc>=.60 and
       slope>=0 and bar.close>bar.ma20 and num(bar.close/b_low-1)<=.22)
    return dict(symbol=code,signal_type=kind,
        signal_date=bar.date.date().isoformat(),signal_index=i,
        crash_peak=num(peak),bottom=num(bottom),a_high=num(a_high),b_low=num(b_low),
        bottom_date=d.iloc[bottom_i].date.date().isoformat(),
        a_high_date=d.iloc[a_i].date.date().isoformat(),
        crash_pct=100*(bottom/peak-1),a_rebound_pct=100*(a_high/bottom-1),
        b_retrace=retrace,b_days=i-b_start,days_from_bottom=i-bottom_i,
        breakout_ref=num(ref),breakout_pct=100*(bar.close/ref-1) if ref else np.nan,
        rsi=rsi,vol_ratio=v,macd_hist=macd,macd_rising=bool(bar.hist_rising),
        ma20_slope5_pct=100*slope,ma20_at_signal=num(bar.ma20),
        ma60_at_signal=num(bar.close/d.iloc[max(0,i-59):i+1].close.mean()-1) if i>=59 else np.nan,
        close_location=num(close_loc),gap_pct=100*gap,
        b_volume_vs_a=b_mean/a_mean if a_mean and np.isfinite(a_mean) else np.nan,
        risk_to_B_pct=100*(bar.close/b_low-1) if b_low else np.nan,
        distance_to_A_pct=100*(a_high/bar.close-1),
        prior20_turnover=num(avg_turnover),quality_hypothesis=quality)


def inspect_symbol(code, raw, summer_from, summer_to, turnover_limit):
    d=enrich(raw.sort_values('date')).reset_index(drop=True)
    d['avg_turnover20']=(d.close*d.volume).shift(1).rolling(20).mean()
    d['ma60']=d.close.rolling(60).mean()
    n=len(d)
    if n<105:return None,[],d
    state='SEARCH';events=[]
    ph=p0=a_high=b_low=b_ref=np.nan
    peak_i=bottom_i=a_i=b_start=None
    signal_i=a_break_i=None
    a_break_ref=np.nan
    dates=d.date.to_numpy();hi=d.high.to_numpy();lo=d.low.to_numpy();
    close=d.close.to_numpy();vol=d.volume.to_numpy();rsi=d.rsi.to_numpy()
    vr=d.vol_ratio.to_numpy();hist=d.macd_hist.to_numpy();ma20=d.ma20.to_numpy()
    turnover=d.avg_turnover20.to_numpy()

    def reset():
        nonlocal state,ph,p0,a_high,b_low,b_ref,peak_i,bottom_i,a_i,b_start,signal_i,a_break_i,a_break_ref
        state='SEARCH';ph=p0=a_high=b_low=b_ref=a_break_ref=np.nan
        peak_i=bottom_i=a_i=b_start=signal_i=a_break_i=None

    for i in range(65,n):
        if vol[i]<=0:continue
        if state=='SEARCH':
            left=max(0,i-65)
            pp=int(np.argmax(hi[left:i]))+left
            if hi[pp]>0 and close[i]/hi[pp]<=.70 and i-pp<=55:
                ph=hi[pp];peak_i=pp;p0=lo[i];bottom_i=i;state='BOTTOM'
        elif state=='BOTTOM':
            if lo[i]<p0:p0=lo[i];bottom_i=i
            if i-bottom_i>45 or i-peak_i>120:
                reset();continue
            if i-bottom_i>=2 and close[i]>=p0*1.18:
                a_high=hi[i];a_i=i;state='A'
        elif state=='A':
            if hi[i]>a_high:a_high=hi[i];a_i=i
            if a_high>=p0*1.25 and i-a_i>=3 and lo[i]<=a_high-.25*(a_high-p0):
                b_low=lo[i];b_start=i;state='B'
            elif i-bottom_i>120:reset()
        elif state=='B':
            if lo[i]<b_low:b_low=lo[i]
            depth=(a_high-b_low)/(a_high-p0)
            if b_low<=p0*1.005 or depth>=.94 or i-bottom_i>135:
                reset();continue
            start=max(a_i+1,i-12)
            ref_hi=np.max(hi[start:i]) if i-start>=3 else np.nan
            b_ref=min(ref_hi,a_high) if np.isfinite(ref_hi) else np.nan
            if i-b_start<5 or not np.isfinite(b_ref):continue
            broke=close[i]>b_ref*1.005 and close[i-1]<=b_ref*1.005
            momentum=(46<=rsi[i]<=76 and vr[i]>=1.2 and
                np.isfinite(hist[i]) and hist[i]>hist[i-1] and close[i]>ma20[i])
            if broke and momentum and turnover[i]>=turnover_limit:
                kind='B_上緣突破' if close[i]<a_high*1.01 else 'A_高點直接突破'
                events.append(event_payload(code,d,i,kind,ph,p0,a_high,b_low,
                                            bottom_i,a_i,b_start,b_ref,turnover[i]))
                signal_i=i
                if close[i]>=a_high*1.01:
                    a_break_i=i;a_break_ref=a_high;state='ABOVE_A'
                else:state='C'
            elif close[i]>=a_high*1.01 and close[i-1]<a_high*1.01:
                # 純價格突破可列當下狀態，但不當成滿足B波量價買進訊號。
                a_break_i=i;a_break_ref=a_high;state='ABOVE_A'
        elif state in ('C','ABOVE_A'):
            if close[i]<b_low*.995 or lo[i]<=p0:
                state='FAILED';continue
            if i-bottom_i>180:
                reset();continue
            if state=='C' and close[i]>=a_high*1.01 and close[i-1]<a_high*1.01:
                a_break_i=i;a_break_ref=a_high;state='ABOVE_A'
        elif state=='FAILED':
            if i-bottom_i>185 or lo[i]<p0:reset()

    i=n-1;last=d.iloc[-1]
    recent=bottom_i is not None and summer_from<=dates[bottom_i]<=summer_to
    depth=((a_high-b_low)/(a_high-p0) if np.isfinite(a_high) and np.isfinite(b_low) and a_high>p0 else np.nan)
    near_gap=np.nan
    if state=='B' and a_i is not None:
        start=max(a_i+1,i-12)
        ref_hi=np.max(hi[start:i]) if i-start>=3 else np.nan
        b_ref=min(ref_hi,a_high) if np.isfinite(ref_hi) else np.nan
        near_gap=100*(b_ref/close[i]-1) if np.isfinite(b_ref) else np.nan
    age_sig=(i-signal_i) if signal_i is not None else np.nan
    age_A=(i-a_break_i) if a_break_i is not None else np.nan
    stage='NO_PATTERN'
    if recent and state!='SEARCH':
        if state=='A':stage='A_FORM'
        elif state=='B':
            stage=('NEAR_B' if np.isfinite(near_gap) and 0<=near_gap<=4
                else 'B_UNCONFIRMED' if np.isfinite(near_gap) and near_gap<0
                else 'WATCH_B')
        elif state=='C':
            if signal_i is not None and close[i]<b_ref*.99:stage='B_RETEST'
            elif signal_i is not None and i-signal_i<=5:stage='B_BREAK'
            elif signal_i is not None and i-signal_i<=20:stage='C_RUN'
            else:stage='C_MATURE'
        elif state=='ABOVE_A':
            if close[i]<a_break_ref*.99:stage='A_PULLBACK'
            elif a_break_i is not None and i-a_break_i<=5:stage='A_BREAK'
            else:stage='A_RUN'
        elif state=='FAILED':stage='FAILED'
    # 信號時間比已延伸多個月的A突破更重要；分群不混在一起。
    base={'B_BREAK':85,'NEAR_B':78,'B_UNCONFIRMED':53,'B_RETEST':44,
          'C_RUN':65,'C_MATURE':38,'A_BREAK':61,'A_RUN':37,'A_PULLBACK':25,
          'WATCH_B':30,'A_FORM':15,'FAILED':0,'NO_PATTERN':0}.get(stage,0)
    score=base
    if np.isfinite(depth) and .382<=depth<=.618:score+=7
    if 50<=num(last.rsi)<=70:score+=5
    if 1.2<=num(last.vol_ratio)<=4:score+=5
    if np.isfinite(last.macd_hist) and last.macd_hist>0:score+=5
    if i>=5 and ma20[i]>ma20[i-5]:score+=5
    if stage=='NEAR_B' and 0<=near_gap<=2:score+=4
    prior20=num(last.avg_turnover20)
    recent_trigger=signal_i is not None and (i-signal_i)<=5 and close[i]>=b_ref*.99
    # 收盤後執行的策略只能買隔一交易日，因此剛突破5天不是『今日新買點』。
    new_signal_today=signal_i is not None and signal_i==i
    actionable=bool(recent and prior20>=turnover_limit and stage=='B_BREAK'
                    and recent_trigger and new_signal_today)
    trend_observe=bool(recent and prior20>=turnover_limit and stage in TREND_STAGES)
    row=dict(symbol=code,date=last.date.date().isoformat(),close=num(last.close),
        stage=stage,stage_zh=STAGES[stage],priority_score=score,
        crash_peak=num(ph),bottom=num(p0),a_high=num(a_high),b_low=num(b_low),
        b_resistance=num(b_ref),distance_to_b_break_pct=near_gap,b_retrace=num(depth),
        crash_pct=100*(p0/ph-1) if np.isfinite(ph) else np.nan,
        vol_ratio=num(last.vol_ratio),rsi=num(last.rsi),macd_hist=num(last.macd_hist),
        macd_hist_rising=bool(last.hist_rising),
        ma20_slope5_pct=100*(ma20[i]/ma20[i-5]-1) if i>=5 and ma20[i-5] else np.nan,
        ma20=num(last.ma20),ma60=num(last.ma60),avg_turnover20=prior20,
        b_break_date=d.iloc[signal_i].date.date().isoformat() if signal_i is not None else '',
        a_break_date=d.iloc[a_break_i].date.date().isoformat() if a_break_i is not None else '',
        bottom_date=d.iloc[bottom_i].date.date().isoformat() if bottom_i is not None else '',
        days_from_b_break=num(age_sig),days_from_a_break=num(age_A),
        new_signal_today=bool(new_signal_today) if state=='C' else False,
        above_b_res=bool(np.isfinite(b_ref) and last.close>=b_ref*.99),
        above_ma60=bool(np.isfinite(last.ma60) and last.close>last.ma60),
        distance_to_A_pct=100*(a_high/close[i]-1) if np.isfinite(a_high) else np.nan,
        in_target_summer=bool(recent),qualified=bool(actionable or trend_observe or
                  (stage in WATCH_STAGES and recent and prior20>=turnover_limit)),
        actionable_buy=actionable,trend_observe=trend_observe,
        tradable=bool(prior20>=turnover_limit))
    return row,events,d


def fivewave_proxy(d, start, end, bottom):
    """未來形態的保守代理，不作為當天的預測標記。"""
    s=d.iloc[start:end+1]
    if len(s)<17:return False
    high=s.high.to_numpy();low=s.low.to_numpy(); piv=[]
    for k in range(2,len(s)-2):
        if high[k]>max(high[k-2:k].max(),high[k+1:k+3].max()):piv.append((k,'H',high[k]))
        if low[k]<min(low[k-2:k].min(),low[k+1:k+3].min()):piv.append((k,'L',low[k]))
    piv.sort(key=lambda x:x[0]);clean=[]
    for p in piv:
        if not clean:clean.append(p)
        elif p[1]==clean[-1][1]:
            if (p[1]=='H' and p[2]>clean[-1][2]) or (p[1]=='L' and p[2]<clean[-1][2]):clean[-1]=p
        elif abs(p[2]/clean[-1][2]-1)>=.025:clean.append(p)
    for j in range(len(clean)-4):
        p=clean[j:j+5]
        if [x[1] for x in p]!=['H','L','H','L','H']:continue
        h1,l2,h3,l4,h5=[x[2] for x in p]
        w1=h1-bottom;w3=h3-l2;w5=h5-l4
        if h1>bottom*1.05 and bottom<l2<h1*.98 and h3>h1*1.02 and \
           h1<l4<h3*.98 and h5>h3*1.01 and w3>=min(w1,w5):return True
    return False


def outcome_stats(d,e,start,end,actual_exit):
    future=d.iloc[start:end+1]
    if future.empty:return dict(future_A_break=False,future_trend_recovery=False,
        recovery_before_exit=False,fivewave_proxy=False)
    strong=((future.close>future.ma60)&(future.ma20>future.ma20.shift(5))&
            (future.close>e['a_high']*1.03)).fillna(False)
    # 若整段曾跌破 B 低點，不能叫作順暢的回升行情。
    recovery=(strong.rolling(5,min_periods=5).sum()>=5)
    future_ok=bool(recovery.any() and future.low.min()>e['b_low']*.995)
    held=future.iloc[:actual_exit-start+1]
    held_ok=bool((recovery.iloc[:len(held)]).any() and held.low.min()>e['b_low']*.995)
    return dict(future_A_break=bool((future.close>e['a_high']*1.01).any()),
         future_trend_recovery=future_ok,recovery_before_exit=held_ok,
         fivewave_proxy=fivewave_proxy(d,start,end,e['b_low']))


def trade(d,e,horizon,mode='fixed'):
    """次日開盤成交；持有滿 horizon 才能列入成熟樣本。

    fixed: B低點或最多8%停損; runner: 先同樣停損，浮盈10%且隔日才上移至保本及最高收盤回撤10%。
    """
    i=int(e['signal_index']);start=i+1;end=i+horizon
    if end>=len(d) or num(d.iloc[start].volume)<=0:return None
    entry=num(d.iloc[start].open)
    if not np.isfinite(entry) or entry<=0:return None
    initial_stop=max(entry*.92,e['b_low']*.995)
    if initial_stop>=entry:return None
    stop=initial_stop;max_prior_close=entry
    exit_i=end;exit_price=num(d.iloc[end].close);reason='期滿'
    for j in range(start,end+1):
        day=d.iloc[j]
        if day.volume<=0:continue
        # runner 使用前一交易日收盤更新移動停損，本日高低點不參與停損設定。
        if mode=='runner' and j>start and max_prior_close>=entry*1.10:
            stop=max(stop,entry*1.005,max_prior_close*.90)
        if day.open<=stop:
            exit_i=j;exit_price=num(day.open);reason='跳空停損';break
        if day.low<=stop:
            exit_i=j;exit_price=num(stop);reason='移動停損' if stop>initial_stop else '初始停損';break
        max_prior_close=max(max_prior_close,num(day.close))
    if not np.isfinite(exit_price) or d.iloc[exit_i].volume<=0:return None
    held=d.iloc[start:exit_i+1]
    net=exit_price*(1-COST_SELL)/(entry*(1+COST_BUY))-1
    stats=outcome_stats(d,e,start,end,exit_i)
    v={k:x for k,x in e.items() if k!='signal_index'}
    v.update(mode=mode,horizon=int(horizon),entry_date=d.iloc[start].date.date().isoformat(),
        entry=entry,exit_date=d.iloc[exit_i].date.date().isoformat(),exit=exit_price,
        bars_held=exit_i-start+1,net_pct=100*net,
        mae_pct=100*(held.low.min()/entry-1),mfe_pct=100*(held.high.max()/entry-1),
        stop_distance_pct=100*(1-initial_stop/entry),exit_reason=reason,**stats)
    return v


def summarize(s):
    r=s.net_pct;gain=r[r>0];loss=r[r<0];by_date=s.groupby('signal_date').net_pct.mean()
    return dict(n=len(s),signal_dates=s.signal_date.nunique(),
        win_rate_pct=round(100*(r>0).mean(),2),avg_net_pct=round(r.mean(),3),
        median_net_pct=round(r.median(),3),date_equal_avg_pct=round(by_date.mean(),3),
        profit_factor=round(gain.sum()/(-loss.sum()),3) if len(loss) else np.nan,
        stop_rate_pct=round(100*s.exit_reason.str.contains('停損').mean(),2),
        future_trend_pct=round(100*s.future_trend_recovery.mean(),2),
        recovery_held_pct=round(100*s.recovery_before_exit.mean(),2),
        fivewave_proxy_pct=round(100*s.fivewave_proxy.mean(),2),
        avg_mae_pct=round(s.mae_pct.mean(),2),adequate_n30=bool(len(s)>=30),
        adequate_dates10=bool(s.signal_date.nunique()>=10))


def write_csv(df,path,empty_columns=None):
    if df.empty and len(df.columns)==0 and empty_columns:
        df=pd.DataFrame(columns=empty_columns)
    df.to_csv(path,index=False,encoding='utf-8-sig')


def make_reports(args):
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    data=get_data(args)
    if data.empty:raise RuntimeError('輸入日線為空')
    requested=data.date.max() if args.asof.lower()=='auto' else pd.Timestamp(args.asof)
    data=data[data.date<=requested].copy()
    if data.empty:raise RuntimeError(f'找不到截至 {requested.date()} 前的日線')
    latest=data.date.max()
    if args.asof.lower()=='auto' and (pd.Timestamp.now().normalize()-latest).days>7:
        print('[警告] 行情比執行日落後超過7天：請先確認資料來源更新狀態。',flush=True)
    last=data.groupby('symbol').date.max()
    valid=last[last>=latest-pd.Timedelta(days=7)].index
    data=data[data.symbol.isin(valid)]
    n_universe=len(valid)
    print(f'[行情] 實際最新日 {latest.date()}，有效股票 {n_universe}，日線 {len(data):,} 筆',flush=True)
    if n_universe<args.min_universe:raise RuntimeError(f'實際股票{n_universe}低於最低{args.min_universe}，拒絕假200檔')
    summer_from=np.datetime64(pd.Timestamp(args.summer_from).date())
    summer_to=np.datetime64(pd.Timestamp(args.summer_to).date())
    rows=[];events=[];trades=[]
    for idx,(code,g) in enumerate(data.groupby('symbol'),1):
        row,ev,d=inspect_symbol(code,g,summer_from,summer_to,args.min_turnover)
        if row is not None:rows.append(row)
        for e in ev:
            if not summer_from<=np.datetime64(pd.Timestamp(e['bottom_date']).date())<=summer_to:continue
            events.append({k:v for k,v in e.items() if k!='signal_index'})
            for horizon in HORIZONS:
                a=trade(d,e,horizon,'fixed')
                if a is not None:trades.append(a)
                if horizon in (20,40):
                    b=trade(d,e,horizon,'runner')
                    if b is not None:trades.append(b)
        if idx%300==0:print(f'[進度] {idx}/{n_universe}檔',flush=True)
    all_df=pd.DataFrame(rows)
    if all_df.empty:raise RuntimeError('所有股票都缺少足夠日線，沒有成功計算')
    all_df=all_df.sort_values(['priority_score','avg_turnover20'],ascending=False).reset_index(drop=True)
    all_df['rank']=np.arange(1,len(all_df)+1)
    write_csv(all_df,out/'all_scanned_stocks.csv')
    write_csv(all_df.head(max(200,args.watch_size)),out/'watchlist_top_200_plus.csv')
    legit=all_df[all_df.tradable]
    b=legit[legit.stage.isin(['NEAR_B','B_UNCONFIRMED','B_BREAK','B_RETEST'])]
    c=legit[legit.stage.isin(TREND_STAGES)]
    write_csv(b,out/'B_near_or_breakout.csv')
    write_csv(c,out/'C_progress_and_A_break.csv')
    write_csv(b[b.actionable_buy],out/'B_breakout_buy_signals.csv')
    write_csv(b[b.stage.isin(['NEAR_B','B_UNCONFIRMED'])],out/'B_prebreak_watchlist.csv')
    write_csv(c[c.stage.isin(['C_RUN','A_BREAK','A_RUN'])],out/'C_trend_transition.csv')
    signals=pd.DataFrame(events)
    write_csv(signals,out/'historical_breakout_events.csv',
              ['symbol','signal_type','signal_date','b_retrace','rsi','vol_ratio'])
    t=pd.DataFrame(trades)
    write_csv(t,out/'matured_trades.csv',
              ['symbol','signal_date','signal_type','mode','horizon','net_pct'])
    summary=[];groups=[];paired=[]
    if not t.empty:
        for (mode,kind,h),s in t.groupby(['mode','signal_type','horizon']):
            summary.append(dict(mode=mode,signal_type=kind,horizon=int(h),**summarize(s)))
        # 統一以5/10/20日做研究，不硬要使用樣本極少的40日。
        for h in [5,10,20]:
            q=t[(t['mode']=='fixed')&(t.horizon==h)&(t.signal_type=='B_上緣突破')].copy()
            if q.empty:continue
            q['rsi_band']=pd.cut(q.rsi,[0,45,55,65,75,100],include_lowest=True)
            q['volume_band']=pd.cut(q.vol_ratio,[0,1.2,1.5,2,3,100],include_lowest=True)
            q['B_retrace_band']=pd.cut(q.b_retrace,[0,.382,.5,.618,.75,.85,1],include_lowest=True)
            q['close_location_band']=pd.cut(q.close_location,[-.01,.5,.7,1.01],include_lowest=True)
            q['gap_band']=pd.cut(q.gap_pct,[-100,-1,1,3,100],include_lowest=True)
            q['B_volume_vs_A_band']=pd.cut(q.b_volume_vs_a,[0,.5,.8,1,1.5,100],include_lowest=True)
            q['quality_hypothesis']=q.quality_hypothesis.astype(str)
            for metric in ['rsi_band','volume_band','B_retrace_band','close_location_band',
                           'gap_band','B_volume_vs_A_band','quality_hypothesis']:
                for band,g in q.groupby(metric,observed=True):
                    if len(g)<5:continue
                    groups.append(dict(horizon=h,metric=metric,band=str(band),**summarize(g)))
        # 只比較同一個事件同時已有5/10/20日成熟報酬，避免樣本不對齊。
        for kind,s in t[t['mode']=='fixed'].groupby('signal_type'):
            piv=s.pivot_table(index=['symbol','signal_date'],columns='horizon',values='net_pct',aggfunc='first')
            for a,h in [(5,10),(10,20),(5,20)]:
                common=piv[[a,h]].dropna() if a in piv.columns and h in piv.columns else pd.DataFrame()
                if common.empty:continue
                paired.append(dict(signal_type=kind,earlier_horizon=a,later_horizon=h,
                    paired_n=len(common),avg_earlier_pct=round(common[a].mean(),3),
                    avg_later_pct=round(common[h].mean(),3),
                    later_minus_earlier_pct=round((common[h]-common[a]).mean(),3)))
    write_csv(pd.DataFrame(summary),out/'strategy_performance.csv',
              ['mode','signal_type','horizon','n','win_rate_pct','avg_net_pct'])
    write_csv(pd.DataFrame(groups),out/'feature_comparison_5_10_20d.csv',
              ['horizon','metric','band','n','win_rate_pct','avg_net_pct'])
    write_csv(pd.DataFrame(paired),out/'paired_horizon_comparison.csv',
              ['signal_type','earlier_horizon','later_horizon','paired_n','avg_earlier_pct','avg_later_pct'])
    # v2 不沿用舊的 feature_40day_comparison.csv 空檔名，避免誤判樣本成熟度。
    bnum=int(len(b));cnum=int(len(c));bb=int((b.stage=='B_BREAK').sum())
    buy_today=int(b.actionable_buy.sum())
    best=t[(t['mode']=='fixed')&(t.horizon==10)&(t.signal_type=='B_上緣突破')] if not t.empty else pd.DataFrame()
    lines=['# 台股 B→C→回升 v2 掃描與回測摘要','',
        f'- 設定截止日：{requested.date()}；實際行情最新日：**{latest.date()}**',
        f'- 有效掃描股票：**{n_universe}**；產生排名：**{min(len(all_df),max(200,args.watch_size))}**',
        f'- B相關監看：**{bnum}**；近5日剛突破：**{bb}**；**最新交易日真正新訊號：{buy_today}**；C/趨勢相關：**{cnum}**',
        f'- 歷史入場事件：**{len(events)}**；成熟回測列數（不同期間/模式合計）：**{len(t)}**','',
        '## 本次優化','',
        '- B突破跌回壓力線改標 B_RETEST；站上壓力但不符合量價則 B_UNCONFIRMED。',
        '- A高已突破很久者為 A_RUN，不再混入「剛突破」訊號。',
        '- B_BREAK涵蓋5天內突破，只有 new_signal_today=True 是今日收盤觸發、可研究隔日開盤進場的新訊號。',
        '- NEAR_B 尚未突破，也不是買進訊號。',
        '- 回測 5/10/20/40/60日；5/10/20日分箱；40/60日依成熟樣本數判斷。',
        '- 增加 quality_hypothesis、收盤位階、跳空幅度、B期量縮、回撤程度等事前特徵。',
        '- 增加 runner：上漲10%後才移動停損；每日只用前一交易日的收盤更新。',
        '- 區分 future_trend_recovery（後續行情）和 recovery_before_exit（停損離場前達成）。',
        '- 配對比較各持有期，新增交易日等權平均，提醒訊號同日群聚。','',
        '## 研究限制','',
        '- 一年日線無法證實40/60日最佳參數；feature_comparison 是探索性描述而非樣本外預測。',
        '- 資料來源為第三方，可能具有除權息調整、倖存者偏誤、缺少下市股票等問題。',
        '- 一檔股票不同訊號與跨股票同日交易可能高度相關，不能把事件n當完全獨立樣本。',
        '- runner與fixed都使用次日開盤入場、費用/滑價與保守日K停損假設，漲跌停仍可能無法成交。',
        '- 五波代理是事後型態，樣本過短可能長期為0，不應當作當日交易信號。','']
    if not best.empty:
        x=summarize(best)
        lines += [f'B上緣突破（10日fixed）：{x["n"]}筆，勝率{x["win_rate_pct"]}%，平均淨報酬{x["avg_net_pct"]}%（非樣本外績效）。','']
    (out/'RESULT_SUMMARY.md').write_text('\n'.join(lines),encoding='utf-8')
    topcols=['symbol','stage_zh','close','b_resistance','distance_to_b_break_pct','rsi','vol_ratio']
    def md_table(df,n=12):
        if df.empty:return '目前無符合股票。'
        s=df[topcols].head(n).copy().fillna('')
        return s.to_markdown(index=False) if hasattr(s,'to_markdown') else s.to_string(index=False)
    # 不依賴額外 tabulate 套件才能產生 GitHub Step Summary。
    def simple_md(df,n=10):
        if df.empty:return '目前無符合股票。'
        cols=['symbol','stage_zh','close','rsi','vol_ratio']
        q=df[cols].head(n)
        h='|代碼|階段|收盤|RSI|量比|\n|---|---|---:|---:|---:|'
        for _,p in q.iterrows():h+=f'\n|{p.symbol}|{p.stage_zh}|{p.close:.2f}|{p.rsi:.1f}|{p.vol_ratio:.2f}|'
        return h
    dash='\n'.join(['# 台股 ABC 即時監控',f'行情最新日：{latest.date()} / 已掃描 {n_universe} 檔',
                    f'近5日B突破 {bb} 檔；**今日新訊號 {buy_today} 檔**；B候選 {bnum} 檔；C與趨勢 {cnum} 檔','',
                    '## 今日收盤新突破訊號（研究隔日進場）',simple_md(b[b.actionable_buy]),
                    '','## 近5日突破的觀察名單（非今日新訊號）',simple_md(b[b.stage=='B_BREAK']),
                    '','## 接近B突破',simple_md(b[b.stage=='NEAR_B']),
                    '','## C進展（不等於目前新買點）',simple_md(c[c.stage.isin(['C_RUN','A_BREAK'])]),
                    '','本報表為型態研究。請在Artifacts下載完整CSV，確認訊號當日與可交易性。'])
    (out/'GITHUB_SUMMARY.md').write_text(dash,encoding='utf-8')
    print(f'[完成] 掃描 {n_universe} 檔；B監控 {bnum}（5日突破{bb}、今日新訊號{buy_today}）；C趨勢 {cnum}；事件 {len(events)}',flush=True)
    if summary:print(pd.DataFrame(summary)[['mode','signal_type','horizon','n','win_rate_pct','avg_net_pct','signal_dates']].to_string(index=False),flush=True)
    return dict(scanned=n_universe,b=bnum,c=cnum,event=len(events))


def main():
    p=argparse.ArgumentParser(description='台股 B上緣突破→C→回升 v2 全市場量化選股')
    p.add_argument('--data',default=None,help='既有台股日線OHLCV csv / csv.gz')
    p.add_argument('--asof',default='auto',help='最新日=auto，或 YYYY-MM-DD')
    p.add_argument('--out',default='results');p.add_argument('--workers',type=int,default=10)
    p.add_argument('--max-symbols',type=int,default=0)
    p.add_argument('--min-universe',type=int,default=200)
    p.add_argument('--watch-size',type=int,default=200)
    p.add_argument('--min-turnover',type=float,default=1.e7)
    p.add_argument('--summer-from',default='2026-06-01')
    p.add_argument('--summer-to',default='2026-08-31')
    a=p.parse_args();make_reports(a)

if __name__=='__main__':main()
