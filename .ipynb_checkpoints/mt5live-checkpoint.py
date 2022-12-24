import time
import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"    
#import tensorflow as tf
import matplotlib.pyplot as plt
import MetaTrader5 as mt5
import pandas as pd    
import numpy as np
from datetime import datetime, timedelta
#import math
#from IPython.display import clear_output
#import ta
from collections import deque
#import math
#from tqdm import tqdm
#import pytz
import cv2

hour_offset = 8 # time - hour_offset = ny local time
lookback = 200
mt5.initialize()
authorized=mt5.login(25031341, password = "!geH2e4Pi!Ka", server = "TickmillUK-Demo")
mt5.account_info()
symbols = ["EURUSD", "GBPUSD"]
print(authorized)
dev = 1



ea_magic_number = 9986989 # if you want to give every bot a unique identifier

def get_info(symbol):
    '''https://www.mql5.com/en/docs/integration/python_metatrader5/mt5symbolinfo_py
    '''
    # get symbol properties
    info=mt5.symbol_info(symbol)
    return info

def open_trade(action, symbol, lot):
    '''https://www.mql5.com/en/docs/integration/python_metatrader5/mt5ordersend_py
    '''
    # prepare the buy request structure
    symbol_info = get_info(symbol)

    if action == 'buy':
        trade_type = mt5.ORDER_TYPE_BUY
        price = mt5.symbol_info_tick(symbol).ask
    elif action =='sell':
        trade_type = mt5.ORDER_TYPE_SELL
        price = mt5.symbol_info_tick(symbol).bid
    point = mt5.symbol_info(symbol).point

    buy_request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": lot,
        "type": trade_type,
        "price": price,
        "deviation": dev,
        "magic": ea_magic_number,
        "comment": "tx_bot",
        "type_time": mt5.ORDER_TIME_GTC, # good till cancelled
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    # send a trading request
    result = mt5.order_send(buy_request)        
    return result, buy_request 

def close_position(pos):
    close_trade(pos.symbol, "sell" if pos.type == 0 else "buy" , pos.ticket, pos.volume)
    
def close_trade(symbol, action, ticket, vol):
    '''https://www.mql5.com/en/docs/integration/python_metatrader5/mt5ordersend_py
    '''
    # create a close request
    if action == 'buy':
        trade_type = mt5.ORDER_TYPE_BUY
        price = mt5.symbol_info_tick(symbol).ask
    elif action =='sell':
        trade_type = mt5.ORDER_TYPE_SELL
        price = mt5.symbol_info_tick(symbol).bid
    position_id=ticket
    lot = vol

    close_request={
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": lot,
        "type": trade_type,
        "position": position_id,
        "price": price,
        "deviation": dev,
        "magic": ea_magic_number,
        "comment": "tx_bot",
        "type_time": mt5.ORDER_TIME_GTC, # good till cancelled
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    # send a close request
    result=mt5.order_send(close_request)

def close_positions(symbol):
    positions = mt5.positions_get()
    for i in positions:
        if i.symbol == symbol and i.comment == "tx_bot":
            close_position(i)
       

def get_prices(symbol, tf):
    t = int(time.time()) + 60*60*24

    prices = mt5.copy_rates_from(symbol, tf, t, lookback)

    t = [x[0] for x in prices]
    o = [x[1] for x in prices]
    h = [x[2] for x in prices]
    l = [x[3] for x in prices]
    c = [x[4] for x in prices]
    
    true_time = []
    for i in t:
        dt_object = datetime.fromtimestamp(int(i)) - timedelta (hours=hour_offset)
        true_time.append(dt_object)
    
    data = {"Time": true_time, "Open":o, "High":h, "Low":l, "Close":c, "timestamp": t}
    df = pd.DataFrame(data)
    print(df)
    return df


def cv_plot_candles(candles, pd_arrays, d_range, p_d_range):
    w = 1300
    h = 600
    canvas = np.zeros((h,w,3), np.uint8) 
    l = len(candles)
    single_candle_w = w / l * 0.95
    max_h = 0
    max_l = 1000000
    for i in candles:
        if i.h > max_h:
            max_h = i.h
        if i.l < max_l:
            max_l = i.l
    hlrange = max_h - max_l
    def scale_p(p):
        return (p - max_l) / hlrange * h
    
    for i in range(len(candles)):
        color = (0,200,0) if candles[i].c > candles[i].o else (0,0,200)
        cv2.rectangle(canvas, (int(i*single_candle_w),int(scale_p(candles[i].o))), (int((i+1)*single_candle_w),int(scale_p(candles[i].c))), color, -1)
        cv2.line(canvas, (int((i+0.5)*single_candle_w),int(scale_p(candles[i].h))), (int((i+0.5)*single_candle_w),int(scale_p(candles[i].l))), color)
        
        sl = candles[i].swing_l_h[0]
        sh = candles[i].swing_l_h[1]
        
        if sh:
            cv2.circle(canvas, (int((i+0.5)*single_candle_w),int(scale_p(candles[i].h))), int(single_candle_w/2), (15,100,50), 3) 
        if sl:
            cv2.circle(canvas, (int((i+0.5)*single_candle_w),int(scale_p(candles[i].l))), int(single_candle_w/2), (15,50,100), 3) 
        
        bs_liquidity_taken = candles[i].buyside_taken
        ss_liquidity_taken = candles[i].sellside_taken
        
        if ss_liquidity_taken:
            cv2.rectangle(canvas, (int((i)*single_candle_w),int(scale_p(candles[i].o))), (int((i+1)*single_candle_w),int(scale_p(candles[i].c))), (255,255,0), 1)
        if bs_liquidity_taken:
            cv2.rectangle(canvas, (int((i)*single_candle_w),int(scale_p(candles[i].o))), (int((i+1)*single_candle_w),int(scale_p(candles[i].c))), (255,0,255), 1)
            
        if candles[i].xclosedcandles_broken[0] != 0:
            color = (50,255,0) if candles[i].xclosedcandles_broken[0] == 1 else (50,0,255)
            cv2.rectangle(canvas, (int((i)*single_candle_w),int(scale_p(candles[i].xclosedcandles_broken[1]))), (int((i+1)*single_candle_w),int(scale_p(candles[i].xclosedcandles_broken[2]))), color, 1)
              
    pd_array_draw_width = 40
    for i in pd_arrays:
        if type(i) == liquidity:
            color = (0,200,0) if i.l_type == -1 else (0,0,200)
            cv2.line(canvas, (w-pd_array_draw_width-20, int(scale_p(i.price))), (int(w), int(scale_p(i.price))), color, 1)
        
        if type(i) == fair_value_gap:
            color = (200,0,200) if i.bull_bear == -1 else (200,200,0)
            if i.retraded == True:
                color = (100,color[1],color[2])
            if i.closed == True:
                color = (50,color[1],color[2])
            cv2.rectangle(canvas, (int(w-pd_array_draw_width),int(scale_p(i.low))), (int(w),int(scale_p(i.high))), color, -1)
            
        if type(i) == xclosedcandles_broken:
            color = (200,0,200) if i.bull_bear == -1 else (200,200,0)
            if i.retraded == True:
                color = (100,color[1],color[2])
            if i.closed == True:
                color = (50,color[1],color[2])
            cv2.rectangle(canvas, (int(w-pd_array_draw_width),int(scale_p(i.low))), (int(w),int(scale_p(i.high))), color, 2)
            
            
            
        
    cv2.line(canvas, (int(d_range[0][1] * single_candle_w), int(scale_p(d_range[0][0]))), (int(d_range[1][1] * single_candle_w), int(scale_p(d_range[1][0]))), (255,0,0), 2)
    cv2.line(canvas, (int(p_d_range[0][1] * single_candle_w), int(scale_p(p_d_range[0][0]))), (int(p_d_range[1][1] * single_candle_w), int(scale_p(p_d_range[1][0]))), (255,50,50), 2)
    
    
    canvas = canvas[::-1]
    
    cv2.imshow("", canvas)
    
    cv2.waitKey()
        

#### actions
num_actions = 15
def_action = [0 for _ in range(num_actions)]
action_swing_high_created = 0
action_swing_low_created = 1
action_buyside_taken = 2 
action_sellside_taken = 3    

action_fvg_bull_created = 4
action_fvg_bear_created = 5
action_bull_fvg_retraded = 6
action_bear_fvg_retraded = 7
action_bull_fvg_closed = 8
action_bear_fvg_closed = 9

action_ob_bull_created = 10
action_ob_bear_created = 11
action_bull_ob_retraded = 12
action_bear_ob_retraded = 13
action_bull_ob_broken = 14
action_bear_ob_broken = 15

class liquidity:
    price = 0
    l_type = 0 ## 1 = buyside, -1 = sellside
    def __init__(self, price, buyside_sellside):
        self.price = price
        self.l_type = buyside_sellside
        
class fair_value_gap:
    bull_bear = 0
    low = 0
    high = 0
    retraded = False
    closed = False
    
    def __init__(self, bull_bear, low, high):
        self.bull_bear = bull_bear
        self.low = low
        self.closed = False
        self.retraded = False
        self.high = high

class xclosedcandles_broken:
    bull_bear = 0 # downclose broken = bull, upclose broken = bear
    low = 0
    high = 0
    retraded = 0
    closed = 0
    
    def __init__(self, bull_bear, low, high):
        self.bull_bear = bull_bear
        self.low = low
        self.closed = False
        self.retraded = False
        self.high = high
        
        
class candle:
    def __init__(self, candle):
        self.o = candle[1]
        self.h = candle[2]
        self.l = candle[3]
        self.c = candle[4]
        self.time = candle[0]
        
    buyside_taken = False
    sellside_taken = False
    swing_l_h = [False, False]
    
    xclosedcandles_broken = [0,0,0] # up/down, low, high   - debug
    
    action = def_action
    
    def add_action(self, index):
        self.action[index] = 1
        

class timeframechart:
    candles = []
    pd_arrays = []
    d_range = [(0,0), (0,0)]
    parent_d_range = [(0,0), (0,0)]
    current_dealing_range_direction = 0
    
    def __init__(self, symbol, timeframe):
        self.symbol = symbol
        self.timeframe = timeframe
        
    def reset(self):
        self.candles = []
        self.pd_arrays = []
        self.d_range = [(0,0), (0,0)]  # (price, candle), (price, candle)
        self.parent_d_range = [(0,0), (0,0)]
        self.liquidity_pools = []
        self.current_dealing_range_direction = 0
        
    def update(self):
        self.reset()
        prices = get_prices(self.symbol, self.timeframe)
        for i in prices.iloc:
            c = candle(list(i))
            self.candles.append(c)
        return self.process_candles()
    
    def detect_swing_point(self, current_candles):
        swing_high = False
        swing_low = False
        if current_candles[-1].h<current_candles[-2].h and current_candles[-3].h<current_candles[-2].h:
            swing_high = True
        if current_candles[-1].l>current_candles[-2].l and current_candles[-3].l>current_candles[-2].l:
            swing_low = True
        return swing_low, swing_high
        
    liquidity_pools = []
    def process_candles(self):
        current_candles = []
        last_liquidity_taken = 0
        
        for i in self.candles:
            current_candles.append(i)
            
            ###################swing points / liquidity buildup
            if len(current_candles) > 3:
                swing_l_h = self.detect_swing_point(current_candles)
                current_candles[-2].swing_l_h = swing_l_h
                
                if current_candles[-2].swing_l_h[0]:
                    self.liquidity_pools.append([current_candles[-2].l, "sellside", len(current_candles)-2, False])
                    current_candles[-2].add_action(action_swing_low_created)
                    
                if current_candles[-2].swing_l_h[1]:
                    self.liquidity_pools.append([current_candles[-2].h, "buyside", len(current_candles)-2, False])
                    current_candles[-2].add_action(action_swing_high_created)
                
            current_candles[-1].swing_l_h = [False, False]
            ###################swing points / liquidity buildup
            
            
            ################################## liquidity runs
            current_candles[-1].buyside_taken = False
            current_candles[-1].sellside_taken = False
            for i in range(len(self.liquidity_pools)):
                if current_candles[-1].h > self.liquidity_pools[i][0] and self.liquidity_pools[i][1] == "buyside":
                    current_candles[-1].buyside_taken = True
                    current_candles[-1].add_action(action_buyside_taken)
                    self.liquidity_pools[i][3] = True
                    last_liquidity_taken = "buyside"
                    
                if current_candles[-1].l < self.liquidity_pools[i][0] and self.liquidity_pools[i][1] == "sellside":
                    current_candles[-1].sellside_taken = True
                    current_candles[-1].add_action(action_sellside_taken)
                    self.liquidity_pools[i][3] = True
                    last_liquidity_taken = "sellside"
            
            #delete liquidity taken
            while True:
                sth = False
                for i in range(len(self.liquidity_pools)):
                    if self.liquidity_pools[i][3] == True:
                        del self.liquidity_pools[i]
                        sth = True
                        break
                if not sth: break
            ################################## liquidity runs
            
            ################################### dealing range
            if last_liquidity_taken == "buyside":
                if not self.current_dealing_range_direction == "bull":
                    self.parent_d_range = self.d_range
                    self.d_range = [self.d_range[0], (current_candles[-1].h, len(current_candles)-1)]
                    self.current_dealing_range_direction = "bull"
                
                if self.d_range[1][0] < current_candles[-1].h:
                    self.d_range[1] = (current_candles[-1].h, len(current_candles)-1)
                    
                    if self.d_range[1][0] > self.parent_d_range[1][0]:
                        for o in range(1, len(current_candles)):
                            index = len(current_candles) - o
                            found = False
                            if current_candles[index].buyside_taken == True and current_candles[index].h > self.d_range[1][0]:
                                for p in range(index, len(current_candles)):
                                    if current_candles[p].swing_l_h[1] == True:
                                        self.parent_d_range[1] = (current_candles[p].h, p)
                                        found = True
                                        break
                            if found:break
                            
                        for o in range(self.parent_d_range[1][1], len(current_candles)):
                            if current_candles[o].l < self.parent_d_range[0][0]:
                                self.parent_d_range[0] = (current_candles[o].l, o)
                                #self.d_range[0] = (current_candles[o].l, o)
                    
                                
                        
                        
                    
                
            if last_liquidity_taken == "sellside":
                if not self.current_dealing_range_direction == "bear":
                    self.parent_d_range = self.d_range
                    self.d_range = [(current_candles[-1].l, len(current_candles)-1), self.d_range[1]]
                    self.current_dealing_range_direction = "bear"

                if self.d_range[0][0] > current_candles[-1].l:
                    self.d_range[0] = (current_candles[-1].l, len(current_candles)-1)
                    
                    if self.d_range[0][0] < self.parent_d_range[0][0]:
                        for o in range(1, len(current_candles)):
                            index = len(current_candles) - o
                            found = False
                            if current_candles[index].sellside_taken == True and current_candles[index].l < self.d_range[0][0]:
                                for p in range(index, len(current_candles)):
                                    if current_candles[p].swing_l_h[0] == True:
                                        self.parent_d_range[0] = (current_candles[p].l, p)
                                        found = True
                                        break
                            if found:break
                            
                        for o in range(self.parent_d_range[0][1], len(current_candles)):
                            if current_candles[o].h > self.parent_d_range[1][0]:
                                self.parent_d_range[1] = (current_candles[o].h, o)
                                #self.d_range[1] = (current_candles[o].h, o)
            ################################### dealing range
            
            
        self.candles = current_candles
        
        
        ################## pd arrays
        for i in range(len(self.candles)):
            ### pd arrays repriced
            for o in range(len(self.pd_arrays)):
                
                #### fair value gap repricing
                if type(self.pd_arrays[o]) == fair_value_gap:
                    if self.pd_arrays[o].bull_bear == 1:
                        if self.candles[i].l < self.pd_arrays[o].high:
                            self.candles[i].add_action(action_bull_fvg_retraded)
                            self.pd_arrays[o].retraded = True
                            
                            if self.candles[i].l < self.pd_arrays[o].low:
                                self.candles[i].add_action(action_bull_fvg_closed)
                                self.pd_arrays[o].closed = True
                                
                                
                    if self.pd_arrays[o].bull_bear == -1:
                        if self.candles[i].h > self.pd_arrays[o].low:
                            self.candles[i].add_action(action_bear_fvg_retraded)
                            self.pd_arrays[o].retraded = True
                            
                            if self.candles[i].h > self.pd_arrays[o].high:
                                self.candles[i].add_action(action_bear_fvg_closed)
                                self.pd_arrays[o].closed = True
            
            
            
            ### fair value gaps ###
            if i > 1 and i + 1 < len(self.candles):
                ## bull fvg
                if self.candles[i-1].h < self.candles[i+1].l:
                    fvg = fair_value_gap(1, self.candles[i-1].h, self.candles[i+1].l)
                    self.pd_arrays.append(fvg)
                    self.candles[i].add_action(action_fvg_bull_created)
                
                ## bear fvg
                if self.candles[i-1].l > self.candles[i+1].h:
                    fvg = fair_value_gap(-1, self.candles[i+1].h, self.candles[i-1].l)
                    self.pd_arrays.append(fvg)
                    self.candles[i].add_action(action_fvg_bear_created)
                    
                    
            
        ######## xclosed candles broken
        last_xclosed_candles = [0,0,0]
        current_xclosed_candles = [0,0,0]
        ob_created = False
        for i in range(len(self.candles)):        
            current_candle_close = 1 if self.candles[i].c > self.candles[i].o else -1
            
            if current_xclosed_candles[0] != current_candle_close:
                last_xclosed_candles = current_xclosed_candles
                current_xclosed_candles = [current_candle_close, 1000000,0]
                ob_created = False
                
            current_xclosed_candles[1] = min(current_xclosed_candles[1], self.candles[i].l)
            current_xclosed_candles[2] = max(current_xclosed_candles[2], self.candles[i].h)
            
            if last_xclosed_candles[0] == -1 and self.candles[i].c > last_xclosed_candles[2]:
                #downclose broken
                if not ob_created:
                    ob = xclosedcandles_broken(1, last_xclosed_candles[1], last_xclosed_candles[2])
                    ob_created = True
                    self.pd_arrays.append(ob)
                    self.candles[i].add_action(action_ob_bull_created)
                    self.candles[i].xclosedcandles_broken = [1,last_xclosed_candles[1], last_xclosed_candles[2]]
            
            if last_xclosed_candles[0] == 1 and self.candles[i].c < last_xclosed_candles[1]:
                #downclose broken
                if not ob_created:
                    ob = xclosedcandles_broken(-1, last_xclosed_candles[1], last_xclosed_candles[2])
                    ob_created = True
                    self.pd_arrays.append(ob)
                    self.candles[i].add_action(action_ob_bear_created)
                    self.candles[i].xclosedcandles_broken = [-1,last_xclosed_candles[1], last_xclosed_candles[2]]
                    
        ######## xclosed candles broken
            
                    
        ################## pd arrays
        
        ## append liquidity pools to pd_arrays
        for i in self.liquidity_pools:
            if i[1] == "buyside":
                liquidity_pool = liquidity(i[0], 1)
            if i[1] == "sellside":
                liquidity_pool = liquidity(i[0], -1)
            self.pd_arrays.append(liquidity_pool)
        
        cv_plot_candles(self.candles, self.pd_arrays, self.d_range, self.parent_d_range)
        return self.candles, self.pd_arrays, self.d_range, self.parent_d_range, self.current_dealing_range_direction
            
    
class symboltimeframe:
    timeframecharts = []
    def __init__(self, symbol, timeframes):
        self.symbol = symbol
        self.timeframes = timeframes
        for i in self.timeframes:
            t = timeframechart(self.symbol, i)
            self.timeframecharts.append(t)
            
    def update(self):    
        for i in range(len(self.timeframecharts)):
            chart = self.timeframecharts[i].update()    
    
        
    
class manager:
    symboltimeframes = []
    def __init__(self, symbols, timeframes):
        self.symbols = symbols
        self.timeframes = timeframes
        
        for i in self.symbols:
            st = symboltimeframe(i, self.timeframes)
            self.symboltimeframes.append(st)
        
        
    def update(self):
        for i in range(len(self.symboltimeframes)):
            self.symboltimeframes[i].update()

m = manager(["GBPUSD"], [mt5.TIMEFRAME_H1])
m.update()