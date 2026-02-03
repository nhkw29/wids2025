import gymnasium as gym
from gymnasium import spaces
import numpy as np 
import random

from engine.matching_engine import MatchingEngine
from engine.order import Order
from engine.event_loop import EventLoop
from agents.agents import MarketMaker, NoiseTrader

class GymTradingEnvironment(gym.Env):
    metadata = {'render_modes': ['human']}

    def __init__(self):
        super(GymTradingEnvironment, self).__init__()
        
        self.loop = EventLoop()
        self.action_space = spaces.Discrete(3)
        
        self.observation_space = spaces.Box(low=-10.0, high=10.0, shape=(5,), dtype=np.float32)

        self.order_book = None
        self.agents = []
        self.max_steps = 1000  
        
        self.insider_inventory = 0
        self.cash_balance = 100000.0
        self.portfolio_value = 100000.0
        self.tape_reader_index = 0
        
        self.peak_portfolio_value = 100000.0
        self.max_drawdown = 0.0
        self.risk_aversion = 0.5  

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.loop = EventLoop()
        self.order_book = MatchingEngine()
        
        self.insider_inventory = 0
        self.cash_balance = 100000.0
        self.portfolio_value = 100000.0
        self.tape_reader_index = 0
        
        self.peak_portfolio_value = 100000.0
        self.max_drawdown = 0.0
        
        self.agents = []
        for i in range(5): 
            self.agents.append(MarketMaker(f"MM_{i}", inventory_limit=1000))
        for i in range(10): 
            self.agents.append(NoiseTrader(f"NT_{i}", sigma=3.0))
        random.shuffle(self.agents)

        self.loop.schedule(0.1, self._background_agent_step)
        self.loop.run_until(20.0)

        self.tape_reader_index = len(self.order_book.tape)
        
        return self._get_obs(), {}

    def step(self, action):
        fixed_qty = 10 
        snap = self.order_book.get_snapshot()
        mid_price = snap['mid_price'] if snap['mid_price'] > 0 else 100.0
        
        aggressive_offset = 0.05 
        if action == 1: 
            price = round(mid_price + aggressive_offset, 2)
            self._place_order('buy', price, fixed_qty)
        elif action == 2: 
            price = round(mid_price - aggressive_offset, 2)
            self._place_order('sell', price, fixed_qty)
            
        self.loop.run_until(self.loop.current_time + 1.0)
        self._process_fills()

        prev_value = self.portfolio_value
        self.portfolio_value = self.cash_balance + (self.insider_inventory * mid_price)
        
        
        pnl_diff = (self.portfolio_value - prev_value) / 100.0
        
        self.peak_portfolio_value = max(self.peak_portfolio_value, self.portfolio_value)
        current_drawdown = (self.peak_portfolio_value - self.portfolio_value) / self.peak_portfolio_value
        self.max_drawdown = max(self.max_drawdown, current_drawdown)
        
        drawdown_penalty = current_drawdown * 10.0 
        
        inventory_penalty = (self.insider_inventory / 100.0) ** 2
        
        reward = pnl_diff - (self.risk_aversion * (drawdown_penalty + inventory_penalty))

        terminated = self.cash_balance <= 0 
        truncated = self.loop.current_time >= (self.max_steps + 20.0) 
        
        info = {
            "pnl": self.portfolio_value - 100000.0,
            "drawdown": current_drawdown,
            "inventory": self.insider_inventory,
            "reward_pnl_component": pnl_diff,
            "reward_penalty_component": (drawdown_penalty + inventory_penalty)
        }
        
        return self._get_obs(), reward, terminated, truncated, info

    def _get_obs(self):
        snap = self.order_book.get_snapshot()
        mid = snap['mid_price'] if snap['mid_price'] > 0 else 100.0

        rel_bid = (snap['best_bid'] - mid) / mid if snap['best_bid'] else 0
        rel_ask = (snap['best_ask'] - mid) / mid if snap['best_ask'] else 0
        rel_spread = snap['spread'] / mid
        norm_inventory = self.insider_inventory / 100.0
        norm_cash = (self.cash_balance - 100000.0) / 10000.0

        return np.array([rel_bid, rel_ask, rel_spread, norm_inventory, norm_cash], dtype=np.float32)

    def _place_order(self, side, price, qty):
        def execute():
            unique_id = int(self.loop.current_time * 1000000) + random.randint(0, 1000)
            order = Order(
                agent_id="Insider",
                side=side,
                price=price,
                qty=qty,
                order_type='limit',
                timestamp=self.loop.current_time,
                order_id=f"RL_{unique_id}"
            )
            self.order_book.add_order(order)
        self.loop.schedule(0.05, execute)

    def _process_fills(self):
        current_tape = self.order_book.tape
        for i in range(self.tape_reader_index, len(current_tape)):
            trade = current_tape[i]
            if trade.buyer_id == "Insider":
                self.insider_inventory += trade.qty
                self.cash_balance -= trade.qty * trade.price
            elif trade.seller_id == "Insider":
                self.insider_inventory -= trade.qty
                self.cash_balance += trade.qty * trade.price
        self.tape_reader_index = len(current_tape)

    def _background_agent_step(self):
        agent = random.choice(self.agents)
        snap = self.order_book.get_snapshot()
        actions = agent.act(snap)
        
        if isinstance(actions, dict): actions = [actions]
        elif actions is None: actions = []

        for action in actions:
            if action['type'] == 'CANCEL':
                self.order_book.cancel_order(action['order_id'])
            else:
                order_type_str = action.get('type', 'limit')
                if '_' in order_type_str:
                     order_type = order_type_str.split('_')[-1].lower()
                else:
                     order_type = order_type_str.lower()
                
                oid = action.get('order_id', f"{action['agent_id']}_{random.randint(0, 1000000000)}")

                o = Order(
                    agent_id=action['agent_id'],
                    side=action['side'],
                    qty=action['qty'],
                    price=action.get('price'),
                    order_type=order_type,
                    timestamp=self.loop.current_time,
                    order_id=oid
                )
                self.order_book.add_order(o)
            
        next_delay = random.uniform(0.01, 0.1) 
        self.loop.schedule(next_delay, self._background_agent_step)