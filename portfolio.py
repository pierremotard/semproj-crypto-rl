import pandas as pd
import numpy as np

class Portfolio:
    def __init__(self, asset, initial_balance):
        self.initial_account_balance = initial_balance
        self.asset = asset
        self.orders = []
        self.balance = self.initial_account_balance
        self.net_worth = self.initial_account_balance
        self.shares_held = 0

        self.cost_basis = 0
        self.total_shares_sold = 0
        self.total_sales_value = 0
        self.PNL = 0

        print("--- START A NEW PORTFOLIO FOR ", self.asset, " ---")
        print(f'Available cash: {self.balance}')

    def __str__(self):
        print("Portfolio display:")
        print(f'    Balance available: {self.balance}')
        print(f'    PNL:               {self.PNL}')
        print(f'    Net worth:         {self.net_worth}')

    def get_orders(self):
        print("------")
        print("List orders")
        for order in self.orders:
            print(
                f' Order to {order.get_side()} {order.get_amount()} at {order.get_price()} on date {order.get_time()}')
        print("------")

    def update_balance(self, value):
        self.balance += value

    def get_balance(self):
        return self.balance

    def update_shares_held(self, value):
        self.shares_held += value

    def get_shares_held(self):
        return self.shares_held

    def update_net_worth(self, current_price):
        self.net_worth = self.balance + self.shares_held * current_price

    def get_net_worth(self, current_price):
        self.update_net_worth(current_price)
        return self.net_worth

    def get_infos(self):
        return np.array([
            self.balance,
            self.shares_held,
            self.net_worth
        ])
    
    def order_one(self, action_type, price, fee):
        if action_type == 'buy':
            full_cost = price + fee
            self.update_balance(-full_cost)
            self.update_shares_held(1)
            self.update_net_worth(price)
        elif action_type == 'sell':
            full_cost = price - fee
            self.update_balance(full_cost)
            self.update_shares_held(-1)
            self.update_net_worth(price)
        else:
            pass

    def execute_order(self, action_type, amount, price, fee):
        if action_type == 'buy':
            total_possible = self.balance / price
            shares_bought = total_possible * amount
            order_cost = shares_bought * price
            full_cost = order_cost + fee
            self.update_balance(-full_cost)
            self.update_shares_held(shares_bought)
            self.update_net_worth(price)
        elif action_type == 'sell':
            shares_sold = self.shares_held * amount
            order_cost = shares_sold * price
            full_cost = order_cost - fee
            self.update_balance(full_cost)
            self.update_shares_held(-shares_sold)
            self.update_net_worth(price)
        else:
            pass
        # self.get_portfolio()

    def reset(self):
        self.balance = self.initial_account_balance
        self.net_worth = self.initial_account_balance
        self.shares_held = 0
        self.cost_basis = 0
        self.total_shares_sold = 0
        self.total_sales_value = 0
        self.orders = []

    def get_portfolio(self, price):
        print(f'Balance: {self.balance}')
        print(f'Net worth: {self.get_net_worth(price)}')
        print(f'Change net worth : {(self.net_worth / self.initial_account_balance - 1) * 100}')
        print(f'Shares held: {self.shares_held}')
        print("-------------")
        return self.get_net_worth(price)

