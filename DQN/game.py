
import numpy as np
import csv

class Game:

    def __init__(self):

        self.money = 1000000 # 초기자본 100만원
        self.number_of_stock = 0 # 초기 주식수 0개
        self.open_price = 0
        self.close_price = 0
        self.states = self._read_file('D:\\stockprice_estimaion\\DATA\\new_A000020.csv')

        self.total_reward = 0.
        self.current_reward = self.money
        self.total_game = 0
        self.index_i = 0

    def _read_file(self, filename):
        with open(filename, 'r', encoding='utf-8') as fd:
            cin = csv.reader(fd)
            arrCsv = [row for row in cin]
            states = np.array(arrCsv)
            return states

    def _get_state(self):
        state = self.states[self.index_i, 1:40] # low
        self.open_price = float(state[1])
        self.close_price = float(state[4])
        return state

    def reset(self):
        self.current_reward = self.money
        self.total_game += 1
        self.money = 1000000
        self.index_i = 0

        return self._get_state()

    def _action_buy(self):
        if self.money >= self.open_price:
            self.money -= self.open_price
            self.number_of_stock += 1

    def _action_sell(self):
        if self.number_of_stock > 0:
            self.money += self.open_price
            self.number_of_stock -= 1

    def _is_game_over(self):
        if self.current_reward < 0:
            return True
        else:
            return False

    def step(self, action):

        before = (self.money + (self.number_of_stock * self.open_price))

        if action == 0:
            self._action_buy()
        elif action == 1:
            self._action_sell()

        after = (self.money + (self.number_of_stock * self.close_price))

        game_over = self._is_game_over()

        if game_over:
            # 장애물에 충돌한 경우 -2점을 보상으로 줍니다. 장애물이 두 개이기 때문입니다.
            # 장애물을 회피했을 때 보상을 주지 않고, 충돌한 경우에만 -1점을 주어도 됩니다.
            reward = -100000
        else:
            self.index_i += 1
            reward = (before-after)
            self.current_reward += reward

        return self._get_state(), reward, game_over
