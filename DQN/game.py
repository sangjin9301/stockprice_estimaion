
import numpy as np
import csv

class Game:

    def __init__(self):

        self.money = 100000 # 초기자본 10만원
        self.number_of_stock = 0 # 초기 주식수 0개
        self.open_price = 0
        self.close_price = 0
        self.states = self._read_file('D:\\stockprice_estimaion\\DATA\\new_A000020.csv')
        self.total_reward = 0.
        self.total_game = 0
        self.index_i = 0

    def _read_file(self, filename):
        with open(filename, 'r', encoding='utf-8') as fd:
            cin = csv.reader(fd)
            arrCsv = [row for row in cin]
            states = np.array(arrCsv)
            return states

    def _get_state(self):
        state = self.states[self.index_i, :40] # low
        self.open_price = float(state[0])
        self.close_price = float(state[3])
        return state

    def reset(self):
        self.total_game += 1
        self.money = 100000
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
        if (self.money + (self.number_of_stock * self.close_price)) < 0:
            return True
        else:
            return False

    def step(self, action):

        before = self.money + (self.number_of_stock * self.open_price)
        if action == 1:
            self._action_buy()
        elif action == 0:
            self._action_sell()

        after = self.money + (self.number_of_stock * self.close_price)
        game_over = self._is_game_over()

        if game_over:
            reward = -1000000
            print("---over-----------------------------------------------------------------------------")
        else:
            self.index_i += 1
            reward = (after - before)
            print("------------step--------------" + str(self.index_i))
            print("시가 : " + str(self.open_price))
            print("종가 : " + str(self.close_price))
            print("#stock : " + str(self.number_of_stock))
            print("action : "+str(action))  # 1:매수, 0:매각, 2:non-Action
            print("=수익 : " + str(reward))
            print("=보유주가 : " + str(self.close_price*self.number_of_stock))
            print("=자본 : " + str(self.money))
            print("=자본 + 보유주가 : " + str(self.money+(self.close_price*self.number_of_stock)))

        return self._get_state(), reward, game_over
