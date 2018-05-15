
import numpy as np
import csv

class Game:

    def __init__(self):

        self.INIT_MONEY = 10000000
        self.money = self.INIT_MONEY # 초기자본 1000만원
        self.number_of_stock = 0 # 초기 주식수 0개
        self.open_price = 0
        self.close_price = 0
        self.states = self._read_file('D:\\stockprice_estimaion\\DATA\\new_A005930.csv')
        self.total_reward = 0.
        self.total_game = 0
        self.index_i = 0
        self.current_reward = 0
        self.win_rate = 0
        self.deposit_1 = 0
        self.deposit_2 = 0

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
        self.rate = float(self.states[self.index_i, 40])
        return state

    def reset(self):
        self.total_game += 1
        self.money = self.INIT_MONEY
        self.index_i = 0
        self.win_rate = 0
        self.number_of_stock = 0
        self.states = self._read_file('D:\\stockprice_estimaion\\DATA\\new_A005930.csv')

        return self._get_state()

    def _action_buy(self):
        quantity = self.money*0.5/self.open_price
        self.money -= self.open_price*quantity
        self.money -= (self.open_price*quantity * 0.0016)+700
        self.number_of_stock += quantity

    def _action_sell(self):
        self.deposit_2 = self.open_price*(self.number_of_stock/2)
        self.money -= (self.open_price*(self.number_of_stock/2)* 0.0016)+700 # 수수료
        self.number_of_stock = self.number_of_stock/2

    def _action_nothing(self):
        self.money -= 700

    def _is_game_over(self):
        if ((self.money + (self.number_of_stock * self.close_price)) < 0) | (self.index_i > 999):
            self.total_reward = self.money
            return True
        else:
            return False

    def step(self, action):
        self.money += self.deposit_1
        self.deposit_1 = self.deposit_2
        self.deposit_2 = 0

        before = self.money + (self.number_of_stock * self.open_price) + self.deposit_2 + self.deposit_1
        if action == 1:
            self._action_buy()
        elif action == 0:
            self._action_sell()
        else:
            self._action_nothing()

        after = self.money + (self.number_of_stock * self.close_price) + self.deposit_2 + self.deposit_1
        game_over = self._is_game_over()

        reward = self.rate*(self.number_of_stock*self.close_price)-700
        if game_over:
            # print("---over-----------------------------------------------------------------------------")
            # print("Rate : "+str(self.win_rate/self.index_i))
            self.current_reward = (self.money+(self.close_price*self.number_of_stock))

        else:
            self.index_i += 1
            if (action == 1) & (reward > 0):
                self.win_rate += 1
            elif (action == 0) & (reward <0):
                self.win_rate += 1
            print("------------step--------------" + str(self.index_i))
            print("시가 : " + str(self.open_price))
            print("종가 : " + str(self.close_price))
            print("#stock : " + str(self.number_of_stock))
            # print("action : "+str(action))  # 0:매각, 1:매수, 2:non-Action
            print("=수익 : " + str(reward))
            # print("=보유주가 : " + str(self.close_price*self.number_of_stock))
            # print("=자본 : " + str(self.money))
            print("=자산 : " + str(self.money+(self.close_price*self.number_of_stock)+self.deposit_2+self.deposit_1))

        return self._get_state(), reward, game_over, (self.money+(self.close_price*self.number_of_stock)+self.deposit_2+self.deposit_1)
