# 장애물 회피 게임 즉, 자율주행차:-D 게임을 구현합니다.
import numpy as np
import random

class Game:
    
    def __init__(self, closing_price):
        """게임에 필요한 변수들 선언"""
        self.money = 1000000 # 초기자본 100만원
        self.number_of_stock = 0 # 초기 주식수 0개
        self.closing_price = closing_price # (.csv)에서 읽을 종가
        self.capital = (self.money + (self.number_of_stock * self.closing_price))

        self.total_reward = 0.
        self.current_reward = self.capital
        self.total_game = 0


    def _get_state(self):
        """ todo - 초기 게임 상태를 가져온다. => .csv의 first element만 가져오면 된다.
            1. 파일 전체를 읽어와서 2차원 배열로 만들고,
            2. 첫번째 요소를 반환한다.
        """

        state = np.zeros((self.screen_width, self.screen_height))

        state[self.car["col"], self.car["row"]] = 1

        if self.block[0]["row"] < self.screen_height:
            state[self.block[0]["col"], self.block[0]["row"]] = 1

        if self.block[1]["row"] < self.screen_height:
            state[self.block[1]["col"], self.block[1]["row"]] = 1

        return state


    def reset(self):
        """자동차, 장애물의 위치와 보상값들을 초기화합니다."""
        self.current_reward = 0
        self.total_game += 1

        self.car["col"] = int(self.screen_width / 2)

        self.block[0]["col"] = random.randrange(self.road_left, self.road_right + 1)
        self.block[0]["row"] = 0
        self.block[1]["col"] = random.randrange(self.road_left, self.road_right + 1)
        self.block[1]["row"] = 0

        self._update_block()

        return self._get_state()

    def _update_car(self, move):
        """액션에 따라 자동차를 이동시킵니다.
        자동차 위치 제한을 도로가 아니라 화면의 좌우측 끝으로 하고,
        도로를 넘어가면 패널티를 주도록 학습해서 도로를 넘지 않게 만들면 더욱 좋을 것 같습니다.
        """

        # 자동차의 위치가 도로의 좌측을 넘지 않도록 합니다: max(0, move) > 0
        self.car["col"] = max(self.road_left, self.car["col"] + move)
        # 자동차의 위치가 도로의 우측을 넘지 않도록 합니다.: min(max, screen_width) < screen_width
        self.car["col"] = min(self.car["col"], self.road_right)

    def _update_block(self):
        """장애물을 이동시킵니다.
        장애물이 화면 내에 있는 경우는 각각의 속도에 따라 위치 변경을,
        화면을 벗어난 경우에는 다시 방해를 시작하도록 재설정을 합니다.
        """
        reward = 0

        if self.block[0]["row"] > 0:
            self.block[0]["row"] -= self.block[0]["speed"]
        else:
            self.block[0]["col"] = random.randrange(self.road_left, self.road_right + 1)
            self.block[0]["row"] = self.screen_height
            reward += 1

        if self.block[1]["row"] > 0:
            self.block[1]["row"] -= self.block[1]["speed"]
        else:
            self.block[1]["col"] = random.randrange(self.road_left, self.road_right + 1)
            self.block[1]["row"] = self.screen_height
            reward += 1

        return reward

    def _is_gameover(self):
        # 장애물과 자동차가 충돌했는지를 파악합니다.
        # 사각형 박스의 충돌을 체크하는 것이 아니라 좌표를 체크하는 것이어서 화면에는 약간 다르게 보일 수 있습니다.
        if ((self.car["col"] == self.block[0]["col"] and
             self.car["row"] == self.block[0]["row"]) or
            (self.car["col"] == self.block[1]["col"] and
             self.car["row"] == self.block[1]["row"])):

            self.total_reward += self.current_reward

            return True
        else:
            return False

    def step(self, action):
        # action: 0: 좌, 1: 유지, 2: 우
        # action - 1 을 하여, 좌표를 액션이 0 일 경우 -1 만큼, 2 일 경우 1 만큼 옮깁니다.
        self._update_car(action - 1)
        # 장애물을 이동시킵니다. 장애물이 자동차에 충돌하지 않고 화면을 모두 지나가면 보상을 얻습니다.
        escape_reward = self._update_block()
        # 움직임이 적을 경우에도 보상을 줘서 안정적으로 이동하는 것 처럼 보이게 만듭니다.
        stable_reward = 1. / self.screen_height if action == 1 else 0
        # 게임이 종료됐는지를 판단합니다. 자동차와 장애물이 충돌했는지를 파악합니다.
        gameover = self._is_gameover()

        if gameover:
            # 장애물에 충돌한 경우 -2점을 보상으로 줍니다. 장애물이 두 개이기 때문입니다.
            # 장애물을 회피했을 때 보상을 주지 않고, 충돌한 경우에만 -1점을 주어도 됩니다.
            reward = -2
        else:
            reward = escape_reward + stable_reward
            self.current_reward += reward

        return self._get_state(), reward, gameover