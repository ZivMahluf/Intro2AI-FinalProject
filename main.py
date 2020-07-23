from AggressivePlayer import AggressivePlayer
from BasicPlayer import BasicPlayer
from DefensivePlayer import DefensivePlayer
from HumanPlayer import HumanPlayer
from LearningPlayer import LearningPlayer
from RandomPlayer import RandomPlayer
from DurakGameRunner import DurakRunner


def main():
    games = 3
    game = DurakRunner()
    game.add_player(AggressivePlayer(game.HAND_SIZE, "Ziv"))
    game.add_player(BasicPlayer(game.HAND_SIZE, "Idan"))
    game.add_player(HumanPlayer(game.HAND_SIZE, "Vitaly"))
    game.add_player(BasicPlayer(game.HAND_SIZE, "Eyal"))
    game.add_player(RandomPlayer(game.HAND_SIZE, "Yoni"))
    game.add_player(DefensivePlayer(game.HAND_SIZE, "Jeff"))
    game.play_games(games, render=True, verbose=True)
    game.end()


if __name__ == '__main__':
    main()