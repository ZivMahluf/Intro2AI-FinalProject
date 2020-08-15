from Types import Tuple
import numpy as np
from PPONetwork import PPONetwork, PPOModel
import tensorflow as tf
import joblib
from Deck import Deck
from DurakEnv import DurakEnv
from PPOPlayer import PPOPlayer
import logging
import time
import os


class PPOTrainer(object):

    def __init__(self, sess, *, games_per_batch=5, training_steps_per_game=5, lam=0.95, gamma=0.995, ent_coef=0.01, vf_coef=0.5,
                 max_grad_norm=0.5, min_learning_rate=0.000001, learning_rate, clip_range, save_every=100):
        """
        Constructor.
        Sets up the parameters of the training networks and the algorithm.
        :param sess: Tensorflow session in which to run.
        :param games_per_batch: Number of games in each batch.
        :param training_steps_per_game: Number of training steps performed in each game.
        :param lam: coefficient used to calculate loss
        :param gamma: discount factor.
        :param ent_coef: coefficient used to calculate loss
        :param vf_coef: coefficient used to calculate loss
        :param max_grad_norm: coefficient used to calculate loss
        :param min_learning_rate: minimal learning rate.
        :param learning_rate: learning rate of the players.
        :param clip_range: used to prevent large changes in the network
        :param save_every: Frequency of saving of the model.
        """
        # network/model for training
        output_dim = PPOPlayer.output_dim
        input_dim = PPOPlayer.input_dim
        self.trainingNetwork = PPONetwork(sess, input_dim, output_dim, "trainNet")
        self.trainingModel = PPOModel(sess, self.trainingNetwork, input_dim, output_dim, ent_coef, vf_coef, max_grad_norm)

        # player networks which makes decisions -
        # allowing for later on experimenting with playing against older versions of the network (so decisions they make are not trained on).
        self.playerNetworks = dict()

        # for now each player uses the same (up to date) network to make it's decisions.
        self.playerNetworks[1] = self.playerNetworks[2] = self.playerNetworks[3] = self.playerNetworks[4] = self.trainingNetwork
        self.trainOnPlayer = [True, True, True, True]

        tf.compat.v1.global_variables_initializer().run(session=sess)

        self.players = [PPOPlayer(DurakEnv.HAND_SIZE, "PPO 0", self.playerNetworks[1]),
                        PPOPlayer(DurakEnv.HAND_SIZE, "PPO 1", self.playerNetworks[2]),
                        PPOPlayer(DurakEnv.HAND_SIZE, "PPO 2", self.playerNetworks[3]),
                        PPOPlayer(DurakEnv.HAND_SIZE, "PPO 3", self.playerNetworks[4])]

        # environment
        game = DurakEnv(self.players, False)
        self.state = game.reset()
        self.vectorizedGame = game

        # params
        self.games_per_batch = games_per_batch
        self.training_steps_per_game = training_steps_per_game
        self.inpDim = input_dim
        self.lam = lam
        self.gamma = gamma
        self.learningRate = learning_rate
        self.minLearningRate = min_learning_rate
        self.clipRange = clip_range
        self.saveEvery = save_every

        self.rewardNormalization = 5.0  # divide rewards by this number (so reward ranges from -1.0 to 3.0)

        # test networks - keep network saved periodically and run test games against current network
        self.testNetworks = {}

        # final 4 observations need to be carried over (for value estimation and propagating rewards back)
        self.prevObs = []
        self.prevGos = []
        self.prevAvailAcs = []
        self.prevRewards = []
        self.prevActions = []
        self.prevValues = []
        self.prevDones = []
        self.prevNeglogpacs = []

        # episode/training information
        self.totTrainingSteps = 0
        self.gamesDone = 0
        self.losses = []

        logging.info("finished PPO Trainers init")

    def run(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Runs a game for a number of steps and collects information of the game.
        :return: information collected from the game.
        """
        # run vectorized games for nSteps and generate mini batch to train on.
        mb_obs, mb_pGos, mb_actions, mb_values, mb_neglogpacs, mb_rewards, mb_dones, mb_availAcs = [], [], [], [], [], [], [], []
        done = False
        game = self.vectorizedGame
        state = game.reset()
        steps = 0
        while not done and steps < 500:
            turn_player = game.get_turn_player()
            available_actions = game.get_available_actions()
            action = turn_player.get_action(state, game.to_attack())
            value, neglogpac = turn_player.get_val_neglogpac()
            new_state, reward, done = game.step(action)  # update the game

            # add to list
            mb_obs.append(turn_player.last_converted_state.flatten())
            mb_pGos.append(turn_player)
            mb_actions.append(Deck.get_index_from_card(action))
            mb_values.append(value[0])
            mb_neglogpacs.append(neglogpac[0])
            mb_rewards.append(reward)
            mb_dones.append(done)
            mb_availAcs.append(turn_player.last_converted_available_cards.flatten())

            # update current state
            state = new_state
            steps += 1

        # add dones to last plays
        for i in range(1, len(game.players) + 1):
            mb_dones[-i] = True

        # convert to numpy and finish game
        self.gamesDone += 1
        self.vectorizedGame = game
        self.state = state
        mb_obs = np.asarray(tuple(mb_obs), dtype=np.float64)
        mb_availAcs = np.asarray(tuple(mb_availAcs), dtype=np.float64)
        mb_rewards = np.asarray(tuple(mb_rewards), dtype=np.float64)
        mb_actions = np.asarray(tuple(mb_actions), dtype=np.int64)
        mb_values = np.asarray(tuple(mb_values), dtype=np.float64)
        mb_neglogpacs = np.asarray(tuple(mb_neglogpacs), dtype=np.float64)
        mb_dones = np.asarray(tuple(mb_dones), dtype=np.bool)

        # discount/bootstrap value function with generalized advantage estimation:
        mb_returns = np.zeros_like(mb_rewards)
        mb_advs = np.zeros_like(mb_rewards)
        for k in range(4):
            lastgaelam = 0
            for t in reversed(range(k, len(mb_rewards) - 4, len(game.players))):
                nextNonTerminal = 1.0 - mb_dones[t]
                nextValues = mb_values[t + len(game.players)]
                delta = mb_rewards[t] + self.gamma * nextValues * nextNonTerminal - mb_values[t]
                mb_advs[t] = lastgaelam = delta + self.gamma * self.lam * nextNonTerminal * lastgaelam

        mb_values = mb_values
        mb_returns = mb_advs + mb_values

        return mb_obs, mb_availAcs, mb_returns, mb_actions, mb_values, mb_neglogpacs

    def get_batch(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        :return: Information regarding a batch of games for training.
        """
        states, availAcs, returns, actions, values, neglogpacs = [], [], [], [], [], []
        for _ in range(self.games_per_batch):
            st, av, re, ac, va, ne = self.run()
            states.append(st)
            availAcs.append(av)
            returns.append(re)
            actions.append(ac)
            values.append(va)
            neglogpacs.append(ne)

        # convert to numpy
        states = np.asarray(tuple(states), dtype=object)
        availAcs = np.asarray(tuple(availAcs), dtype=object)
        returns = np.asarray(tuple(returns), dtype=object)
        actions = np.asarray(tuple(actions), dtype=object)
        values = np.asarray(tuple(values), dtype=object)
        neglogpacs = np.asarray(tuple(neglogpacs), dtype=object)

        logging.info("Finished getting a batch")

        return states, availAcs, returns, actions, values, neglogpacs

    def train(self, total_num_games) -> None:
        """
        Trains the PPO players against themselves.
        :param total_num_games: Number of training games.
        """

        nUpdates = total_num_games // self.games_per_batch

        for update in range(1, nUpdates + 1):

            alpha = 1.0 - update / nUpdates
            lrnow = self.learningRate * alpha
            if lrnow < self.minLearningRate:
                lrnow = self.minLearningRate
            cliprangenow = self.clipRange * alpha

            states, availAcs, returns, actions, values, neglogpacs = self.get_batch()

            curr_params = self.trainingNetwork.getParams()
            mb_lossvals = []

            # train on the games after shuffling them
            for game_idx in np.random.randint(0, self.games_per_batch - 1, self.games_per_batch):
                steps = 0
                while steps + self.training_steps_per_game < states[game_idx].shape[0]:  # less than the amount of steps (actions) in the game
                    mb_inds = np.arange(steps, steps + self.training_steps_per_game)
                    mb_lossvals.append(self.trainingModel.train(lrnow, cliprangenow, states[game_idx][mb_inds],
                                                                availAcs[game_idx][mb_inds], returns[game_idx][mb_inds],
                                                                actions[game_idx][mb_inds], values[game_idx][mb_inds],
                                                                neglogpacs[game_idx][mb_inds]))
                    if steps == states[game_idx].shape[0] - 1:
                        break
                    if steps + self.training_steps_per_game < states[game_idx].shape[0]:
                        # basic step
                        steps += self.training_steps_per_game
                    else:
                        # go over last indices, which are less than self.training_steps_per_game
                        steps = states[game_idx].shape[0] - self.training_steps_per_game - 1

            logging.info("Finished training in update num: %s" % update * self.games_per_batch)

            lossvals = np.mean(mb_lossvals, axis=0)
            self.losses.append(lossvals)

            new_params = self.trainingNetwork.getParams()
            for param in new_params:
                if np.sum(np.isnan(param)) > 0:
                    # remove changes in network
                    self.trainingNetwork.loadParams(curr_params)
                    logging.warning("Had to reset the params in update num: %s" % nUpdates)
                    break

            if update % self.saveEvery == 0:
                name = "PPOParams/model" + str(update * self.games_per_batch)
                self.trainingNetwork.saveParams(name)
                joblib.dump(self.losses, "losses.pkl")

            print("finished " + str(update * self.games_per_batch))
