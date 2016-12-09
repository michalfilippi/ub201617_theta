import collections
import itertools


class HMM:
    """Class for representation of hidden markov models.
    """

    def __init__(self, transition_matrix, emission_matrix,
                 init_state_dist=None):
        """
        :param transition_matrix: dictionary of dictionaries representing
        probability matrix for transitions
        :param emission_matrix: dictionary of dictionaries representing
        probability matrix for emissions
        :param init_state_dist: dictionary of probabilities for state to be
        initial state, if None then uniform distribution is used
        """

        self.transition_matrix = transition_matrix
        self.emission_matrix = emission_matrix
        if init_state_dist == None:
            self.init_state_dist = dict()
            for state in transition_matrix:
                self.init_state_dist[state] = 1 / len(transition_matrix)
        else:
            self.init_state_dist = init_state_dist

    def path_prob(self, path):
        """Calculates probability of given path using probability matrix for
        transitions.

        :param path: path over states
        :return: probability of given path
        """

        prob = self.init_state_dist[path[0]]
        prev = path[0]
        for curr in path[1:]:
            prob *= self.transition_matrix[prev][curr]
            prev = curr

        return prob

    def output_prob_given_path(self, path, output):
        """Calculates probability of given output assuming path over states as
        given in 'path' using probability matrix for emissions.

        :param path: path over states
        :param output: output of HMM
        :return: probability of given output
        """

        prob = 1
        for o, s in zip(output, path):
            prob *= self.emission_matrix[s][o]

        return prob

    def most_probable_path_given_output(self, output):
        """Calculates the most probable path over states assuming seen output as
        given in 'output'.
        Algorithm is called Viterbi Algorithm.

        :param output: output of HMM
        :return: tuple (path, prob), where path is the most probable path over
        states and prob is it's probability
        """

        probs = dict()
        prev = dict()
        states = list(self.transition_matrix.keys())

        for state in states:
            emiss_prob = self.emission_matrix[state][output[0]]
            probs[(state, 0)] = self.init_state_dist[state] * emiss_prob

        for i, o in enumerate(output):
            if i == 0:
                continue

            for state in states:
                emiss_prob = self.emission_matrix[state][o]
                predecessors = []
                for prev_state in states:
                    trans_prob = self.transition_matrix[prev_state][state]
                    prob = probs[(prev_state, i - 1)] * trans_prob * emiss_prob
                    predecessors.append((prob, (prev_state, i - 1)))
                best_predecessor = max(predecessors)

                probs[(state, i)], prev[(state, i)] = best_predecessor

        # traceback most probable path
        reversed_path = []
        end_nodes = [(probs[(state, len(output) - 1)], (state, len(output) - 1))
                     for state in states]

        prob, current = max(end_nodes)
        reversed_path.append(current[0])

        while current[1] != 0:
            current = prev[current]
            reversed_path.append(current[0])

        return ''.join(reversed(reversed_path)), prob

    def states(self):
        """Returns unordered list of all possible states of HMM.

        :return: list of all possible states of HMM
        """

        return list(self.transition_matrix.keys())

    def alphabet(self):
        """Returns unordered list of characters being emitted from states of
        HMM. It might not return all the characters if the emission matrix is
        not complete.... too lazy to do it properly now. :) (ToDo)

        :return: list of characters being emitted from states of HMM
        """

        return list(self.emission_matrix[self.states()[0]].keys())

    def viterbi_learning(self, output, iterations):
        """Method for training HMM to fit given output as best as possible
        using Viterbi learning method. Both probability matrices need to be
        already initialized.

        :param output: given output of HMM
        :param iterations: number of learning iterations
        :return: None
        """

        for _ in range(iterations):
            path = self.most_probable_path_given_output(output)[0]

            new_tr_matrix = dict()
            new_em_matrix = dict()

            counter_em = collections.Counter(zip(path, output))
            for st in self.states():
                new_em_matrix[st] = dict()
                normalization_sum = 0

                # estimate probabilities
                for c in self.alphabet():
                    new_em_matrix[st][c] = counter_em[(st, c)]
                    normalization_sum += new_em_matrix[st][c]

                # normalize probabilities
                for c in self.alphabet():
                    if normalization_sum != 0:
                        new_em_matrix[st][c] /= normalization_sum
                    else:
                        # uniform distribution
                        new_em_matrix[st][c] = 1 / len(self.alphabet())

            counter_tr = collections.Counter(zip(path[:-1], path[1:]))
            for st_1 in self.states():
                new_tr_matrix[st_1] = dict()
                normalization_sum = 0

                # estimate probabilities
                for st_2 in self.states():
                    new_tr_matrix[st_1][st_2] = counter_tr[(st_1, st_2)]
                    normalization_sum += new_tr_matrix[st_1][st_2]

                # normalize probabilities
                for st_2 in self.states():
                    if normalization_sum != 0:
                        new_tr_matrix[st_1][st_2] /= normalization_sum
                    else:
                        # uniform distribution
                        new_tr_matrix[st_1][st_2] = 1 / len(self.states())

            # update matrices
            self.transition_matrix = new_tr_matrix
            self.emission_matrix = new_em_matrix

    def forward_filtering(self, output, normalize=True):
        """Calculates probability distributions for
        P(X_i=s, output[:i] | HMM) for all i and s.

        :param output: observed output of HMM
        :param normalize: True if probabilities should be normalized
        :return: dictionary of probability distributions for
        P(X_i=s, output[:i] | HMM)
        """

        states = self.states()
        p = dict()

        for i, c in enumerate(output):
            normalization_sum = 0

            if i == 0:
                for state in states:
                    # use uniform distribution for initial states
                    prob = (self.init_state_dist[state] *
                            self.emission_matrix[state][c])
                    p[(i, state)] = prob
                    normalization_sum += prob

            else:
                for s in states:
                    prob = 0
                    for prev_s in states:
                        prob += (self.transition_matrix[prev_s][s] *
                                 p[(i - 1, prev_s)])
                    prob *= self.emission_matrix[s][c]
                    p[(i, s)] = prob
                    normalization_sum += prob

            # normalize probabilities
            if normalize:
                for s in states:
                    p[(i, s)] /= normalization_sum

        return p

    def backward_filtering(self, output):
        """Calculates probability distributions for P(output[i+1:] | X_i=s, HMM)
        for all i and s.

        :param output: observed output of HMM
        :return: dictionary of probability distributions for
        P(output[i+1:] | X_i=s, HMM)
        """

        states = self.states()
        p = dict()

        for state in states:
            p[(len(output) - 1, state)] = 1

        for i, c in reversed(list(enumerate(output))):
            normalization_sum = 0

            for s in states:
                prob = 0
                for next_s in states:
                    prob += (p[(i, next_s)] *
                             self.transition_matrix[s][next_s] *
                             self.emission_matrix[next_s][c])
                p[(i - 1, s)] = prob
                normalization_sum += prob

            # normalize probabilities
            for s in states:
                p[(i - 1, s)] /= normalization_sum

        return p

    def baum_welch_learning(self, output, iterations):
        """Method for training HMM to fit given output as best as possible
        using Baum-Welch learning method. Both probability matrices need to be
        already initialized.
        https://en.wikipedia.org/wiki/Baum%E2%80%93Welch_algorithm

        :param output: given output of HMM
        :param iterations: number of learning iterations
        :return: None
        """

        states = self.states()
        len_o = len(output)

        for _ in range(iterations):
            f_filtering = self.forward_filtering(output)
            b_filtering = self.backward_filtering(output)

            new_tr_matrix = dict()
            new_em_matrix = dict()
            new_init_states = dict()

            gamma = dict()
            xi = dict()

            # calculate gamma
            # gamma[(t, s)] = P(X_t = s | output)
            for t in range(len_o):
                normalization_sum = 0
                for s in states:
                    gamma[(t, s)] = f_filtering[(t, s)] * b_filtering[(t, s)]
                    normalization_sum += gamma[(t, s)]

                # normalize
                for s in states:
                    gamma[(t, s)] /= normalization_sum

            # calculate xi
            # xi[(t, s_1, s_2)] = P(X_t = s_1, X_t+1 = s_2 | output)
            for t in range(len_o - 1):
                normalization_sum = 0
                for s_1, s_2 in itertools.product(states, repeat=2):
                    xi[(t, s_1, s_2)] = (f_filtering[(t, s_1)] *
                                         self.transition_matrix[s_1][s_2] *
                                         b_filtering[(t + 1, s_2)] *
                                         self.emission_matrix[s_2][output[t + 1]])
                    normalization_sum += xi[(t, s_1, s_2)]

                # normalize
                for s_1, s_2 in itertools.product(states, repeat=2):
                    xi[(t, s_1, s_2)] /= normalization_sum

            # calculate new transition matrix
            for s_1, s_2 in itertools.product(states, repeat=2):
                sum_gamma = sum([gamma[(t, s_1)] for t in range(len_o - 1)])
                sum_xi = sum([xi[(t, s_1, s_2)] for t in range(len_o - 1)])
                if s_1 not in new_tr_matrix:
                    new_tr_matrix[s_1] = dict()
                new_tr_matrix[s_1][s_2] = sum_xi / sum_gamma

            # calculate new emission matrix
            for s in states:
                gamma_s = [gamma[(t, s)] for t in range(len_o)]
                normalization_sum = sum(gamma_s)
                new_em_matrix[s] = dict()

                for c in self.alphabet():
                    fil_gamma_s = map(lambda x: x[0] if x[1] == c else 0,
                                      zip(gamma_s, output))
                    new_em_matrix[s][c] = sum(fil_gamma_s) / normalization_sum

            # calculate new init state distributions
            for s in states:
                new_init_states[s] = gamma[(0, s)]

            # update matrices
            self.transition_matrix = new_tr_matrix
            self.emission_matrix = new_em_matrix
            self.init_state_dist = new_init_states

    def output_prob(self, output):
        """Calculates P(output | HMM).

        :param output: given observed sequence
        :return: probability of output
        """

        f_filtering = self.forward_filtering(output, False)
        return sum([f_filtering[(len(output) - 1, s)] for s in self.states()])

    def __str__(self):
        """Converts HMM into string using the same notation as HMM problems on
        rosalind.

        :return: human readable description of HMM as string
        """

        output = []
        for state in sorted(self.states()):
            output.append(state)
            output.append("\t")
        output.append('\n')
        for state_1 in sorted(self.states()):
            output.append(state_1)
            output.append("\t")
            for state_2 in sorted(self.states()):
                prob = round(self.transition_matrix[state_1][state_2], 3)
                output.append(str(prob))
                output.append("\t")
            output.append("\n")
        output.append("--------\n")
        for c in sorted(self.alphabet()):
            output.append('\t')
            output.append(c)
        output.append("\n")
        for state in sorted(self.states()):
            output.append(state)
            output.append("\t")
            for c in sorted(self.alphabet()):
                prob = round(self.emission_matrix[state][c], 3)
                output.append(str(prob))
                output.append('\t')
            output.append("\n")
        return ''.join(output)

