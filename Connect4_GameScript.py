import numpy as np
import random as rand
import time
import os

# This script allows you to play Connect4 against an ai.

# This is a wrapper class that holds the logic engine and returns a decision.
class logic_engine_wrapper:

    # Defaults logic_engine to 0 so that it will just move over one space each turn.
    def __init__(self,logic_engine=0):
        self.logic_engine = logic_engine

    # Returns the logic_engine's move based on the provided board.
    def get_move(self,board):
        if type(self.logic_engine) is int:
            if self.logic_engine >= 6:
                self.logic_engine = 0
                return 6
            else:
                self.logic_engine += 1
                return self.logic_engine - 1
        else:
            output = self.logic_engine.get_output_vector(board.get_board_vector())
            greatest = 0
            for i in range(len(output)):
                if output[i] > output[greatest]:
                    greatest = i
            return greatest

# This class represents a Connect4 board and contains all necessary utilities for it.
# On boards player1 = 1, player2 = -1, and blank = 0.
# The higher the first index the higher on the board i.e. index [5][6] is the top right corner of the board.
class game_board:

    # Creates a 7 x 6 game board.
    def __init__(self):
        self.board = [[0.0 for _ in range(7)] for _ in range(6)]

    # Prints the current game board.
    def print_board(self,player1_symbol='@',player2_symbol='#',clear=False):
        if clear:
            os.system('cls')
        for row in self.board[::-1]:
            print('|',end='')
            for num in row:
                symbol = player1_symbol if num == 1 else (player2_symbol if num == -1 else ' ')
                print(symbol,end='')
            print('|')
        print('---------')
        print(' 1234567 ')

    # Drops a piece into the indicated slot using a range of [0,6]
    def drop_piece(self,slot,player):
        if slot > 6 or slot < 0:
            return False
        else:
            for i in range(6):
                if self.board[i][slot] == 0:
                    self.board[i][slot] = player
                    return True
            return False

    # Returns the board squashed into a numpy array. (This is a utility function for net use)
    def get_board_vector(self):
        board_vector = []
        for row in self.board:
            board_vector += row
        return np.array(board_vector)

    # Returns a player if one of them as won otherwise returns 0.
    def get_winner(self):
        # Checks rows.
        for row in self.board:
            previous = None
            count = 0
            for num in row:
                if count >= 4 and previous != 0:
                    return previous
                if num == previous:
                    count += 1
                else:
                    previous = num
                    count = 1
        # Checks columns.
        for x in range(7):
            previous = None
            count = 0
            for y in range(6):
                num = self.board[y][x]
                if count >= 4 and previous != 0:
                    return previous
                if num == previous:
                    count += 1
                else:
                    previous = num
                    count = 1
        # Checks diagonals.
        for y in range(6):
            for x in range(7):
                num = self.board[y][x]
                if num != 0 and y < 3:
                    if x < 4:
                        if num == self.board[y+1][x+1] and num == self.board[y+2][x+2] and num == self.board[y+3][x+3]:
                            return num
                    elif x > 2:
                        if num == self.board[y+1][x-1] and num == self.board[y+2][x-2] and num == self.board[y+3][x-3]:
                            return num

        return 0


# This function enables the user to play against a logic_engine or lets two logic_engines play against each other.
def play_game(logic_engines=[logic_engine_wrapper()],verbose=True,ai_delay=0.2):
    board = game_board()
    current_player = 1 if rand.choice([True,False]) else -1
    max_num_of_turns = 6 * 7
    for turn in range(max_num_of_turns):
        if verbose:
            board.print_board(clear=True)
        if len(logic_engines) == 2:
            if verbose:
                print("Thinking...")
            time.sleep(ai_delay)
            slot = None
            if current_player == 1:
                proxy_board = game_board()
                for y in range(len(proxy_board.board)):
                    for x in range(len(proxy_board.board[y])):
                        proxy_board.board[y][x] = -1 * board.board[y][x]
                slot = logic_engines[0].get_move(proxy_board)
            else:
                slot = logic_engines[1].get_move(board)
            board.drop_piece(slot,current_player)
        else:
            if current_player == 1:
                slot = int(input('Input a slot: ')) - 1
                board.drop_piece(slot,current_player)
            else:
                if verbose:
                    print("Thinking...")
                time.sleep(ai_delay)
                slot = logic_engines[0].get_move(board)
                board.drop_piece(slot,current_player)
        winner = board.get_winner()
        if winner != 0:
            if verbose:
                board.print_board(clear=True)
            return winner
        else:
            current_player = 1 if current_player == -1 else -1
        if turn == (6 * 7) - 1 and verbose:
            board.print_board(clear=True)
    return 0

# Classes and functions for network interconnection.

#Sigmoid function.
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Info String Seperators:
net_section_seperator = '|'
net_sub_section_seperator = ':' # i.e. each hidden layer and each bias layer
net_matrix_row_seperator = '>'
net_vector_index_seperator = ','

# Feedforward network class that's used.
class net:

    # wiki_net class constructor.
    def __init__(self,input_size,hidden_layers,non_linears=sigmoid,use_biases=True,name="N/A"):
        self.hidden_layers = []
        self.non_linears = []
        self.biases = []
        self.name = name
        if type(non_linears) != list:
            non_linears = [non_linears] * len(hidden_layers)
            previous_size = input_size
        elif len(non_linears) == 1:
            non_linears *= len(hidden_layers)
        for i in range(len(hidden_layers)):
            self.hidden_layers.append(2 * np.random.random((previous_size,hidden_layers[i])) - 1)
            self.non_linears.append(non_linears[i])
            self.biases.append(2 * np.random.random(hidden_layers[i]) - 1 if use_biases else np.zeros(hidden_layers[i]))
            previous_size = hidden_layers[i]

    # Gets the output vector for a given input vector.
    def get_output_vector(self,input_vector):
        output_vector = input_vector
        for i in range(len(self.hidden_layers)):
            output_vector = self.non_linears[i](np.dot(output_vector,self.hidden_layers[i] + self.biases[i]))
        return output_vector

    # Returns a string that contains all of the network's features in the following order, [Name, Score, Hidden_Layers, Biases]
    def generate_info_string(self,score=0):
        info_string = f'{self.name}{net_section_seperator}{score}{net_section_seperator}'
        count_1 = 0
        count_2 = 0
        count_3 = 0
        for layer in self.hidden_layers:
            for vector in layer:
                for index in vector:
                    info_string += f'{index}'
                    info_string += net_vector_index_seperator if count_1 != len(vector) - 1 else ''
                    count_1 += 1
                count_1 = 0
                info_string += net_matrix_row_seperator if count_2 != len(layer) - 1 else ''
                count_2 += 1
            count_2 = 0
            info_string += net_sub_section_seperator if count_3 != len(self.hidden_layers) - 1 else ''
            count_3 += 1
        info_string += net_section_seperator
        count_1 = 0
        count_2 = 0
        for bias in self.biases:
            for index in bias:
                info_string += f'{index}'
                info_string += net_vector_index_seperator if count_1 != len(bias) - 1 else ''
                count_1 += 1
            count_1 = 0
            info_string += net_sub_section_seperator if count_2 != len(self.biases) - 1 else ''
            count_2 += 1
        return info_string

    # Loads the net's attribute's from a given string and returns it's score.
    # String Format is listed above the generate_info_string method
    def load_info_from_string(self,info_string,non_linear=sigmoid):
        sections = info_string.split(net_section_seperator)
        self.name = sections[0]
        score = float(sections[1])
        self.hidden_layers = []
        self.biases = []
        self.non_linears = [non_linear]
        hidden_layer_sections = sections[2].split(net_sub_section_seperator)
        for layer in hidden_layer_sections:
            vectors = []
            for text_vector in layer.split(net_matrix_row_seperator):
                vector = []
                for index in text_vector.split(net_vector_index_seperator):
                    vector.append(float(index))
                vectors.append(vector)
            self.hidden_layers.append(np.array(vectors))
        self.non_linears *= len(self.hidden_layers)
        text_biases = sections[3].split(net_sub_section_seperator)
        for text_bias in text_biases:
            bias = []
            for index in text_bias.split(net_vector_index_seperator):
                bias.append(float(index))
            self.biases.append(np.array(bias))
        return score

# Generates a net_net with preset values (this is basically a way to make net generation easier).
def generate_standard_net(name="N/A"):
    return net(input_size=42,hidden_layers=[7],name=name)

# Loads a net population from a given file in format (population,generation)
def load_wrapped_net_population(name):
    file = None
    try:
        file = open(f'Populations/{name}.snp','r')
    except:
        print(f'Failed to find file for population : {name}')
        return None
    if file != None:
        contents = file.read().split('\n')
        generation = int(contents[0].split(' ')[1])
        population = []
        for val in contents[1:]:
            net = generate_standard_net()
            score = net.load_info_from_string(val)
            population.append((net,score))
        return (population,generation)

# Allows the user to watch all the nets in the population duke it out.
def watch_net_games(population,auto_run=False):
    for i in range(len(population) - 1):
        for j in range(len(population) - i):
            if j != 0:
                logic_engine_1 = logic_engine_wrapper(logic_engine=population[i][0])
                logic_engine_2 = logic_engine_wrapper(logic_engine=population[i+j][0])
                result = play_game(logic_engines=[logic_engine_1,logic_engine_2],verbose=True)
                print(f'WINNER : {logic_engine_1.logic_engine.name if result == 1 else logic_engine_2.logic_engine.name}')
                if not auto_run:
                    input('PRESS ANY KEY TO CONTINUE')
                else:
                    time.sleep(0.75)
    print('ALL GAMES COMPLETED')

population = load_wrapped_net_population('eigthTrial')[0]
#watch_net_games(population,auto_run=True)
logic_engine_1 = logic_engine_wrapper(logic_engine=population[0][0])
print(play_game(logic_engines=[logic_engine_1]))
#print(play_game())
