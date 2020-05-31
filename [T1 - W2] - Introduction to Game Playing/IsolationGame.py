# TODO: implement the __init__ class below by adding properties
# that meet the three requirements specified
import copy 

class GameState:

    def __init__(self):        
        self.cellState = False
        self.playerTurn = 1
        self.currentPosition_P1 = [0,0]
        self.currentPosition_P2 = [0,0] 
        self.board = [[0, 0 ,0], [0,0,1] ]  
        self.board_copy = []


    def get_legal_moves(self):
        """ Return a list of all legal moves available to the
        active player.  Each player should get a list of all
        empty spaces on the board on their first move, and
        otherwise they should get a list of all open spaces
        in a straight line along any row, column or diagonal
        from their current position. (Players CANNOT move
        through obstacles or blocked squares.) Moves should
        be a pair of integers in (column, row) order specifying
        the zero-indexed coordinates on the board.
        """
        legal_moves = []

        for line_index, line in enumerate(self.board):
            for column_index,column in enumerate(line):
                               
                if self.board[line_index][column_index] == 0:
                    legal_moves.append([line_index,column_index])                
        
        return legal_moves

        

    def forecast_move(self, move):
        """ Return a new board object with the specified move
        applied to the current game state.
        
        Parameters
        ----------
        move: tuple
            The target position for the active player's next move
            (e.g., (0, 0) if the active player will move to the
            top-left corner of the board)
        """
        legal_moves = self.get_legal_moves()
        new_move = list(move)

        if new_move in legal_moves:
            board_copy = copy.deepcopy(self.board) 
            #board_copy[new_move[0]][new_move[1]] = 1
            self.board[new_move[0]][new_move[1]] = 1

        return self
        # TODO: implement this function!
        pass   
        


if __name__ == "__main__":
    # This code is only executed if "gameagent.py" is the run
    # as a script (i.e., it is not run if "gameagent.py" is
    # imported as a module)
    emptyState = GameState()  # create an instance of the object