def won(board):

    if  (board[0][0] == board[0][1] == board[0][2] and board[0][0] != " ") or \
        (board[1][0] == board[1][1] == board[1][2] and board[1][0] != " ") or \
        (board[2][0] == board[2][1] == board[2][2] and board[2][0] != " ") or \
        (board[0][0] == board[1][0] == board[2][0] and board[0][0] != " ") or \
        (board[0][1] == board[1][1] == board[2][1] and board[0][1] != " ") or \
        (board[0][2] == board[1][2] == board[2][2] and board[0][2] != " ") or \
        (board[0][0] == board[1][1] == board[2][2] and board[0][0] != " ") or \
        (board[0][2] == board[1][1] == board[2][0] and board[0][2] != " "):
        return True
    else: 
        return False

def draw(board):
    for i in range(3):
        for j in range(3):
            if board[i][j] == " ":
                return False
    return True

def check_input(board, row, col):

    if row > -1 and row <  3 and col > -1 and col < 3:
        if board[row][col] == " ":
            return True
    return False

def print_board(board):

    print()
    print("     0   1   2    ")
    print(f'0  | {board[0][0]} | {board[0][1]} | {board[0][2]} |')
    print("   -------------")
    print(f'1  | {board[1][0]} | {board[1][1]} | {board[1][2]} |')
    print("   -------------")
    print(f'2  | {board[2][0]} | {board[2][1]} | {board[2][2]} |')
    print()

if __name__ == "__main__": 

    game_board = [[' ',' ',' '],[' ',' ',' '],[' ',' ',' ']]

    players = {0:'x', 1:'o'}

    turn = 0

    while turn != -1: 

        print_board(game_board)
        print(f'Player {turn}, input row and column in the format row space col: ')

        row, col = input().split()

        while not check_input(game_board, int(row), int(col)):
            print("Invalid move, try again: ")
            row, col = input().split()

        game_board[int(row)][int(col)] = players[turn]

        if won(game_board) and not draw(game_board):
            print(f"Player {turn} won!")
            turn = -1
            print_board(game_board)
        elif draw(game_board):
            print("Game ended in draw.")
            turn = -1
            print_board(game_board)
        else: 
            turn = (turn+1)%2



    
        
        
        




