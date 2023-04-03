# Caroline Smith
# Problem Four: Monte Carlo
# February 13 2023

import random 

def roll_dice(): 

    dieOne = random.randint(1,7)
    dieTwo = random.randint(1,7)

    return dieOne + dieTwo

if __name__ == "__main__": 

    nums = []
    n = input("Enter number of iterations: ")

    for i in range(int(n)):
        nums.append(int(roll_dice()))

    for i in range(2, 13): 
        percentage = (nums.count(i)/int(n))*100
        print(str(i) + ": " + f'{percentage:.2f}' + '%')
