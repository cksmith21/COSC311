# Caroline Smith
# Problem One: Triangular Number Sequence 
# February 13 2023

def find_triangular_numbers():

    number = []
    evenSum, oddSum, temp = 0, 0, 0

    # main loop

    for i in range(1, 20): 
        temp = int(1/2*i*(i+1))
        number.append(temp)

        if(temp % 2 == 0):
            evenSum += int(temp)
        else: 
            oddSum += int(temp)

    #print(*list) prints out a list with spaces separating eachh value
    print(*number)
    print("Odd sum: " + str(oddSum))
    print("Even sum: " + str(evenSum))

if __name__ == "__main__": 
    
    find_triangular_numbers()