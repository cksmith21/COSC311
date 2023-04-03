# Caroline Smith
# Problem Three: Octagon
# February 13 2023

def octagon(sideLength):
    
    for i in range(sideLength-1):
        print(" " * (sideLength - 1 - i) + "*"*(sideLength+i*2))
    
    for i in range(sideLength):
        print("*"*int((sideLength + (sideLength-1)*2)))
    
    counter = 1
    for i in range((sideLength + (sideLength-1)*2)-2, sideLength-1, -2):
        
        print(" "*counter + "*"*i)
        counter+=1


if __name__ == "__main__": 

    sideLength = input("Enter a side length that is greater than or equal to 2: ")

    while(int(sideLength) < 2):

        print("Enter a side length that is greater than or equal to 2.")

        sideLength = input()

    octagon(int(sideLength))
