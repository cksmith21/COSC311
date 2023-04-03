# Caroline Smithh
# Problem Two: Decision Tree
# February 13 2023

def play_tennis(outlook, humidity, wind): 

    if outlook == "sunny": 
        if humidity == "high":
            print("Do not play tennis.")
        elif humidity == "normal":
            print("Play tennis.")
        else:
            pass
    elif outlook == "rain":
        if wind == "strong":
            print("Do not play tennis.")
        elif wind == "weak":
            print("Play tennis.")
        else: 
            pass
    elif outlook == "overcast":
        print("Play tennis.")
    else: 
        pass

if __name__ == "__main__":

    outlook = input("What is the outlook? Enter 'Sunny', 'Rain', 'Overcast': ")

    humidity = input("What is the humidity? Enter 'High' or 'Normal': ")

    wind = input("What is the wind? Enter 'Strong' or 'Weak': ") 

    play_tennis(outlook.lower(), humidity.lower(), wind.lower())