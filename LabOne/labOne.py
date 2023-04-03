import matplotlib.pyplot as plt

if __name__ == "__main__":


    word_dict = {}

    with open('SciencePaper.txt', 'r') as paper:

        for line in paper:

            # removes all punctuation
            words = line.upper().replace(',','').replace(';','').replace('(','').replace(')','').replace('!','').replace('?','').replace('.','').replace('-','').replace('"','').replace("'",'').split()

            # creates a dictionary with the word as the key and its apperances as the value
            for word in words:

                try: 
                    word_dict[word] += 1
                except:
                    word_dict[word] = 1

    # print the number of unique words
    num_unique_words = len(word_dict)
    print(f'The number of unique words in the dictionary is {num_unique_words}.')

    # find the top 10 words by appearance

    sorted_vals = sorted(word_dict.values(),reverse=True)
    sorted_dict = {}

    for val in sorted_vals:
        for key in word_dict.keys():
            if word_dict[key] == val:
                    sorted_dict[key] = word_dict[key]

    top_ten = {k: sorted_dict[k] for k in list(sorted_dict)[:10]}
    
    print("The top ten word appearances are: ")
    print(top_ten)

    # find the frequencies of Summerfelt, wastewater, greenhouse, salmon

    word_list = ['Summerfelt', 'wastewater', 'greenhouse', 'salmon']

    freq = [(word.upper(), word_dict[word.upper()]) for word in word_list if word.upper() in word_dict]

    print("Frequencies for 'Summerfelt', 'wastewater', 'greenhouse', and 'salmon': " )

    for item in freq:
      print(item[0] + ": " + str(item[1]))
    
    # find the words that appear 1 time, 2 times, 5 times, and 10 times

    one_app, two_app, five_app, ten_app = [], [], [], []

    for word in word_dict.keys(): 
        if word_dict[word] == 1:
            one_app.append(word) 
        elif word_dict[word] == 2:
            two_app.append(word)
        elif word_dict[word] == 5:
            five_app.append(word)
        elif word_dict[word] == 10:
            ten_app.append(word)
        else:
            pass
        

    print('The words that appear 1 time are: ' + ' '.join(one_app) + '\n') 
    print('The words that appear 2 times are: ' + ' '.join(two_app) + '\n')
    print('The words that appear 5 times are: ' + ' '.join(five_app) + '\n')
    print('The words that appear 10 times are: ' + ' '.join(ten_app) + '\n')

    new_word_dict = {}
    for word,count in word_dict.items():
        try:
            new_word_dict[count].append(word)
        except:
            new_word_dict[count] = [word]

    appearances = list(new_word_dict.keys())
    avg_len = [sum([len(word) for word in value]) / len(value) for value in new_word_dict.values()]

    plt.bar(appearances, avg_len)  
    plt.xlabel("Appearances")
    plt.ylabel("Average Length")