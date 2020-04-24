frequency_dict = {}

for i in range(500):
    print("working on " + str(i))
    unique = np.unique(next(iter(train_loader))['mask'].numpy())
    print(unique)
    for element in unique:
        if element in frequency_dict:
            frequency_dict[element]+=1
        else:
            frequency_dict.update({element : 1})

for key in frequency_dict:
    print(str(key) + ": " + str(frequency_dict[key]))

assert False, 'Hold up wait a minute you thought I was finished'