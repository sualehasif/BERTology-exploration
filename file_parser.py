def parse_reviews(file_path):
    sentence_list = []
    cur_sentence = ""
    with open(file_path) as f:
        for line in f:
            line = line.split()
            if line:
                word = line[0]
                if word.isalpha():
                    cur_sentence += word
                    cur_sentence += " "
                else:
                    cur_sentence = cur_sentence[:-1]
                    cur_sentence += word
                    cur_sentence += " "
            else:
                sentence_list.append(cur_sentence[:-1])
                cur_sentence = ""
    
    return sentence_list