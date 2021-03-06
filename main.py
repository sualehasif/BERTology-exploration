import os
from file_parser import parse_reviews


if __name__ == "__main__": 
    print("Starting the task")

    review_file_path = str(os.getcwd()) + "/restaurant_reviews.txt"
    review_sentences = parse_reviews(review_file_path)

    print(review_sentences[:15])
    