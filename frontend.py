import csv
from bavourai.client import BavourClient


bavour = BavourClient("llama3.2")


def read_customer_reviews(csv_filepath: str)
    """
    Reads customer review data from a CSV file and returns a list of validated CustomerReview objects.
    """
    reviews = []

    with open(csv_filepath, mode='r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            try:
                review = f"This product {row['Product']} received this rating {row['Rating']} from a customer because of {row['Customer Comment']}"
                review_dict = {
                    "role": "user",
                    "content": review,
                    "metadata": {
                        "product": row['Product'],
                        "rating": row['Rating'],
                    }
                }
                reviews.append(review_dict)
            except Exception as e:
                # print(f"Skipping invalid row: {row} due to error: {e}")
                continue
    
    return reviews

# USUAGE
# Read the CSV file
reviews_list = read_customer_reviews("customer_reviews.csv")

# Add the reviews to the Bavour database(ChromaDB for now)
# bavour.add_data(prompt=reviews_list)

# Query search optimized for searching based on rating
response = bavour.search(
    query="Find a product with rating equal to 3", 
    total_expected_result=1
)
print({"Final Response": response})