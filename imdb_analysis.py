
# IMDb Movie Review Analysis using PySpark

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, length, trim, lower
import csv
import os


# Create Spark Session
spark = SparkSession.builder \
    .appName("IMDb Movie Review Analysis") \
    .master("local[*]") \
    .getOrCreate()

#1 Read the CSV dataset into a DataFrame
df = spark.read \
    .option("header", True) \
    .option("inferSchema", False) \
    .option("quote", '"') \
    .option("escape", '"') \
    .option("mode", "PERMISSIVE") \
    .csv("IMDB Dataset.csv")

print("\nDataset loaded successfully!")

#2 Explore the data

# Display schema
print("\nDataset Schema:")
df.printSchema()

# Display first 5 rows
print("\nFirst 5 Rows:")
df.show(5, truncate=False)

#3 Clean the Data

# Count records before cleaning
before_count = df.count()

print("Rows before cleaning:", before_count)

# Remove NULL and empty reviews
cleaned_df = df.filter(
    col("review").isNotNull() &
    (length(trim(col("review"))) > 0) &
    col("sentiment").isNotNull() &
    (length(trim(col("sentiment"))) > 0)
)

# Count records after cleaning
after_count = cleaned_df.count()

print("Rows after cleaning:", after_count)
print("Rows removed:", before_count - after_count)

#4 Analyze the Data

# Count total reviews
total_reviews = cleaned_df.count()

print("Total number of reviews:", total_reviews)

# Count positive reviews
positive_reviews = cleaned_df.filter(
    lower(trim(col("sentiment"))) == "positive"
).count()

# Count negative reviews
negative_reviews = cleaned_df.filter(
    lower(trim(col("sentiment"))) == "negative"
).count()

print("Positive reviews:", positive_reviews)
print("Negative reviews:", negative_reviews)

#5 Transform the Data

# Create a new column that stores the length of each review
transformed_df = cleaned_df.withColumn(
    "review_length",
    length(col("review"))
)

print("Column 'review_length' created successfully!")

# Filter reviews longer than 500 characters
long_reviews = transformed_df.filter(
    col("review_length") > 500
)

print("\nReviews longer than 500 characters:")

long_reviews.select(
    "review",
    "sentiment",
    "review_length"
).show()

# Count long reviews
long_review_count = long_reviews.count()

print("Number of reviews longer than 500 characters:",
      long_review_count)

#6 Save the Results

os.makedirs("output", exist_ok=True)

output_file = "output/cleaned_imdb_reviews.csv"

with open(output_file, "w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)

    writer.writerow(["review", "sentiment", "review_length"])

    for row in transformed_df.select(
        "review", "sentiment", "review_length"
    ).toLocalIterator():

        writer.writerow([
            row["review"],
            row["sentiment"],
            row["review_length"]
        ])

print("Dataset saved successfully!")
print("Output location:", os.path.abspath(output_file))
print("Total rows saved:", after_count)

# Stop Spark Session
spark.stop()
