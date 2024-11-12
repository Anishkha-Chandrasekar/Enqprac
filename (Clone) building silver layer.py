# Databricks notebook source
from pyspark.sql.functions import *
from pyspark.sql.types import *

# COMMAND ----------

# MAGIC %md
# MAGIC #### Read all data into dataframes

# COMMAND ----------

# MAGIC %sql
# MAGIC drop database anishkha_c_databricks_npmentorskool_onmicrosoft_com.silver_ecom cascade;

# COMMAND ----------

tables_df = spark.sql("SHOW TABLES IN anishkha_c_databricks_npmentorskool_onmicrosoft_com.enqprac02")
display(tables_df)

for row in tables_df.collect():
    table_name = row['tableName']
    df_name = f"{table_name}_df"
    globals()[df_name] = spark.table(f"anishkha_c_databricks_npmentorskool_onmicrosoft_com.enqprac02.{table_name}")

# COMMAND ----------

customers_df.display()

# COMMAND ----------

# MAGIC %md
# MAGIC ####Create schema - silver layer

# COMMAND ----------

# MAGIC %sql
# MAGIC create schema if not exists anishkha_c_databricks_npmentorskool_onmicrosoft_com.silver_ecom

# COMMAND ----------

# MAGIC %md
# MAGIC #### Address Data Issues

# COMMAND ----------

addresses_df.display()

# COMMAND ----------

file_path = "dbfs:/mnt/anishkha_gcs/addresses.csv"

# Read the CSV file into a Spark DataFrame
df_spark = spark.read.csv(
    file_path,
    header=True,
    inferSchema=True,
    quote='"',
    escape='"',
    multiLine=True
)

# Convert the Spark DataFrame to a Pandas DataFrame
addresses_df_clean = df_spark.toPandas()

# Display the DataFrame
display(addresses_df_clean)

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------

# MAGIC %md 
# MAGIC #### 1. Customers data - Data Quality Checks & Loading into Silver layer

# COMMAND ----------

from pyspark.sql.functions import col, sum

missing_values = customers_df.select([sum(col(c).isNull().cast("int")).alias(c) for c in customers_df.columns])
display(missing_values)

# Check for Duplicates
duplicate_count = customers_df.groupBy(customers_df.columns).count().filter("count > 1").count()
print(f"Number of duplicate rows: {duplicate_count}")

# Summary Statistics
summary_stats = customers_df.describe()
display(summary_stats)

# Check for duplicates in the customer_id column
duplicates = customers_df.groupBy("CustomerID").count().filter("count > 1")

# Display the duplicate customer_ids
display(duplicates)

# Optionally, count the number of duplicate customer_ids
duplicate_count = duplicates.count()
print(f"Number of duplicate customer_ids: {duplicate_count}")


# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE TABLE anishkha_c_databricks_npmentorskool_onmicrosoft_com.silver_ecom.customers (
# MAGIC     CustomerID VARCHAR(50) PRIMARY KEY,
# MAGIC     FirstName VARCHAR(100),
# MAGIC     LastName VARCHAR(100),
# MAGIC     Email VARCHAR(255) NOT NULL,
# MAGIC     PhoneNumber BIGINT,
# MAGIC     DateOfBirth DATE,
# MAGIC     RegistrationDate DATE,
# MAGIC     PreferredPaymentMethodID VARCHAR(50)
# MAGIC )
# MAGIC USING DELTA;

# COMMAND ----------

# Append data from customers_df to the silver.customers table
customers_df.write.mode("append").saveAsTable("anishkha_c_databricks_npmentorskool_onmicrosoft_com.silver_ecom.customers")

# COMMAND ----------

# MAGIC %md 
# MAGIC #### Suppliers & Shipping Tier Data - Data Quality Checks

# COMMAND ----------

from pyspark.sql.functions import col, sum

missing_values = suppliers_df.select([sum(col(c).isNull().cast("int")).alias(c) for c in suppliers_df.columns])
display(missing_values)

# Check for Duplicates
duplicate_count = suppliers_df.groupBy(suppliers_df.columns).count().filter("count > 1").count()
print(f"Number of duplicate rows: {duplicate_count}")

# Summary Statistics
summary_stats = suppliers_df.describe()
display(summary_stats)

# Check for duplicates in the customer_id column
duplicates = suppliers_df.groupBy("SupplierID").count().filter("count > 1")

# Display the duplicate customer_ids
display(duplicates)

# Optionally, count the number of duplicate customer_ids
duplicate_count = duplicates.count()
print(f"Number of duplicate customer_ids: {duplicate_count}")


# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE TABLE anishkha_c_databricks_npmentorskool_onmicrosoft_com.silver_ecom.suppliers (
# MAGIC     SupplierID VARCHAR(50) PRIMARY KEY,
# MAGIC     SupplierName VARCHAR(255),
# MAGIC     SupplierEmail VARCHAR(255),
# MAGIC     City VARCHAR(50)
# MAGIC
# MAGIC )
# MAGIC USING DELTA;
# MAGIC
# MAGIC CREATE TABLE anishkha_c_databricks_npmentorskool_onmicrosoft_com.silver_ecom.shipping_tier (
# MAGIC     ShippingTierID VARCHAR(50) PRIMARY KEY,
# MAGIC     TierName VARCHAR(255),
# MAGIC     Cost_Rs int
# MAGIC )
# MAGIC USING DELTA;

# COMMAND ----------

# Append data from vendors_df to the silver.vendor table
suppliers_df.write.mode("append").saveAsTable("anishkha_c_databricks_npmentorskool_onmicrosoft_com.silver_ecom.suppliers")

# COMMAND ----------

# Append data from silver.shipping_methods to the silver.shipping_methods table
shipping_tier_df.write.mode("append").option("mergeSchema","true").saveAsTable("anishkha_c_databricks_npmentorskool_onmicrosoft_com.silver_ecom.shipping_tier")

# COMMAND ----------

# MAGIC %md
# MAGIC #### Orders Data - Data Quality Check

# COMMAND ----------

missing_values = orders_df.select([sum(col(c).isNull().cast("int")).alias(c) for c in orders_df.columns])
display(missing_values)

# Check for Duplicates
duplicate_count = orders_df.groupBy(orders_df.columns).count().filter("count > 1").count()
print(f"Number of duplicate rows: {duplicate_count}")

# Summary Statistics
summary_stats = orders_df.describe()
display(summary_stats)

# Check for duplicates in the OrderID column
duplicates = orders_df.groupBy("OrderID").count().filter("count > 1")

# Display the duplicate OrderID
display(duplicates)

# Optionally, count the number of duplicate OrderID
duplicate_count = duplicates.count()
print(f"Number of duplicate OrderID: {duplicate_count}")

# COMMAND ----------

# integrity check

# Find CustomerID in orders_df that are not in customers_df using left anti join
non_matching_customer_ids = orders_df.join(customers_df, orders_df.CustomerID == customers_df.CustomerID, "left_anti")

# Display the result
display(non_matching_customer_ids)

# COMMAND ----------

# MAGIC %sql
# MAGIC
# MAGIC CREATE TABLE anishkha_c_databricks_npmentorskool_onmicrosoft_com.silver_ecom.orders (
# MAGIC     OrderID VARCHAR(50) PRIMARY KEY,
# MAGIC     CustomerID VARCHAR(50),
# MAGIC     OrderDate TIMESTAMP NOT NULL,
# MAGIC     ShippingDate TIMESTAMP,
# MAGIC     ExpectedDeliveryDate TIMESTAMP,
# MAGIC     ActualDeliveryDate TIMESTAMP,
# MAGIC     ShippingTierID VARCHAR(50),
# MAGIC     SupplierID VARCHAR(50),
# MAGIC     FOREIGN KEY (CustomerID) REFERENCES anishkha_c_databricks_npmentorskool_onmicrosoft_com.silver_ecom.customers(CustomerID),
# MAGIC     FOREIGN KEY (SupplierID) REFERENCES anishkha_c_databricks_npmentorskool_onmicrosoft_com.silver_ecom.suppliers(SupplierID),
# MAGIC     FOREIGN KEY (ShippingTierID) REFERENCES anishkha_c_databricks_npmentorskool_onmicrosoft_com.silver_ecom.shipping_tier(ShippingTierID)
# MAGIC )
# MAGIC USING DELTA;

# COMMAND ----------

# Cast the date columns to TimestampType
orders_df = orders_df.withColumn("OrderDate", orders_df["OrderDate"].cast(TimestampType()))
orders_df = orders_df.withColumn("ShippingDate", orders_df["ShippingDate"].cast(TimestampType()))
orders_df = orders_df.withColumn("ExpectedDeliveryDate", orders_df["ExpectedDeliveryDate"].cast(TimestampType()))
orders_df = orders_df.withColumn("ActualDeliveryDate", orders_df["ActualDeliveryDate"].cast(TimestampType()))

# Append data to the silver_ecom.orders table
orders_df.write.mode("append").option("mergeSchema", "true").saveAsTable("anishkha_c_databricks_npmentorskool_onmicrosoft_com.silver_ecom.orders")

# COMMAND ----------

# MAGIC %md
# MAGIC #### Products Data - Data Quality Checks

# COMMAND ----------

from pyspark.sql.functions import col, sum
missing_values = products_df.select([sum(col(c).isNull().cast("int")).alias(c) for c in products_df.columns])
display(missing_values)

# Check for Duplicates
duplicate_count = products_df.groupBy(products_df.columns).count().filter("count > 1").count()
print(f"Number of duplicate rows: {duplicate_count}")

# Summary Statistics
summary_stats = products_df.describe()
display(summary_stats)

# COMMAND ----------

# Get data types of all columns of products_df
data_types = products_df.dtypes
display(data_types)

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE TABLE anishkha_c_databricks_npmentorskool_onmicrosoft_com.silver_ecom.products (
# MAGIC     Product_ID VARCHAR(50) PRIMARY KEY,
# MAGIC     Product_Name VARCHAR(255) NOT NULL,
# MAGIC     Product_Category VARCHAR(255),
# MAGIC     Product_Sub_Category VARCHAR(100),
# MAGIC     Product_Rating DOUBLE,
# MAGIC     Number_of_product_ratings DOUBLE,
# MAGIC     Discounted_Price int,
# MAGIC     Actual_Price int
# MAGIC );

# COMMAND ----------

from pyspark.sql.functions import regexp_replace, col

# Cast columns to appropriate data types
products_df = products_df.withColumn("Product_Rating", products_df["Product_Rating"].cast("double"))
products_df = products_df.withColumn("Number_of_product_ratings", products_df["Number_of_product_ratings"].cast("double"))
products_df = products_df.withColumn("Actual_Price", regexp_replace(col("Actual_Price"), "[₹]", "")) \
               .withColumn("Actual_Price", regexp_replace(col("Actual_Price"), ",", "").cast("int"))
products_df = products_df.withColumn("Discounted_Price", regexp_replace(col("Discounted_Price"), "[₹]", "")) \
               .withColumn("Discounted_Price", regexp_replace(col("Discounted_Price"), ",", "").cast("int"))
products_df.display()

# COMMAND ----------

# Append data to the silver.products table with schema merging
products_df.write.mode("append").option("mergeSchema", "true").saveAsTable("anishkha_c_databricks_npmentorskool_onmicrosoft_com.silver_ecom.products")

# COMMAND ----------

# MAGIC %md
# MAGIC #### Payment Methods Data - Data Quality Checks

# COMMAND ----------

# MAGIC %sql
# MAGIC
# MAGIC CREATE TABLE anishkha_c_databricks_npmentorskool_onmicrosoft_com.silver_ecom.payment_methods (
# MAGIC     PaymentMethodID VARCHAR(50) PRIMARY KEY,
# MAGIC     MethodName VARCHAR(255)
# MAGIC )
# MAGIC USING DELTA;

# COMMAND ----------

# Append data from payment_methods_df to the silver.payment_methods table
payment_methods_df.write.mode("append").saveAsTable("anishkha_c_databricks_npmentorskool_onmicrosoft_com.silver_ecom.payment_methods")

# COMMAND ----------

# MAGIC %md
# MAGIC #### Orders Items Data - Data Quality Checks

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE TABLE anishkha_c_databricks_npmentorskool_onmicrosoft_com.silver_ecom.order_items (
# MAGIC     OrderItemID VARCHAR(50) PRIMARY KEY,
# MAGIC     OrderID VARCHAR(50),
# MAGIC     ProductID VARCHAR(50),
# MAGIC     Quantity INT,
# MAGIC     FOREIGN KEY (OrderID) REFERENCES anishkha_c_databricks_npmentorskool_onmicrosoft_com.silver_ecom.orders(OrderID),
# MAGIC     FOREIGN KEY (ProductID) REFERENCES anishkha_c_databricks_npmentorskool_onmicrosoft_com.silver_ecom.products(Product_ID)
# MAGIC )
# MAGIC USING DELTA;

# COMMAND ----------

orders_items_df.write.mode("append").saveAsTable("anishkha_c_databricks_npmentorskool_onmicrosoft_com.silver_ecom.order_items")

# COMMAND ----------

# MAGIC %md
# MAGIC #### Payments Data - Data Quality Checks

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE TABLE anishkha_c_databricks_npmentorskool_onmicrosoft_com.silver_ecom.payments (
# MAGIC     PaymentID VARCHAR(50) PRIMARY KEY,
# MAGIC     OrderID VARCHAR(50),
# MAGIC     PaymentDate TIMESTAMP,
# MAGIC     GiftCardUsage VARCHAR(30),
# MAGIC     GiftCardAmount DOUBLE,
# MAGIC     CouponUsage VARCHAR(30),
# MAGIC     CouponAmount DOUBLE,
# MAGIC     PaymentMethodID VARCHAR(50),
# MAGIC     FOREIGN KEY (OrderID) REFERENCES anishkha_c_databricks_npmentorskool_onmicrosoft_com.silver_ecom.orders(OrderID)
# MAGIC )
# MAGIC USING DELTA;

# COMMAND ----------

payments_df = payments_df.withColumn("PaymentDate", payments_df["PaymentDate"].cast(TimestampType()))

# COMMAND ----------

payments_df.write.mode("append").saveAsTable("anishkha_c_databricks_npmentorskool_onmicrosoft_com.silver_ecom.payments")

# COMMAND ----------

# MAGIC %md
# MAGIC #### Returns Data - Data Quality Checks

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE TABLE anishkha_c_databricks_npmentorskool_onmicrosoft_com.silver_ecom.returns (
# MAGIC     OrderID VARCHAR(50),
# MAGIC     Return_reason VARCHAR(500),
# MAGIC     FOREIGN KEY (OrderID) REFERENCES anishkha_c_databricks_npmentorskool_onmicrosoft_com.silver_ecom.orders(OrderID)
# MAGIC )
# MAGIC USING DELTA;

# COMMAND ----------

returns_df.write.mode("append").saveAsTable("anishkha_c_databricks_npmentorskool_onmicrosoft_com.silver_ecom.returns")


# COMMAND ----------

# MAGIC %md
# MAGIC #### Address Data - Data Quality Checks

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE TABLE anishkha_c_databricks_npmentorskool_onmicrosoft_com.silver_ecom.addresses (
# MAGIC     AddressID VARCHAR(255) PRIMARY KEY,
# MAGIC     CustomerID VARCHAR(255),
# MAGIC     AddressLine1 VARCHAR(500),
# MAGIC     City VARCHAR(255),
# MAGIC     State VARCHAR(255),
# MAGIC     PinCode FLOAT,
# MAGIC     AddressType VARCHAR(255),
# MAGIC     FOREIGN KEY (CustomerID) REFERENCES anishkha_c_databricks_npmentorskool_onmicrosoft_com.silver_ecom.customers(CustomerID)
# MAGIC )
# MAGIC USING DELTA;

# COMMAND ----------

# Convert 'PinCode' field to the correct data type if needed
addresses_df = addresses_df.withColumn("PinCode", col("PinCode").cast("float"))

# Save the DataFrame as a table
addresses_df.write.mode("append").option("mergeSchema", "true").saveAsTable("anishkha_c_databricks_npmentorskool_onmicrosoft_com.silver_ecom.addresses")

# COMMAND ----------

# MAGIC %sql
# MAGIC show tables in anishkha_c_databricks_npmentorskool_onmicrosoft_com.silver_ecom;
