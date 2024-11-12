# Databricks notebook source
from pyspark.sql.functions import *

# COMMAND ----------

products_path = 'dbfs:/mnt/anishkha_gcs/products.csv'
products_df = spark.read.option("header","true").option("inferSchema","true").csv(products_path)
products_df.display()

# COMMAND ----------

orders_items_path = 'dbfs:/mnt/anishkha_gcs/orders_items.csv'
orders_items_df = spark.read.option("header","true").option("inferSchema","true").csv(orders_items_path)
orders_items_df.display()

# COMMAND ----------

products_df.createGlobalTempView("products")

# COMMAND ----------

orders_items_df.createGlobalTempView("orders_items")

# COMMAND ----------

orders_path = 'dbfs:/mnt/anishkha_gcs/orders.csv'
orders_df = spark.read.option("header","true").option("inferSchema","true").csv(orders_path)
orders_df.display()

# COMMAND ----------

shipping_tier_path = 'dbfs:/mnt/anishkha_gcs/shipping_tier.csv'
shipping_tier_df = spark.read.option("header","true").option("inferSchema","true").csv(shipping_tier_path)
shipping_tier_df.display()

# COMMAND ----------

orders_df.createGlobalTempView("orders")

# COMMAND ----------

shipping_tier_df.createGlobalTempView("shipping_tier")

# COMMAND ----------

customers_path = 'dbfs:/mnt/anishkha_gcs/customers.csv'
customers_df = spark.read.option("header","true").option("inferSchema","true").csv(customers_path)
customers_df.display()

# COMMAND ----------

customers_df.createGlobalTempView("customers")

# COMMAND ----------

returns_path = 'dbfs:/mnt/anishkha_gcs/returns.csv'
returns_df = spark.read.option("header","true").option("inferSchema","true").csv(returns_path)
returns_df.display()

# COMMAND ----------

returns_df.createGlobalTempView("returns")

# COMMAND ----------

suppliers_path = 'dbfs:/mnt/anishkha_gcs/suppliers.csv'
suppliers_df = spark.read.option("header","true").option("inferSchema","true").csv(suppliers_path)
suppliers_df.display()

# COMMAND ----------

suppliers_df.createGlobalTempView("suppliers")

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE GLOBAL TEMP VIEW supplier_product_count AS
# MAGIC SELECT 
# MAGIC     s.SupplierID, 
# MAGIC     s.SupplierName,
# MAGIC     s.SupplierEmail,
# MAGIC     s.City,
# MAGIC     COUNT(p.Product_ID) AS NumberOfProducts
# MAGIC FROM 
# MAGIC     global_temp.suppliers s
# MAGIC JOIN 
# MAGIC     global_temp.orders o ON o.SupplierID = s.SupplierID
# MAGIC JOIN
# MAGIC     global_temp.orders_items oi ON oi.OrderID = o.OrderID
# MAGIC JOIN 
# MAGIC     global_temp.products p ON oi.ProductID = p.Product_ID
# MAGIC GROUP BY 
# MAGIC     s.SupplierID, s.SupplierName, s.SupplierEmail, s.City

# COMMAND ----------

from pyspark.sql import functions as F

avg_ratings_df = products_df.withColumn("Product_Rating", F.col("Product_Rating").cast("float")) \
    .groupBy("Product_ID") \
    .agg(F.round(F.avg("Product_Rating"), 1).alias("AvgRating"))

highly_rated_products_df = avg_ratings_df.filter(F.col("AvgRating") >= 4.5)


display(highly_rated_products_df)

# COMMAND ----------

from pyspark.sql.functions import *

avg_ratings_df = products_df.withColumn("Product_Rating", col("Product_Rating").cast("float")) \
    .groupBy("Product_ID") \
    .agg(round(avg("Product_Rating"), 1).alias("AvgRating"))

highly_rated_products_df = avg_ratings_df.filter(col("AvgRating") >= 4.5)


display(highly_rated_products_df)

# COMMAND ----------

from pyspark.sql.functions import datediff

orders_df = orders_df.withColumn("Days_Between", datediff("ShippingDate", "OrderDate"))

display(orders_df)

# COMMAND ----------

cte1 = (
    orders_items_df
    .join(products_df, orders_items_df.ProductID == products_df.Product_ID)
    .withColumn("sales", col("Quantity")*regexp_replace(col("Discounted_Price"),'[₹,]',''))
    .select("OrderID",orders_items_df.ProductID,"sales")
)
cte2 = (
    cte1
    .join(orders_df, "OrderID")
    .withColumn("year",year("OrderDate"))
    .withColumn("month",month("OrderDate"))
    .select("year","month","sales")
    .groupBy("year","month")
    .agg(sum("sales").alias("sales"))
    .orderBy("year","month")
)
cte3 =(
    cte2
    .withColumn("prevMonthSales",lag("sales").over(Window.partitionBy("year").orderBy("month")))
    .withColumn("mom", 100*(col("sales")-col("prevMonthSales"))/col("prevMonthSales"))
)
cte3.display()

# COMMAND ----------

shipping_tier_path = 'dbfs:/mnt/anishkha_gcs/shipping_tier.csv'
shipping_tier_df = spark.read.option("header","true").option("inferSchema","true").csv(shipping_tier_path)
shipping_tier_df.display()

# COMMAND ----------

from pyspark.sql.functions import col, countDistinct, sum, count, regexp_replace

returns_df = returns_df.withColumnRenamed("OrderId", "Returns_OrderId")

report_df = (
    orders_items_df
    .join(products_df, orders_items_df.ProductID == products_df.Product_ID)
    .join(returns_df, orders_items_df.OrderID == returns_df.Returns_OrderId, "left_outer")
    .withColumn("revenue", col("Quantity") * regexp_replace(col("Discounted_Price"), '[₹,]', '').cast("float"))
    .groupBy("ProductID", "Product_Name")
    .agg(
        countDistinct("OrderID").alias("Total_Orders"),
        sum("Quantity").alias("Total_Units_Sold"),
        sum("revenue").alias("Total_Revenue"),
        (sum("revenue") / sum("Quantity")).alias("Avg_Price"),
        count("Returns_OrderId").alias("Total_Returns"),
        (count("Returns_OrderId") / countDistinct("OrderID")).alias("Return_rate")
    )
)

report_df.display()
