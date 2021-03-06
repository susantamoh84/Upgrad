# Use-cases of Hive

  - Cleaning data and converting between data formats such as JSON to tabular etc.
  - Slicing, dicing and joining tables, and publishing reports 
  - Building data applications such as recommender systems
  
  - There are four main components in Hive data models, which are similar to how an RDBMS stores data:
    - Databases
    - Tables
    - Partitions
    - Buckets
    
  - Hive has two types of tables:
    - Managed (or internal) table
    - External table
    
  - You should use external tables when —
  
    - You want to use the data outside of Hive as well. For example, when another existing program is running on the same cluster
    - You want the data to remain stored on the HDFS even after dropping tables because Hive does not delete the data stored outside 
      (of the Hive database).
    - You do not want Hive to control the storage of your data (location/directories of storage/etc.).

  - On the other hand, you use managed tables when —

    - The data is temporary. So, when the Hive table is dropped, the data stored in the internal table is deleted too.
    - You want Hive to manage the life cycle of the data completely, i.e. both store and process it.
    
  - the concept of SerDes, or serialisers and deserialisers. 
  - SerDes basically enable Hive to read and write data to and from various file formats such as ORC.
  

# Partitions and Bucketing

  - To summarise, partitioning is a way to segregate data into multiple directories, 
    such that while querying, Hive can access only the relevant directories, and thus boost querying speed. 
 
  - However, it's not a good practice to create a very large number of directories (e.g. by partitioning on customer ID), 
    since that will increase the amount of data the metastore has to store and may rather degrade the performance.
    
    General Instruction:

  - Before creating any table and partitioning any table, make sure you run these commands. 
    These commands are prerequisites for running the code in the lab without error:


    - ADD JAR /opt/cloudera/parcels/CDH/lib/hive/lib/hive-hcatalog-core-1.1.0-cdh5.11.2.jar;
    - SET hive.exec.max.dynamic.partitions=100000;
    - SET hive.exec.max.dynamic.partitions.pernode=100000;
    
  - Bucketing (also called clustering) distributes the data uniformly in various buckets. 
    This is fundamentally different from partitioning. 
    
  - To compare the two techniques, consider the following example: 

    - You typically perform partitioning on columns (or groups of columns) such that it produces a 
      manageable number of partitions (directories), such as year, month, state, country etc.
    - However, if you partition on say customer ID, the number of partitions will be enormous and may actually reduce 
      performance; thus, you can choose to create buckets on customer ID
      
    - It is important to note that bucketing uses the hash of a column to create multiple 'buckets', whereas partitions 
      are more straightforward - they are simply segmented directories according to the value of a column (year, month etc.)
      

# FileFormats for Apache Hive

  - ORC (Optimized Row Columnar) is an extremely commonly used file format used for faster query performance.
  - Each stripe belonging to an ORC file comprises of "Index Data", "Row Data", and a "Stripe Footer"
    - Index Data: It includes min and max values for each column and the row positions within each column, "Row Data", and a "Stripe Footer
    - Stripe Footer: It includes encoding of each column and the directory of the streams including their location
    - Row data: It is the actual content of the data
    
  - Other popular file formats are Parquets, Avro, RC etc.
  
  
# Advanced Data Analysis

  - Hive CheetSheets: https://martin.atlassian.net/wiki/spaces/lestermartin/pages/19431488/Hive+Cheat+Sheet
  - Hive UDFs:
    - correlation between 2 columns
  - Basic Text Analysis:
    - A Brief Introduction to N-grams:
      - The concept of n-grams is a very basic form of language modelling. An n-gram is basically a continuous series of n words in a given sample of text or speech.
      - An n-gram model is an essential tool in language processing, which can be used for sentiment analysis, character relationship analysis, etc. Also, n-grams can be used to conduct  'sentiment analysis' of a review in the Amazon dataset.
      
    - Code:
    --IMPORTANT: BEFORE CREATING ANY TABLE, MAKE SURE YOU RUN THIS COMMAND 
      ADD JAR /opt/cloudera/parcels/CDH/lib/hive/lib/hive-hcatalog-core-1.1.0-cdh5.11.2.jar;

      SELECT explode( ngrams( sentences( lower(reviewtext) ), 2, 6))
      FROM amazon_reviews_year_month_partition_orc
      where yr = 2013 and mnth = 1;
      
# Complex Data-types using Hive

  - Primitive data types are simple data types and they are the basic building blocks of any programming language. 
    For example, integer, string, etc.
  - Complex data types are used to store complex numbers and complex arithmetic structures such as struct, matrix, and arrays.
  
