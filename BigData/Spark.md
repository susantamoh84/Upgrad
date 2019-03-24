# Overview Of Spark

  - Apache Spark is an open source cluster-computing framework. It is known mainly for its high-speed cluster computing capabilities 
    and support for programming languages such as Python, R, Java, SQL, Scala etc. It is fast, scalable and easy to use.
    
  - main features of Apache Spark. Let's recap them:

    - Spark outperforms MapReduce by a large margin, especially when the scale of data reaches more than a few hundreds of gigabytes.
    - Spark is an open source project, and thus has active contributions from all around the world. 
      You can contribute to the Spark project too.

    - Spark can be accessed using any of the standard languages in a data scientist's toolkit (R, SQL, Python)
    - Spark has the capability to connect to a variety of data sources like S3, Cassandra, HDFS, HBase, Avro, Parquet, etc.
    
# RDD:

  - Following are the main properties of RDDs:

  - Distributed collection of data: RDDs exist in a distributed form, over different worker nodes. 
    This enables the storage of large sizes of data. The Driver node is responsible for creating this distribution as well as tracking it.

  - Fault tolerance: This refers to the property of RDDs to regenerate if they are lost during computation.

  - Parallel operations: While the RDDs exist as distributed files across worker nodes, their processing also happens in parallel.

  - Ability to use varied data sources: RDDs are not dependent on any specific structure of an input data source. 
    They are adaptive and can be built from different sources.
    
# Accessing RDD in a tabular form: The Spark DataFrame

  - Some key differences between RDDs and DataFrames are listed here:

    - Structure: An RDD is a distributed collection of data spread across many machines in a cluster. 
      A DataFrame is also distributed but consists of named columns. 
      It is conceptually much closer to a table in a relational database.

    - Formats: RDDs need the user to explicitly specify the schemas of the data, where DataFrames allow Spark to manage and 
      create schemas.

    - Applications: RDDs are more commonly used while working with unstructured data, whereas while working with structured data 
      (such as RDBMS, CSV files etc.), DataFrames are much more convenient.
      
# RDD Operations

  - RDD Operations are of two types - transformations and actions
  - Until an action is called, no transformation is run. This property is called lazy evaluation and forms the core of Spark.
  - Any RDD can be recovered by its lineage. This lineage is basically a graphical representation (in the form of a DAG) of the 
    transformation history leading up to the specific RDD
    
  - Like most Hadoop based systems, Spark also uses the schema-on-read model.


