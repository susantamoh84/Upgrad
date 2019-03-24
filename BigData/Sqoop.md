# Data Ingestion Using sqoop

  - Data ingestion is necessary since traditional systems such as MySQL databases, Teradata, Netezza, Oracle, etc. often become incapable 
    of handling growing volumes of data, and you need to ingest the data from these systems into big data systems.
    
  - Sqoop is used to ingest structured data
  - Flume is a tool used for semi-structured or unstructured data.
  
  - the steps involved in an import command are:

    - User runs sqoop command through the shell
    - Sqoop collects metadata from the RDBMS (e.g. MySQL)
    - Sqoop launches MapReduce job (the Sqoop command will automatically be converted into the Java nature of a MapReduce job, 
      using the metadata information)
    - Mappers are created against primary key ranges of the RDBMS, and the data is written into the Hadoop ecosystem

  - To summarise, Sqoop follows the following procedure for data ingestion:

    - It looks at the range of the primary key (from the splitting/primary key column).     
    - It sets the lower value of the primary key to some variable.   
    - It sets the higher value of the primary key to another variable.
    - It generates SQL queries to fetch the data parallelly.
    
  - If the values of the primary key column/splitting column are not uniformly distributed across its range, 
    this can result in unbalanced tasks. In such cases, you are advised to choose a different column using 
    the '--split-by' clause explicitly.
    
  - The session covered the following commands:

    - list-databases: This command is used to list the databases available on the RDS.
    - list-tables: This command lists all the tables of a database mentioned in the command.
    - import: This command will transfer a table from the RDBMS (the RDS, in our case) to the HDFS.
    - import-all-tables: This command will transfer all the tables from a database to the HDFS. 
    - job: This command allows you to save and execute repeatedly used commands.
    - eval: This command will help you execute SQL queries.
 
  - Remember the options you used during the lab session. The important ones are —

    - --connect: This will help you connect to the intended RDS/RDBMS.  Remember the port in your case is 3306 as you're using a 
      MySQL database. 
    - --username: This will specify the username that you want to use to connect to the database.
    - -P: This is always recommended as a prompt for the password.
    - --warehouse-directory: This will directly place the imported tables inside the warehouse directory, which is easily 
      accessible to Hive.
      
