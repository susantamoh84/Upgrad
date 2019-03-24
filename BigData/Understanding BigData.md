# Understanding BigData

  - The 4 V's of BigData
  
    - Volume: This represents the amount of data generated by a company/organisation. The size of big data typically ranges from 
      petabytes (1000 TBs) to exabytes (10^6 TBs). For example, remember the amount of data generated in an internet minute, 
      which leads to the data explosion. Search engines such as Google, Yahoo, etc. deal with enormous volumes of data.

    - Velocity: This indicates the rate at which data is generated/consumed. Social media sites such as Twitter, Facebook, etc. 
      generate data from every activity that a user undertakes, leading to an enormous amount of data every minute.

    - Variety: This represents the different types of data being generated. For example, the different data forms for 
      Gmail may be sign-up/registration data, user login data, inbox emails, sent emails, etc.

    - Veracity: This represents the quality and accuracy of the data. Previously, veracity was not considered to be a 
      characteristic of big data. But with increasing analyses on generated data, veracity plays an important role.
      
      
# BigData Processing Framework - Hadoop

  - Key Terminologies in Hadoop:
  
    - Block: The data is divided into smaller chunks called blocks.
    
    - Commodity machine or node: This is a machine without any special hardware to process huge amounts of data, such as a 
      personal computer. A commodity machine/node stores the data.
      
    - Rack: This is a collection of commodity machines/nodes.
    
    - Data centres: These are physical locations where multiple racks are stored together. A group of racks makes a data centre.
    
    - Cluster: This is a collection of data centres. It can also be the case that only a certain rack in the data centre is a part of 
      the cluster. So in other words, a cluster has many nodes which are part of a rack, which is a part of a data centre.  
      
    - Client node: This is used to read and write data in the Hadoop cluster. This special type of node is not a part of the 
      cluster and does not store blocks.
      
    - Master node: This assigns tasks to other nodes
    
    - Slave node: This node executes the assigned tasks by master node
    
    - 4 different types of nodes: NameNode, Secondary Node, StandBy Node, Data Node
    
# Advantages of Distributed File Systems

  - Scalability: This method of distributing data among different machines is easily scalable because as the data grows, 
    you just have to add more nodes to the cluster to increase its storage and processing capacity.
  - Fault tolerance: If the name node fails, you have a backup node to take over and keep the cluster alive. In case a few 
    data blocks get corrupted, the cluster will still function, unlike a single computer where if a part of the file gets corrupted, 
    the whole file becomes unavailable.
  - Low-cost implementation: Hadoop runs on commodity machines, which are low-cost computers that lead to a lower implementation cost. 
  - Flexibility: Hadoop can process any kind of data (structured, semi-structured, and unstructured).
  
# Yet Another Resource Negotiator

  - 3 layers of Hadoop:
  
    - The Storage Layer: HDFS
    - The resource management layer: YARN
    - The Data-processing layer: Map-Reduce
    
  - Coming to the structure of YARN, there are 4 components in it:

    - Resource manager: It looks after the resources of the entire cluster. 
      The slave machines run multiple tasks which require different resources for execution, such as memory (RAM), CPU, etc. 
      These resources are allocated by the resource manager.

    - Node manager: The node manager keeps track of the resources running on the slave machine (one data node) it is residing on. 
      It manages the resources allocated for each of the applications (also known as tasks) inside the node and 
      communicates the status of the resources to the resource manager.

    - Application Master: A job is subdivided into multiple tasks. Each task is individually managed by the application master. 
      It looks after the lifecycle of the task and sends timely requests to the resource manager for the required containers.

    - Containers: To perform each task, the application master will require different components like memory (RAM) or CPU or 
      storage disks. These components are collectively termed as containers and are managed by the node manager. 
      These containers are the actual place where the task is performed.
      
      