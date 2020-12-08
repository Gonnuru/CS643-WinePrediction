<p align="center">
	<img src="https://user-images.githubusercontent.com/28786206/101530599-22e4dd80-3960-11eb-92a6-989e8c7fbf3b.jpg">
</p>

# Wine Quality Prediction ML model in Spark over AWS

> A parallel Machine Learning application in Amazon AWS cloud platform built to predict the quality of Wine utilizing Apache Spark. 

> Github Link: https://github.com/Gonnuru/CS643-WinePrediction

> Docker Hub: https://hub.docker.com/repository/docker/sampathgonnuru/cs643-project2
--- 
## Table of Contents

- [EMR Installation](#EMR-Installation)
- [EC2 Instance without Docker](#EC2-Instance-without-Docker)
- [EC2 Instance With Docker](#EC2-Instance-With-Docker)
- [TroubleShooting](#Troubleshooting)

---

### EMR Installation

1) **Uploading Files**

*TrainingDataset.csv, ValidationDataset.csv and trained model data* were uploaded to S3 Bucket:

***URL***: ```
url : s3://mywineproject/ ```

2) **Creating an EMR cluster**

* **2.1**. Login to AWS console and create a IAM role for ec2 instance to give access to s3 so that ec2 instace can have access to download (CSV files) and upload files (Model file) to s3.

* **2.2**. Alternatively, you can also set your s3 bucket to public to give access to your files, **Note: This is generally not recommended as it has high security risks. If you are using an AWS Educate account you cannot modify your IAM policy and in such a scenario you can utilize this step**

3) **Creating a Cluster**

***Step 1:*** In the AWS dashboard under the `analytics` section click `EMR`

***Step 2:*** Now Click `Create Cluster` 

***Step 3:*** In the `General Configuration` for `Cluster Name` type desired cluster name.

* **Step 3.1**: Under ``Software configuration` in the application column click the button which shows `Spark: Spark 2.4.7 on Hadoop 2.10.1 YARN and Zeppelin 0.8.2``. 
          
* **Step 3.2:** Under `Hardware Configuration` click `m4.large` rather than the default `m5.xlarge` as the default m5.xlarge incurs a cost of $0.043/hr in contrast to the $0.03 for m4.large. Keep in mind that EMR incurs an additional 25% cost post first usage. 
          
* **Step 3.4:** Select `4` instances under the column `Number of instances` 
          
* **Step 3.5:** Under `Security and access` click the EC2 key pair already created else create a new one
          
***Step 4:*** Click Create Cluster button. Wait for around 15 minutes for the cluster to start functioning. 

***Alternative Step***: If you want to create the above procedure using command line you can use this 

```
aws emr create-cluster --applications Name=Hadoop Name=Spark --ec2-attributes '{"InstanceProfile":"EMR_EC2_DefaultRole","SubnetId":"subnet-5a042c64","EmrManagedSlaveSecurityGroup":"sg-0233a1837a16a7902","EmrManagedMasterSecurityGroup":"sg-0a31cc8dff123ed0b"}' --release-label emr-5.29.0 --log-uri 's3n://aws-logs-700559207820-us-east-1/elasticmapreduce/' --steps '[{"Args":["spark-submit","--deploy-mode","client","--packages","org.apache.hadoop:hadoop-aws:2.7.7","s3://mywineproject/train.py"],"Type":"CUSTOM_JAR","ActionOnFailure":"TERMINATE_CLUSTER","Jar":"command-runner.jar","Properties":"","Name":"Spark application"}]' --instance-groups '[{"InstanceCount":1,"EbsConfiguration":{"EbsBlockDeviceConfigs":[{"VolumeSpecification":{"SizeInGB":32,"VolumeType":"gp2"},"VolumesPerInstance":3}]},"InstanceGroupType":"MASTER","InstanceType":"m4.large","Name":"Master Instance Group"},{"InstanceCount":4,"EbsConfiguration":{"EbsBlockDeviceConfigs":[{"VolumeSpecification":{"SizeInGB":32,"VolumeType":"gp2"},"VolumesPerInstance":3}]},"InstanceGroupType":"CORE","InstanceType":"m4.large","Name":"Core Instance Group"}]' --configurations '[{"Classification":"spark","Properties":{}}]' --auto-terminate --service-role EMR_DefaultRole --enable-debugging --name 'CS643WineApp' --scale-down-behavior TERMINATE_AT_TASK_COMPLETION --region us-east-1
```

---

### EC2 Instance without Docker

1) **Creating EC2 Instance**
* **Step 1:** Under Compute Column in the `AWS Management Console` Click `EC2`
* **Step 2:** Under the `Instances` click `Create Instance`
* **Step 3:** Select the `AMI` of your choice. `Amazon Linux 2 AMI` is usually preferred
* **Step 4:** Select `Instance Type` I've chosen t2.micro as I'm using AWS Educate and this gives me t2.micro under free tier elgible
* **Step 5:** Here one can either `review and launch` or `tweak security, configuration and storage` features of EC2.
* Launch EC2 Instance

2) **Installing Spark on EC2 Instance**
* **Step 1:** Update EC2 using the command 
```bash
sudo yum update -y
```
* **step 2:** Check python version ```python3 --version```
* **Step 3:** Instal pip 
```bash
sudo pip install --upgrade pip
```
* **step 4:** Install Java and check if it works
```bash
sudo apt-get install default-jre
java --version
```
* **step 5:** Install Py4j
```bash
pip install py4j
```
* **Step 6:** Install Spark and hadoop
```bash
wget http://archive.apache.org/dist/spark/spark-3.0.0/spark-3.0.0-bin-hadoop2.7.tgz
sudo tar -zxvf spark-3.0.0-bin-hadoop2.7.tgz
```

* **step 7:** Install findspark
```bash
sudo pip install findspark
```

3) **Running your Application in EC2**
* Copy the predict.py ![](https://github.com/Gonnuru/CS643-WinePrediction/blob/main/predict.py) file to the Ec2 instance ```
scp -i <"your .pem file"> predict.py :~/predict.py```

* Run the following command in Ec2 instance to start the model prediction :
Example of using S3 file as argument 
```bash
spark-submit --packages org.apache.hadoop:hadoop-aws:2.7.7 predict.py s3://mywineproject/ValidationDataset.csv
```
---

### EC2 Instance With Docker

1) **Installation**
> Assuming that the above steps (#EC2-Instance-without-Docker) were clear for the setting up of EC2. Go ahead with the below steps post the setting up and running of EC2 to install docker
* **Step 1:** Command for installing the most recent Docker Community Edition package.
```bash
sudo yum install docker -y
```
* **Step 2:** Start the Docker service.
```bash
sudo service docker start
```
* **Step 3:**  Add the ec2-user to the docker group so you can execute Docker commands without using sudo.
```bash
sudo usermod -a -G docker ec2-user
```
* **Step 4:** Verify that the ec2-user can run Docker commands without sudo.
```bash
docker  --version or docker info
```

2) **Building Dockerfile**
* **Step 1:** Type `touch Dockerfile` to create a Dockerfile
* **Step 2:** nano Dockerfile and create the Dockerfile Image to automate the process
* **Step 3:** 
```bash
sudo docker build . -f Dockerfile -t <Image name of your choice>
```

3) **Pushing and Pulling created Image to DockerHub**
* **Step 1:** Login to your dockerhub account through ec2
```bash
docker login: Type your credentials
```
* **Step 2:** In order to push docker type the following commands
```bash
docker tag <Local Ec2 Repository name>:<Tag name> <dockerhub username>/<local Ec2 Repository name>
```
```bash
docker push <dockerhub username>/<local Ec2 Repository name>
```
* **Step 3:** Pulling your Dockerimage back to Ec2 
```bash
docker pull <dockerhub username>/<Repository name>:<tag name>
```
Example:
```bash
docker pull sampathgonnuru/cs643-project2:latest
```
* **Step 4:** Running my dockerimage
```bash
sudo docker run -t <Given Image name>
```
Example
```bash
docker run -it sampathgonnuru/cs643-project2:latest s3//mywineproject/ValidationDataset.csv 
```
---

### Troubleshooting

> In a scenario where one is unable to execute the program in the desired format then in such a case run the below commands

> These commands can be used for checking installation errors

``` bash 
sudo pip install --upgrade pip
```
```bash 
sudo pip install wheel
```
```bash 
sudo pip install pyspark --no-cache-dir
```
```bash 
sudo pip install findspark
```
```bash 
sudo pip install numpy
```
```bash 
sudo apt install python3-pip
```
```bash 
sudo docker rmi -f image image_name
```
```bash 
sudo docker images
```
```bash 
sudo docker system prune -a
```
```bash 
sudo docker ps
```
```bash 
sudo docker ps -a
```
```bash 
sudo docker run -t image_tag_name
```
```bash 
sudo docker build . -f docker_file_name -t image_tag_name
```
```bash 
sudo docker start container_name
```
```bash 
sudo docker stop container_name
```
```bash 
sudo docker login -u sampathgonnuru <ur docker-hub username>
```
```bash 
sudo groupadd docker
```


