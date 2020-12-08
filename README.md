<p align="center">
	<img src="https://user-images.githubusercontent.com/28786206/101530599-22e4dd80-3960-11eb-92a6-989e8c7fbf3b.jpg">
</p>

# Wine Quality Prediction ML model in Spark over AWS

> A parallel Machine Learning application in Amazon AWS cloud platform built to predict the quality of Wine utilizing Apache Spark. 

## Table of Contents

- [EMR Installation](#installation)
- [EC2 Instance without Docker](#EC2 Instance without Docker)
- [EC2 Instance With Docker](#EC2 Instance With Docke)
- [TroubleShooting](#Troubleshooting)

---

### Installation

1) **Uploading Files**
*TrainingDataset.csv, ValidationDataset.csv and trained model data* were uploaded to S3 Bucket:
***URL***: ```url : s3://mywineproject/```

2) **Creating an EMR cluster**

2.1. Login to AWS console and create a IAM role for ec2 instance to give access to s3 so that ec2 instace can have access to download (CSV files) and upload files (Model file) to s3.

2.2. Alternatively, you can also set your s3 bucket to public to give access to your files, **Note: This is generally not recommended as it has high security risks. If you are using an AWS Educate account you cannot modify your IAM policy and in such a scenario you can utilize this step**

3) **Creating a Cluster**

*Step 1:* In the AWS dashboard under the `analytics` section click `EMR`

*Step 2:* Now Click `Create Cluster` 

*Step 3:* In the `General Configuration` for `Cluster Name` type desired cluster name.

          Under `Software configuration` in the application column click the button which shows `Spark: Spark 2.4.7 on Hadoop 2.10.1 YARN and Zeppelin 0.8.2`
	  
          Under `Hardware Configuration` click `m4.large` rather than the default `m5.xlarge` as the default m5.xlarge incurs a cost of $0.043/hr in contrast to the $0.03 for 	           m4.large. Keep in mind that EMR incurs an additional 25% cost post first usage. 
	  
          Select `4` instances under the column `Number of instances`
	  
          Under `Security and access` click the EC2 key pair already created else create a new one
	  
*Step 4:* Click Create Cluster button. Wait for around 15 minutes for the cluster to start functioning. 

*Alternative Step*: If you want to create the above procedure using command line you can use this 

```
aws emr create-cluster --applications Name=Hadoop Name=Spark --ec2-attributes '{"InstanceProfile":"EMR_EC2_DefaultRole","SubnetId":"subnet-5a042c64","EmrManagedSlaveSecurityGroup":"sg-0233a1837a16a7902","EmrManagedMasterSecurityGroup":"sg-0a31cc8dff123ed0b"}' --release-label emr-5.29.0 --log-uri 's3n://aws-logs-700559207820-us-east-1/elasticmapreduce/' --steps '[{"Args":["spark-submit","--deploy-mode","client","--packages","org.apache.hadoop:hadoop-aws:2.7.7","s3://mywineproject/train.py"],"Type":"CUSTOM_JAR","ActionOnFailure":"TERMINATE_CLUSTER","Jar":"command-runner.jar","Properties":"","Name":"Spark application"}]' --instance-groups '[{"InstanceCount":1,"EbsConfiguration":{"EbsBlockDeviceConfigs":[{"VolumeSpecification":{"SizeInGB":32,"VolumeType":"gp2"},"VolumesPerInstance":3}]},"InstanceGroupType":"MASTER","InstanceType":"m4.large","Name":"Master Instance Group"},{"InstanceCount":4,"EbsConfiguration":{"EbsBlockDeviceConfigs":[{"VolumeSpecification":{"SizeInGB":32,"VolumeType":"gp2"},"VolumesPerInstance":3}]},"InstanceGroupType":"CORE","InstanceType":"m4.large","Name":"Core Instance Group"}]' --configurations '[{"Classification":"spark","Properties":{}}]' --auto-terminate --service-role EMR_DefaultRole --enable-debugging --name 'CS643WineApp' --scale-down-behavior TERMINATE_AT_TASK_COMPLETION --region us-east-1
```
