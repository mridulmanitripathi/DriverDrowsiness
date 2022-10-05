#!/bin/bash

### Job name
#SBATCH -J Techlabs

## OUTPUT AND ERROR
#SBATCH -o job.%j.out
#SBATCH -e job.%j.err

## specify your mail address (send feedback when job is done)
#SBATCH --mail-type=begin  				 # send email when process begins...
#SBATCH --mail-type=end    				 # ...and when it ends...
#SBATCH --mail-type=fail   		 	         # ...or when it fails.
#SBATCH --mail-user=mridulmanitripathi@gmail.com #mridul.tripathi@rwth-aachen.de	 # send notifications to this email REPLACE WITH YOUR EMAIL

### Change to current directory (directory job is submitted from)
#SBATCH -D ./
##SBATCH --no-requeue


## Setup of execution environment 
##SBATCH --export=NONE

### Avoid fragmentation of cluster
###notChanged

### Set tasks per node specific for each cluster
##SBATCH --ntasks-per-node=

### Request the time you need for execution in hh:mm:ss
#SBATCH -t 100:00:00
 
### Request memory you need for your job
#SBATCH --mem-per-cpu=50G
 
### Request the number of tasks you want to set
##SBATCH -n 1
#SBATCH --nodes=1
#SBATCH --ntasks=1

### Do not share nodes (much better performance)
##SBATCH --exclusive

## Set cpus per task (multiple tasks for hyperthreading)
##SBATCH --cpus-per-task=1


### Prevent loading of default modules
### Change to the work directory
##cd $HOME/workdirectory
### load modules and execute

source ~/.zshrc
source activate CV

python3 -u alexnet.py
