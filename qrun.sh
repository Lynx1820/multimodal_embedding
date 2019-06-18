#!/bin/bash

# wallclock time reservation (format is hours:minutes:seconds).
# man 5 complex
#$ -l h_rt=02:00:00

# 8GB of memory reservation 8x1 = 8GB
#$ -l mem=8G

# name of job
# man 1 qsub
#$ -N my_job_name

# working directory (check for specific requirements for your research group)
# man 1 qsub
#$ -wd /nlp/users/dkeren
#$ -pe parallel-onenode 1
# interpret using BASH shell
#$ -S /bin/bash

# join standard error and standard output of script into job_name.ojob_id
#$ -j y -o /nlp/users/dkeren/errs

# export environment variables to job
#$ -V

# make sure I set my $CWD (current working directory)
cd /nlp/users/dkeren/multimodal_embedding

# when am I running
/bin/date

# where am I running
/bin/hostname

# what environment variables are available to this job script, e.g. $JOB_ID
/usr/bin/env

python image_feature_extraction.py --pid $1 --workers $2 --config $3 --dict $4
