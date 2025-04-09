#!/bin/bash
JOBNAME=$1

#!/bin/bash

JOBNAME=$1

cat <<EOF > temp_submit.sh
#!/bin/bash
#SBATCH --job-name=${JOBNAME}
#SBATCH --output=${JOBNAME}.out
#SBATCH --error=${JOBNAME}.err
#SBATCH --partition=general
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=20G
#SBATCH --gres=gpu:1

python main.py --run_name ${JOBNAME}
# python predict.py
EOF

sbatch temp_submit.sh