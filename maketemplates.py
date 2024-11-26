import os, sys

runtype = sys.argv[1]

project_name    = 'cst_tutorial'

homedir = '/home/wonyongc'

local_dir             = f'{homedir}/src/notebooks/ptolemy/LNGS_target/slurm'
local_cst_macro_file  = f'{homedir}/src/notebooks/ptolemy/LNGS_target/LNGS_target.mcs'
project_macro_file    = f'{homedir}/src/notebooks/ptolemy/LNGS_target/slurm/{project_name}.mcs'

della_run_dir          = '/scratch/gpfs/wonyongc/lngs_target'
della_cst_template_dir = '/home/wonyongc/src/ptolemy'

local_run_dir          = '/mnt/d/lngs_target'
local_cst_template_dir = '/mnt/d/lngs_target'

slurm_file      = f'{project_name}.slurm'
della_sh_file   = f'{project_name}_della.sh'
local_sh_file   = f'{project_name}_local.sh'
local_cst_sh_file   = f'{project_name}_cst_local.sh'

# Magnet - catchz1 270m cells, catchz125 325m, z150 342m
# Target shell straight 230m
cpus            = '12'
mem             = '220G'
time            = '72:00:00'

# catcher - 150m cells
# cpus            = '12'
# mem             = '100G'
# time            = '10:00:00'

# 5stage - 180m cells
# cpus            = '12'
# mem             = '160G'
# time            = '12:00:00'

# 5stage particle ring - 180m cells
# cpus            = '12'
# mem             = '140G'
# time            = '48:00:00'

###################################################################################
if runtype == "della":
    with open(f'{local_dir}/{slurm_file}','w') as file:
        file.write(f"""#!/bin/bash
#SBATCH --partition=physics
#SBATCH --job-name={project_name}
#SBATCH --output={della_run_dir}/{project_name}.out # stdout file
#SBATCH --error={della_run_dir}/{project_name}.err  # stderr file
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={cpus}
#SBATCH --mem={mem}
#SBATCH --time={time}
#SBATCH --mail-type=all
#SBATCH --mail-user=wonyongc@princeton.edu
#SBATCH --constraint=cascade

module purge
module load cst/2024
/usr/licensed/Dassault/CST_Studio_Suite_2024/cst_design_environment -as -dump 30 {della_run_dir}/{project_name}.cst

        """)

    ################################################################################### 
    with open(f'{local_dir}/{local_sh_file}','w') as file:
        file.write(f"""
cp  {local_cst_macro_file}        {project_macro_file}
scp {project_macro_file}          della-feynman.princeton.edu:{della_run_dir}/
scp {local_dir}/{slurm_file}      della-feynman.princeton.edu:{della_run_dir}/
scp {local_dir}/{della_sh_file}   della-feynman.princeton.edu:{della_run_dir}/

ssh della-feynman.princeton.edu '{della_run_dir}/{della_sh_file}'

        """)

    ###################################################################################
    with open(f'{local_dir}/{della_sh_file}','w') as file:
        file.write(f"""

cp -f  {della_cst_template_dir}/cst_template.cst    {della_run_dir}/{project_name}.cst
cp -fr {della_cst_template_dir}/cst_template/       {della_run_dir}/{project_name}
cp -f  {della_run_dir}/{project_name}.mcs           {della_run_dir}/{project_name}/Model/3D/b.mcs

# sbatch {della_run_dir}/{slurm_file}

        """)

    os.system(f"chmod +x {local_dir}/{local_sh_file}")
    os.system(f"chmod +x {local_dir}/{della_sh_file}")
    os.system(f"{local_dir}/{local_sh_file}")


if runtype == "local":
    with open(f'{local_dir}/{local_cst_sh_file}','w') as file:
        file.write(f"""
cp -f  {local_cst_macro_file}        {project_macro_file}
# cp -f  {local_cst_template_dir}/cst_template.cst    {local_run_dir}/{project_name}.cst
# cp -fr {local_cst_template_dir}/cst_template/       {local_run_dir}/{project_name}
cp -f  {project_macro_file}                         {local_run_dir}/{project_name}/Model/3D/b.mcs

        """)
    os.system(f"chmod +x {local_dir}/{local_cst_sh_file}")
    os.system(f"{local_dir}/{local_cst_sh_file}")
###################################################################################

