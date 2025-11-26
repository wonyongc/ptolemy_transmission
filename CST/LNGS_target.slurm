#!/bin/bash
#SBATCH --job-name=LNGS_target
#SBATCH --output=/your/output/dir/LNGS_target.out # stdout file
#SBATCH --error=/your/output/dir/LNGS_target.err  # stderr file
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=160G
#SBATCH --time=12:00:00
#SBATCH --mail-type=all
#SBATCH --mail-user=your_email@princeton.edu

module purge
module load cst/2024
/usr/licensed/Dassault/CST_Studio_Suite_2024/cst_design_environment -as -dump 30 /location/to/LNGS_target.cst