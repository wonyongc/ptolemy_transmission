import os

thetas_and_phis = { 
                    10: [0,45,90,135,180,225,270,315],
                    20: [0,45,90,135,180,225,270,315],
					70: [0,45,90,135,180,225,270,315],
					80: [0,45,90,135,180,225,270,315]
                }

for theta,phis in thetas_and_phis.items():
	for phi in phis:
		project_name    = f'th{theta}_phi{phi}'

		local_dir                  = '/home/wonyongc/src/notebooks/ptolemy/LNGS_target/slurm'
		project_particlemon_file   = f'{local_dir}/{project_name}_particlemon.mcr'
		project_pitchanalysis_file = f'{local_dir}/{project_name}_pitchanalysis.mcr'

		della_run_dir        	   = '/scratch/gpfs/wonyongc/lngs_target'
		della_particlemon_file     = f'della-feynman.princeton.edu:{della_run_dir}/{project_name}/Model/3D/particlemon.mcr'
		della_pitchanalysis_file   = f'della-feynman.princeton.edu:{della_run_dir}/{project_name}/Model/3D/pitchanalysis.mcr'

		local_modelrun_file   	   = '/home/wonyongc/src/notebooks/ptolemy/LNGS_target/templates/Model.run'
		della_modelrun_file        = f'della-feynman.princeton.edu:{della_run_dir}/{project_name}/Model/3D/Model.run'

		###################################################################################

		with open(f'{project_particlemon_file}','w') as file:
			file.write(f"""
Sub Main ()

	Dim monitornames as Variant
    Dim monitor As Variant
	Dim z As Double
	Dim pids() As Long

	monitornames = Array("monitor1","monitor2","monitor3","monitor4","monitor5","monitor6")

    Dim dict As Object

	Open "{della_run_dir}/transmission/{project_name}.txt" For Output As #1

	
    For Each monitor In monitornames

		Particle2DMonitorReader.SelectMonitor(monitor)

		For n = 0 To Particle2DMonitorReader.GetNPlanes()-1

			Particle2DMonitorReader.SelectPlane(n)
			z = Particle2DMonitorReader.GetPlaneDistance()

			pids = Particle2DMonitorReader.GetParticleIDs()

			Set dict = CreateObject("Scripting.Dictionary")

			For i = 0 To UBound(pids)

				If Not dict.Exists(pids(i)) Then
            		dict.Add pids(i), 1 ' Add new unique element
        		End If

			Next
			
			Dim uniqueCount As Integer
    		uniqueCount = dict.Count

			avgE = Particle2DMonitorReader.GetQuantityWithReduction("Energy","","mean")

			Write #1, Round(z,2), uniqueCount, avgE

		Next
	Next

	Close #1 

End Sub
			""")

		###################################################################################

		with open(f'{project_pitchanalysis_file}','w') as file:
			file.write(f"""
Sub Main ()

	Dim monitornames as Variant
    Dim monitor As Variant
	Dim z As Double
	Dim pids() As Long
	Dim x() As Single
	Dim y() As Single
	Dim px() As Single
	Dim py() As Single
	Dim pz() As Single
	Dim e() As Single

	monitornames = Array("monitor1","monitor2","monitor3","monitor4","monitor5","monitor6")

	Open "{della_run_dir}/pitchanalysis/{project_name}.txt" For Output As #1

    For Each monitor In monitornames

		Particle2DMonitorReader.SelectMonitor(monitor)

		For n = 0 To Particle2DMonitorReader.GetNPlanes()-1

			Particle2DMonitorReader.SelectPlane(n)
			z = Particle2DMonitorReader.GetPlaneDistance()

			pids = Particle2DMonitorReader.GetParticleIDs()

			x = Particle2DMonitorReader.GetPositionsX()
			y = Particle2DMonitorReader.GetPositionsY()

			px = Particle2DMonitorReader.GetMomentaX()
			py = Particle2DMonitorReader.GetMomentaY()
			pz = Particle2DMonitorReader.GetMomentaZ()

			e = Particle2DMonitorReader.GetQuantityValues("Energy","")

			For i = 0 To UBound(pids)

				Write #1, Round(z,2), pids(i), x(i), y(i), px(i), py(i), pz(i), e(i)

			Next

		Next
	Next

	Close #1 

End Sub
			""")

		os.system(f"scp {project_particlemon_file}      {della_particlemon_file}")
		os.system(f"scp {project_pitchanalysis_file}    {della_pitchanalysis_file}")
		# os.system(f"scp {local_modelrun_file}  {della_modelrun_file}")