import os

lambdas = [50,55,60,65,70,75]
feys = ['020', '025']
convfiles = []
batchfile = '/home/wonyongc/src/notebooks/ptolemy/LNGS_target/slurm/convertCSTtoNPZ_batch.sh'


for lamb in lambdas:
    for fey in feys:
        project_name    = f'bn_adj25n20_o10_post{lamb}neg{lamb}_fey{fey}'
        trajstart       = 0
        trajend         = 5

        local_dir         = '/home/wonyongc/src/notebooks/ptolemy/LNGS_target/slurm'
        project_conv_file = f'{local_dir}/{project_name}_conv.py'

        della_run_dir     = '/scratch/gpfs/wonyongc/lngs_target/traj/'
        della_conv_file   = f'della-feynman.princeton.edu:{della_run_dir}/{project_name}_conv.py'


        with open(f'{project_conv_file}','w') as file:
            file.write(f"""
import numpy as np

def process_CST_traj(infile, outfile):
    data = np.loadtxt(infile, delimiter=",")

    t_bins,\
    x_bins, y_bins, z_bins,\
    vx_bins, vy_bins, vz_bins,\
    phi_bins,\
    ex_bins, ey_bins, ez_bins,\
    bx_bins, by_bins, bz_bins\
        = data.T

    with open(outfile, 'wb') as f:
        np.savez(f,
        t_bins=t_bins,
        x_bins=x_bins, y_bins=y_bins, z_bins=z_bins,
        vx_bins=vx_bins, vy_bins=vy_bins, vz_bins=vz_bins,
        phi_bins=phi_bins,
        ex_bins=ex_bins, ey_bins=ey_bins, ez_bins=ez_bins,
        bx_bins=bx_bins, by_bins=by_bins, bz_bins=bz_bins)
        
def process_CST_trajNoFields(infile, outfile):
    data = np.loadtxt(infile, delimiter=",")

    t_bins,\
    x_bins, y_bins, z_bins,\
    vx_bins, vy_bins, vz_bins,\
    phi_bins\
    = data.T

    with open(outfile, 'wb') as f:
        np.savez(f,
        t_bins=t_bins,
        x_bins=x_bins, y_bins=y_bins, z_bins=z_bins,
        vx_bins=vx_bins, vy_bins=vy_bins, vz_bins=vz_bins,
        phi_bins=phi_bins)

def saveGCStraj(infile, outfile, prefix=3, GCS=True):

    mass = 9.109E-31
    c = 299792458
    eV = 1.60217E-19

    CSTdata = np.load(infile)

    t = CSTdata["t_bins"]
    x = CSTdata["x_bins"]
    y = CSTdata["y_bins"]
    z = CSTdata["z_bins"]
    vx = CSTdata["vx_bins"]
    vy = CSTdata["vy_bins"]
    vz = CSTdata["vz_bins"]
    phi = CSTdata["phi_bins"]
    ex = CSTdata["ex_bins"]
    ey = CSTdata["ey_bins"]
    ez = CSTdata["ez_bins"]
    bx = CSTdata["bx_bins"]
    by = CSTdata["by_bins"]
    bz = CSTdata["bz_bins"]

    beta_x = vx/c
    beta_y = vy/c
    beta_z = vz/c

    gamma_x = np.divide(1,np.sqrt(1-beta_x**2))
    gamma_y = np.divide(1,np.sqrt(1-beta_y**2))
    gamma_z = np.divide(1,np.sqrt(1-beta_z**2))

    KEx = (gamma_x*mass*c**2-mass*c**2)/eV
    KEy = (gamma_y*mass*c**2-mass*c**2)/eV
    KEz = (gamma_z*mass*c**2-mass*c**2)/eV

    if not GCS:
        return t, vy, vz

    vz_signs = np.diff(np.sign(vz[prefix:]))
    vy_signs = np.diff(np.sign(vy[prefix:]))

    signsums = np.abs(np.cumsum(vy_signs))+np.abs(np.cumsum(vz_signs))
    bucket_splits = np.where(np.diff(np.sign(signsums))==-1)[0]

    tGCS = np.empty(bucket_splits.size-1)
    xGCS = np.empty(bucket_splits.size-1)
    yGCS = np.empty(bucket_splits.size-1)
    zGCS = np.empty(bucket_splits.size-1)
    vxGCS = np.empty(bucket_splits.size-1)
    vyGCS = np.empty(bucket_splits.size-1)
    vzGCS = np.empty(bucket_splits.size-1)
    phiGCS = np.empty(bucket_splits.size-1)
    exGCS = np.empty(bucket_splits.size-1)
    eyGCS = np.empty(bucket_splits.size-1)
    ezGCS = np.empty(bucket_splits.size-1)
    bxGCS = np.empty(bucket_splits.size-1)
    byGCS = np.empty(bucket_splits.size-1)
    bzGCS = np.empty(bucket_splits.size-1)
    radiusGCS = np.empty(bucket_splits.size-1)
    KExGCS = np.empty(bucket_splits.size-1)
    KEyGCS = np.empty(bucket_splits.size-1)
    KEzGCS = np.empty(bucket_splits.size-1)

    for i in np.arange(bucket_splits.size-1):
        c1 = bucket_splits[i]
        c2 = bucket_splits[i+1]

        tGCS[i] = t[c1:c2].mean()
        xGCS[i] = x[c1:c2].mean()
        yGCS[i] = y[c1:c2].mean()
        zGCS[i] = z[c1:c2].mean()
        vxGCS[i] = vx[c1:c2].mean()
        vyGCS[i] = vy[c1:c2].mean()
        vzGCS[i] = vz[c1:c2].mean()
        phiGCS[i] = phi[c1:c2].mean()
        exGCS[i] = ex[c1:c2].mean()
        eyGCS[i] = ey[c1:c2].mean()
        ezGCS[i] = ez[c1:c2].mean()
        bxGCS[i] = bx[c1:c2].mean()
        byGCS[i] = by[c1:c2].mean()
        bzGCS[i] = bz[c1:c2].mean()
        KExGCS[i] = KEx[c1:c2].mean()
        KEyGCS[i] = KEy[c1:c2].mean()
        KEzGCS[i] = KEz[c1:c2].mean()

        radiusGCS[i] = ( np.abs(y[c1:c2].min()-y[c1:c2].max())+
                        np.abs(z[c1:c2].min()-z[c1:c2].max()) )/4

    with open(outfile, 'wb') as f:
        np.savez(f,
                t=tGCS,
                x=xGCS, y=yGCS, z=zGCS,
                vx=vxGCS, vy=vyGCS, vz=vzGCS,
                phi=phiGCS,
                ex=exGCS, ey=eyGCS, ez=ezGCS,
                bx=bxGCS, by=byGCS, bz=bzGCS,
                KEx=KExGCS, KEy=KEyGCS, KEz=KEzGCS,
                radius=radiusGCS)

    # return tGCS, xGCS, yGCS, zGCS,\
    #        vxGCS, vyGCS, vzGCS,\
    #        phiGCS, exGCS, eyGCS, ezGCS,\
    #        bxGCS, byGCS, bzGCS,\
    #        KExGCS, KEyGCS, KEzGCS, radiusGCS

    # v = np.stack((vx, vy, vz), axis=1)
    # Bfield = np.stack((bx, by, bz), axis=1)
    # gradB = np.gradient(Bfield, axis=0)
    # Efield = np.stack((ex, ey, ez), axis=1)
    # F = eV*(Efield + np.cross(v, Bfield))
    #
    # if nreduce:
    #     nbins = len(t) // nreduce
    #     ntrunc = nreduce * nbins
    #
    #     t_reshape = np.reshape(t[:ntrunc], (nbins, nreduce))
    #     x_reshape = np.reshape(x[:ntrunc], (nbins, nreduce))
    #     y_reshape = np.reshape(y[:ntrunc], (nbins, nreduce))
    #     z_reshape = np.reshape(z[:ntrunc], (nbins, nreduce))
    #     vx_reshape = np.reshape(vx[:ntrunc], (nbins, nreduce))
    #     vy_reshape = np.reshape(vy[:ntrunc], (nbins, nreduce))
    #     vz_reshape = np.reshape(vz[:ntrunc], (nbins, nreduce))
    #     phi_reshape = np.reshape(phi[:ntrunc], (nbins, nreduce))
    #     ex_reshape = np.reshape(ex[:ntrunc], (nbins, nreduce))
    #     ey_reshape = np.reshape(ey[:ntrunc], (nbins, nreduce))
    #     ez_reshape = np.reshape(ez[:ntrunc], (nbins, nreduce))
    #     bx_reshape = np.reshape(bx[:ntrunc], (nbins, nreduce))
    #     by_reshape = np.reshape(by[:ntrunc], (nbins, nreduce))
    #     bz_reshape = np.reshape(bz[:ntrunc], (nbins, nreduce))
    #     KE_x_reshape = np.reshape(KE_x[:ntrunc], (nbins, nreduce))
    #     KE_y_reshape = np.reshape(KE_y[:ntrunc], (nbins, nreduce))
    #     KE_z_reshape = np.reshape(KE_z[:ntrunc], (nbins, nreduce))
    #     pitch_reshape = np.reshape(pitch[:ntrunc], (nbins, nreduce))
    #
    #     t = t_reshape.mean(axis=1)
    #     x = x_reshape.mean(axis=1)
    #     y = y_reshape.mean(axis=1)
    #     y_min = y_reshape.min(axis=1)
    #     y_max = y_reshape.max(axis=1)
    #     z = z_reshape.mean(axis=1)
    #     vx = vx_reshape.mean(axis=1)
    #     vy = vy_reshape.mean(axis=1)
    #     vz = vz_reshape.mean(axis=1)
    #     phi = phi_reshape.mean(axis=1)
    #     ex = ex_reshape.mean(axis=1)
    #     ey = ey_reshape.mean(axis=1)
    #     ez = ez_reshape.mean(axis=1)
    #     bx = bx_reshape.mean(axis=1)
    #     by = by_reshape.mean(axis=1)
    #     bz = bz_reshape.mean(axis=1)
    #     KE_x = KE_x_reshape.mean(axis=1)
    #     KE_y = KE_y_reshape.mean(axis=1)
    #     KE_z = KE_z_reshape.mean(axis=1)
    #     pitch = pitch_reshape.mean(axis=1)
    #
    #     return t, x, y, y_min, y_max, z,\
    #        vx, vy, vz,\
    #        phi, ex, ey, ez,\
    #        bx, by, bz,\
    #        pitch, KE_x, KE_y, KE_z
    #
    #
    # idx1 = (np.abs(z - zlim_low)).argmin()
    # idx2 = (np.abs(z - zlim_high)).argmin()
    #
    # t = t[idx1:idx2]
    # x = x[idx1:idx2]
    # y = y[idx1:idx2]
    # y_min = y_min[idx1:idx2]
    # y_max = y_max[idx1:idx2]
    # z = z[idx1:idx2]
    # vx = vx[idx1:idx2]
    # vy = vy[idx1:idx2]
    # vz = vz[idx1:idx2]
    # phi = phi[idx1:idx2]
    # ex = ex[idx1:idx2]
    # ey = ey[idx1:idx2]
    # ez = ez[idx1:idx2]
    # bx = bx[idx1:idx2]
    # by = by[idx1:idx2]
    # bz = bz[idx1:idx2]
    #

def saveGCStrajNoFields(infile, outfile, prefix=3, GCS=True):

    mass = 9.109E-31
    c = 299792458
    eV = 1.60217E-19

    CSTdata = np.load(infile)

    t = CSTdata["t_bins"]
    x = CSTdata["x_bins"]
    y = CSTdata["y_bins"]
    z = CSTdata["z_bins"]
    vx = CSTdata["vx_bins"]
    vy = CSTdata["vy_bins"]
    vz = CSTdata["vz_bins"]
    phi = CSTdata["phi_bins"]
    # ex = CSTdata["ex_bins"]
    # ey = CSTdata["ey_bins"]
    # ez = CSTdata["ez_bins"]
    # bx = CSTdata["bx_bins"]
    # by = CSTdata["by_bins"]
    # bz = CSTdata["bz_bins"]

    beta_x = vx/c
    beta_y = vy/c
    beta_z = vz/c

    gamma_x = np.divide(1,np.sqrt(1-beta_x**2))
    gamma_y = np.divide(1,np.sqrt(1-beta_y**2))
    gamma_z = np.divide(1,np.sqrt(1-beta_z**2))

    KEx = (gamma_x*mass*c**2-mass*c**2)/eV
    KEy = (gamma_y*mass*c**2-mass*c**2)/eV
    KEz = (gamma_z*mass*c**2-mass*c**2)/eV

    if not GCS:
        return t, vy, vz

    vz_signs = np.diff(np.sign(vz[prefix:]))
    vy_signs = np.diff(np.sign(vy[prefix:]))

    signsums = np.abs(np.cumsum(vy_signs))+np.abs(np.cumsum(vz_signs))
    bucket_splits = np.where(np.diff(np.sign(signsums))==-1)[0]

    tGCS = np.empty(bucket_splits.size-1)
    xGCS = np.empty(bucket_splits.size-1)
    yGCS = np.empty(bucket_splits.size-1)
    zGCS = np.empty(bucket_splits.size-1)
    vxGCS = np.empty(bucket_splits.size-1)
    vyGCS = np.empty(bucket_splits.size-1)
    vzGCS = np.empty(bucket_splits.size-1)
    phiGCS = np.empty(bucket_splits.size-1)
    # exGCS = np.empty(bucket_splits.size-1)
    # eyGCS = np.empty(bucket_splits.size-1)
    # ezGCS = np.empty(bucket_splits.size-1)
    # bxGCS = np.empty(bucket_splits.size-1)
    # byGCS = np.empty(bucket_splits.size-1)
    # bzGCS = np.empty(bucket_splits.size-1)
    radiusGCS = np.empty(bucket_splits.size-1)
    KExGCS = np.empty(bucket_splits.size-1)
    KEyGCS = np.empty(bucket_splits.size-1)
    KEzGCS = np.empty(bucket_splits.size-1)

    for i in np.arange(bucket_splits.size-1):
        c1 = bucket_splits[i]
        c2 = bucket_splits[i+1]

        tGCS[i] = t[c1:c2].mean()
        xGCS[i] = x[c1:c2].mean()
        yGCS[i] = y[c1:c2].mean()
        zGCS[i] = z[c1:c2].mean()
        vxGCS[i] = vx[c1:c2].mean()
        vyGCS[i] = vy[c1:c2].mean()
        vzGCS[i] = vz[c1:c2].mean()
        phiGCS[i] = phi[c1:c2].mean()
        # exGCS[i] = ex[c1:c2].mean()
        # eyGCS[i] = ey[c1:c2].mean()
        # ezGCS[i] = ez[c1:c2].mean()
        # bxGCS[i] = bx[c1:c2].mean()
        # byGCS[i] = by[c1:c2].mean()
        # bzGCS[i] = bz[c1:c2].mean()
        KExGCS[i] = KEx[c1:c2].mean()
        KEyGCS[i] = KEy[c1:c2].mean()
        KEzGCS[i] = KEz[c1:c2].mean()

        radiusGCS[i] = ( np.abs(y[c1:c2].min()-y[c1:c2].max())+
                        np.abs(z[c1:c2].min()-z[c1:c2].max()) )/4

    with open(outfile, 'wb') as f:
        np.savez(f,
                t=tGCS,
                x=xGCS, y=yGCS, z=zGCS,
                vx=vxGCS, vy=vyGCS, vz=vzGCS,
                phi=phiGCS,
                #  ex=exGCS, ey=eyGCS, ez=ezGCS,
                #  bx=bxGCS, by=byGCS, bz=bzGCS,
                KEx=KExGCS, KEy=KEyGCS, KEz=KEzGCS,
                radius=radiusGCS)

    # return tGCS, xGCS, yGCS, zGCS,\
    #        vxGCS, vyGCS, vzGCS,\
    #        phiGCS, exGCS, eyGCS, ezGCS,\
    #        bxGCS, byGCS, bzGCS,\
    #        KExGCS, KEyGCS, KEzGCS, radiusGCS

    # v = np.stack((vx, vy, vz), axis=1)
    # Bfield = np.stack((bx, by, bz), axis=1)
    # gradB = np.gradient(Bfield, axis=0)
    # Efield = np.stack((ex, ey, ez), axis=1)
    # F = eV*(Efield + np.cross(v, Bfield))
    #
    # if nreduce:
    #     nbins = len(t) // nreduce
    #     ntrunc = nreduce * nbins
    #
    #     t_reshape = np.reshape(t[:ntrunc], (nbins, nreduce))
    #     x_reshape = np.reshape(x[:ntrunc], (nbins, nreduce))
    #     y_reshape = np.reshape(y[:ntrunc], (nbins, nreduce))
    #     z_reshape = np.reshape(z[:ntrunc], (nbins, nreduce))
    #     vx_reshape = np.reshape(vx[:ntrunc], (nbins, nreduce))
    #     vy_reshape = np.reshape(vy[:ntrunc], (nbins, nreduce))
    #     vz_reshape = np.reshape(vz[:ntrunc], (nbins, nreduce))
    #     phi_reshape = np.reshape(phi[:ntrunc], (nbins, nreduce))
    #     ex_reshape = np.reshape(ex[:ntrunc], (nbins, nreduce))
    #     ey_reshape = np.reshape(ey[:ntrunc], (nbins, nreduce))
    #     ez_reshape = np.reshape(ez[:ntrunc], (nbins, nreduce))
    #     bx_reshape = np.reshape(bx[:ntrunc], (nbins, nreduce))
    #     by_reshape = np.reshape(by[:ntrunc], (nbins, nreduce))
    #     bz_reshape = np.reshape(bz[:ntrunc], (nbins, nreduce))
    #     KE_x_reshape = np.reshape(KE_x[:ntrunc], (nbins, nreduce))
    #     KE_y_reshape = np.reshape(KE_y[:ntrunc], (nbins, nreduce))
    #     KE_z_reshape = np.reshape(KE_z[:ntrunc], (nbins, nreduce))
    #     pitch_reshape = np.reshape(pitch[:ntrunc], (nbins, nreduce))
    #
    #     t = t_reshape.mean(axis=1)
    #     x = x_reshape.mean(axis=1)
    #     y = y_reshape.mean(axis=1)
    #     y_min = y_reshape.min(axis=1)
    #     y_max = y_reshape.max(axis=1)
    #     z = z_reshape.mean(axis=1)
    #     vx = vx_reshape.mean(axis=1)
    #     vy = vy_reshape.mean(axis=1)
    #     vz = vz_reshape.mean(axis=1)
    #     phi = phi_reshape.mean(axis=1)
    #     ex = ex_reshape.mean(axis=1)
    #     ey = ey_reshape.mean(axis=1)
    #     ez = ez_reshape.mean(axis=1)
    #     bx = bx_reshape.mean(axis=1)
    #     by = by_reshape.mean(axis=1)
    #     bz = bz_reshape.mean(axis=1)
    #     KE_x = KE_x_reshape.mean(axis=1)
    #     KE_y = KE_y_reshape.mean(axis=1)
    #     KE_z = KE_z_reshape.mean(axis=1)
    #     pitch = pitch_reshape.mean(axis=1)
    #
    #     return t, x, y, y_min, y_max, z,\
    #        vx, vy, vz,\
    #        phi, ex, ey, ez,\
    #        bx, by, bz,\
    #        pitch, KE_x, KE_y, KE_z
    #
    #
    # idx1 = (np.abs(z - zlim_low)).argmin()
    # idx2 = (np.abs(z - zlim_high)).argmin()
    #
    # t = t[idx1:idx2]
    # x = x[idx1:idx2]
    # y = y[idx1:idx2]
    # y_min = y_min[idx1:idx2]
    # y_max = y_max[idx1:idx2]
    # z = z[idx1:idx2]
    # vx = vx[idx1:idx2]
    # vy = vy[idx1:idx2]
    # vz = vz[idx1:idx2]
    # phi = phi[idx1:idx2]
    # ex = ex[idx1:idx2]
    # ey = ey[idx1:idx2]
    # ez = ez[idx1:idx2]
    # bx = bx[idx1:idx2]
    # by = by[idx1:idx2]
    # bz = bz[idx1:idx2]
    #

def process_CST_fields(infile, outfile):
    data = np.loadtxt(infile, delimiter=",")

    x,y,z, phi, ex,ey,ez, bx,by,bz = data.T

    with open(outfile, 'wb') as f:
        np.savez(f,x=x,y=y,z=z,phi=phi,ex=ex,ey=ey,ez=ez,bx=bx,by=by,bz=bz)

def overwritePhi(field_file, phi_file):

    '''
    Overwrites phi results in field_file with phi from phi_file
    '''
    x,y,z,phi,ex,ey,ez,bx,by,bz = CSTfields(phi_file)
    phi_keep = phi.copy()
    x,y,z,phi,ex,ey,ez,bx,by,bz = CSTfields(field_file)

    with open(field_file, 'wb') as f:
        np.savez(f,x=x,y=y,z=z,phi=phi_keep,ex=ex,ey=ey,ez=ez,bx=bx,by=by,bz=bz)

def process_multi_track(folder, filebase,n1,n2):

    for i in range(n1,n2):
        infile = folder +filebase+"_raw_"+str(i)+".txt"

        rawoutfile = folder + filebase + "_raw_"+str(i)+".npz"
        gcsoutfile = folder + filebase + "_GCS_"+str(i)+".npz"
        process_CST_traj(infile, rawoutfile)
        saveGCStraj(rawoutfile, gcsoutfile)

def process_multi_trackNoFields(folder, filebase,n1,n2):

    for i in range(n1,n2):
        infile = folder +filebase+"_raw_"+str(i)+".txt"

        rawoutfile = folder + filebase + "_raw_"+str(i)+".npz"
        gcsoutfile = folder + filebase + "_GCS_"+str(i)+".npz"
        process_CST_trajNoFields(infile, rawoutfile)
        saveGCStrajNoFields(rawoutfile, gcsoutfile)

def process_field(folder, filebase):
    field_file_txt = folder + filebase + "/efields.txt"
    field_file_npz = folder + filebase + "/efields.npz"
    process_CST_fields(field_file_txt, field_file_npz)


folder = "{della_run_dir}"
base  = "{project_name}"

process_multi_track(folder,base,{trajstart},{trajend})
        """)
        
        convfiles.append(f'{della_run_dir}{project_name}_conv.py')

        os.system(f"scp {project_conv_file} {della_conv_file}")

with open(f'{batchfile}','w') as file:
    for cf in convfiles:
        file.write(f'python {cf}\n')

os.system(f"scp {batchfile} della-feynman.princeton.edu:/scratch/gpfs/wonyongc/lngs_target/traj/")
