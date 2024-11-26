import numpy as np

def getRaw(infile):
    data = np.load(infile)
    # z1 = np.argwhere(data["z_bins"]>0.22)[0][0]
    # z1 = 0
    # z2 = np.argwhere(data["z_bins"]>0.22)[0][0]
    t = data["t_bins"].copy()
    x = data["x_bins"].copy()
    y = data["y_bins"].copy()
    z = data["z_bins"].copy()
    vx = data["vx_bins"].copy()
    vy = data["vy_bins"].copy()
    vz = data["vz_bins"].copy()
    phi = data["phi_bins"].copy()
    ex = data["ex_bins"].copy()
    ey = data["ey_bins"].copy()
    ez = data["ez_bins"].copy()
    bx = data["bx_bins"].copy()
    by = data["by_bins"].copy()
    bz = data["bz_bins"].copy()

    return t,x,y,z,\
           vx,vy,vz,\
           phi,\
           ex,ey,ez,\
           bx,by,bz

def getRawNoFields(infile):
    data = np.load(infile)
    # z1 = np.argwhere(data["z_bins"]>0.22)[0][0]
    # z1 = 0
    # z2 = np.argwhere(data["z_bins"]>0.22)[0][0]
    t = data["t_bins"].copy()
    x = data["x_bins"].copy()
    y = data["y_bins"].copy()
    z = data["z_bins"].copy()
    vx = data["vx_bins"].copy()
    vy = data["vy_bins"].copy()
    vz = data["vz_bins"].copy()
    phi = data["phi_bins"].copy()

    return t,x,y,z,\
           vx,vy,vz,\
           phi

def getGCS(infile):
    data = np.load(infile)
    # z1 = np.argwhere(data["z_bins"]>0.22)[0][0]
    # z1 = 0
    # z2 = np.argwhere(data["z"]>0.22)[0][0]

    t = data["t"].copy()
    x = data["x"].copy()
    y = data["y"].copy()
    z = data["z"].copy()
    vx = data["vx"].copy()
    vy = data["vy"].copy()
    vz = data["vz"].copy()
    phi = data["phi"].copy()
    ex = data["ex"].copy()
    ey = data["ey"].copy()
    ez = data["ez"].copy()
    bx = data["bx"].copy()
    by = data["by"].copy()
    bz = data["bz"].copy()
    KEx = data["KEx"].copy()
    KEy = data["KEy"].copy()
    KEz = data["KEz"].copy()
    radius = data["radius"].copy()

    return t,x,y,z,\
           vx,vy,vz,\
           phi,\
           ex,ey,ez,\
           bx,by,bz,\
           KEx,KEy,KEz,\
           radius

def getGCSNoFields(infile):
    data = np.load(infile)
    # z1 = np.argwhere(data["z_bins"]>0.22)[0][0]
    # z1 = 0
    # z2 = np.argwhere(data["z"]>0.22)[0][0]

    t = data["t"].copy()
    x = data["x"].copy()
    y = data["y"].copy()
    z = data["z"].copy()
    vx = data["vx"].copy()
    vy = data["vy"].copy()
    vz = data["vz"].copy()
    phi = data["phi"].copy()
    KEx = data["KEx"].copy()
    KEy = data["KEy"].copy()
    KEz = data["KEz"].copy()
    radius = data["radius"].copy()

    return t,x,y,z,\
           vx,vy,vz,\
           phi,\
           KEx,KEy,KEz,\
           radius

def getField(infile, resize):

    CSTdata = np.load(infile)

    x = CSTdata["x"]
    y = CSTdata["y"]
    z = CSTdata["z"]
    phi = CSTdata["phi"]
    ex = CSTdata["ex"]
    ey = CSTdata["ey"]
    ez = CSTdata["ez"]
    bx = CSTdata["bx"]
    by = CSTdata["by"]
    bz = CSTdata["bz"]

    x = np.reshape(x,resize)
    y = np.reshape(y,resize)
    z = np.reshape(z,resize)
    phi = np.reshape(phi,resize)
    ex = np.reshape(ex,resize)
    ey = np.reshape(ey,resize)
    ez = np.reshape(ez,resize)
    bx = np.reshape(bx,resize)
    by = np.reshape(by,resize)
    bz = np.reshape(bz,resize)

    return x,y,z,phi,ex,ey,ez,bx,by,bz

def traj_reduce(a, nreduce=2):
    nbins = len(a)//nreduce
    ntrunc = nbins*nreduce

    return np.reshape(a[:ntrunc], (nbins, nreduce)).mean(axis=1)

def convert_to_field_indices(xloc, yloc, zloc):
    return int(101*(xloc-(-0.05))/0.1),\
           int(31*(yloc-(-0.015))/0.03),\
           int(1101*(zloc-(-0.2))/1.1)


def normed(b):
    bmaxi = b.argmax()
    bmax = b[bmaxi]
    bn = b.copy()
    bn = bn/bmax
    return bn,bmax


def normed_flat_before_bmax(b):
    bmaxi = b.argmax()
    bmax = b[bmaxi]
    bn = b.copy()
    for i in range(bmaxi):
        bn[i] = bmax
    bn = bn/bmax
    return bn,bmax

def normed_to_z(z, b):
    bi = np.where(z>-0.1)[0][0]
    # bmaxi = b.argmax()
    bmax = b[bi]
    bn = b.copy()
    # for i in range(bi):
    #     bn[i] = bmax
    bn = bn/bmax
    return bn,bmax

def normed_to_value(z, b, maxval):
    # bi = np.where(z>-0.1)[0][0]
    # bmaxi = b.argmax()
    # bmax = b[bi]
    bn = b.copy()
    # for i in range(bi):
    #     bn[i] = bmax
    bn = bn/maxval
    return bn,maxval

class GCS():
    def __init__(self, attrs, fn):
        self.t,self.x,self.y,self.z,\
        self.vx,self.vy,self.vz,\
        self.phi,\
        self.ex,self.ey,self.ez,\
        self.bx,self.by,self.bz,\
        self.KEx,self.KEy,self.KEz,\
        self.radius = getGCS(fn)

        self.tke = self.KEy+self.KEz
        self.E = self.tke + self.KEx

        self.attrs = attrs
class GCSNoFields():
    def __init__(self, attrs, fn):
        self.t,self.x,self.y,self.z,\
        self.vx,self.vy,self.vz,\
        self.phi,\
        self.KEx,self.KEy,self.KEz,\
        self.radius = getGCSNoFields(fn)

        self.tke = self.KEy+self.KEz
        self.E = self.tke + self.KEx

        self.attrs = attrs
class Raw():
    def __init__(self, attrs, fn):

        mass = 9.109E-31
        c = 299792458
        eV = 1.60217E-19

        self.t,self.x,self.y,self.z,\
        self.vx,self.vy,self.vz,\
        self.phi,\
        self.ex,self.ey,self.ez,\
        self.bx,self.by,self.bz \
            = getRaw(fn)

        beta_x = self.vx/c
        beta_y = self.vy/c
        beta_z = self.vz/c

        gamma_x = np.divide(1,np.sqrt(1-beta_x**2))
        gamma_y = np.divide(1,np.sqrt(1-beta_y**2))
        gamma_z = np.divide(1,np.sqrt(1-beta_z**2))

        self.KEx = (gamma_x*mass*c**2-mass*c**2)/eV
        self.KEy = (gamma_y*mass*c**2-mass*c**2)/eV
        self.KEz = (gamma_z*mass*c**2-mass*c**2)/eV

        self.tke = self.KEy+self.KEz
        self.E = self.tke+self.KEx

        self.attrs = attrs

class RawNoFields():
    def __init__(self, attrs, fn):

        mass = 9.109E-31
        c = 299792458
        eV = 1.60217E-19

        self.t,self.x,self.y,self.z,\
        self.vx,self.vy,self.vz,\
        self.phi\
            = getRawNoFields(fn)

        beta_x = self.vx/c
        beta_y = self.vy/c
        beta_z = self.vz/c

        gamma_x = np.divide(1,np.sqrt(1-beta_x**2))
        gamma_y = np.divide(1,np.sqrt(1-beta_y**2))
        gamma_z = np.divide(1,np.sqrt(1-beta_z**2))

        self.KEx = (gamma_x*mass*c**2-mass*c**2)/eV
        self.KEy = (gamma_y*mass*c**2-mass*c**2)/eV
        self.KEz = (gamma_z*mass*c**2-mass*c**2)/eV

        self.tke = self.KEy+self.KEz
        self.E = self.tke+self.KEx

        self.attrs = attrs

class Field():
    def __init__(self, attrs, fn):

        resize = (101,31,1101)

        self.x,self.y,self.z,\
        self.phi,\
        self.ex,self.ey,self.ez,\
        self.bx,self.by,self.bz = getField(fn, resize)

        self.attrs = attrs
