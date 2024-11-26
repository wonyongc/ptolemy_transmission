import numpy as np
import matplotlib.pyplot as plt
import os

def process():

    thetas_and_phis = { 
                        10: [0,45,90,135,180,225,270,315],
                        20: [0,45,90,135,180,225,270,315],
                        30: [0,45,90,135,180,225,270,315],
                        40: [0,45,90,135,180,225,270,315],
                        50: [0,45,90,135,180,225,270,315],
                        60: [0,45,90,135,180,225,270,315],
                        70: [0,45,90,135,180,225,270,315],
                        80: [0,45,90,135,180,225,270,315]
                    }
                    
    rows, cols = 1, 1
    fig, ax1 = plt.subplots(rows, cols, figsize=(12 * cols, 8 * rows))

    for theta,phis in thetas_and_phis.items():

        data = {}

        for phi in phis:

            print(theta, phi)

            filename = f'th{theta}_phi{phi}.txt'

            z, pid, x, y, px, py, pz, e = np.loadtxt(filename, delimiter=',').T
            p = np.stack((px,py,pz), axis=1)
            costh = np.sum(p * np.array([1,0,0]), axis=1)
            pitch = np.arccos(abs(costh))*180/np.pi

            for zi, pidi, xi, yi, pitchi, ei in zip(z, pid, x, y, pitch, e):
                if abs(xi)>0.005: continue
                if zi not in data:
                    # data[zi] = {}
                    data[zi] = []
                # if pidi not in data[zi]:
                    # data[zi][pidi] = []
                # data[zi][pidi].append(pitchi)
                data[zi].append(pitchi)

            # avg_pitch_per_pid = {}
            # for zi, pid_dict in data.items():
                # if zi not in avg_pitch_per_pid:
                    # avg_pitch_per_pid[zi] = []
                # for pid, pitches in pid_dict.items():
                    # avg_pitch_per_pid[zi].append( np.mean( np.array(pitches) ) )    

            # zplot = avg_pitch_per_pid.keys()

            # z_all = []
            # avg_pitch_by_pid = []

            # for zi, pitches in avg_pitch_per_pid.items():
                # for pitchi in pitches:
                    # z_all.append(zi)
                    # avg_pitch_by_pid.append(pitchi)
            
        zplot = data.keys()
        avg_pitch_all_pids = [np.mean(data[z]) for z in zplot]

        ax1.scatter(zplot, avg_pitch_all_pids, s=1, label=f'theta={theta}')
        # ax2.scatter(z_all, avg_pitch_by_pid, s=1, label=f'theta={theta}')


    ax1.set_xlabel('z')
    ax1.set_ylabel('pitch')
    ax1.set_title('z vs. Avg pitch across all PIDs')
    ax1.grid(True)
    ax1.legend()

    # ax2.set_xlabel('z')
    # ax2.set_ylabel('pitch')
    # ax2.set_title('z vs. Avg pitch per PID')
    # ax2.grid(True)
    # ax2.legend()

    fig.suptitle(f'pitch distributions')

    plt.savefig(f'pitch_analysis.png')

if __name__ == '__main__':
    process()