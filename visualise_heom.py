import numpy as np
import matplotlib.pyplot as plt
import h5py
import argparse

def plot(fname, inds):
    h5 = None
    try:
        h5 = h5py.File(fname, 'r')
        t = np.array(h5.get('t'))
        rho = np.array(h5.get('rho'))

        h5.close()
    except:
        print("Failed to read input file")
        return


    for i in inds:
        try:
            plt.plot(t, np.real(rho[:, i]), '-', label=r'$\mathrm{Re}(\rho_{%d}(t))$'%(i))
            plt.plot(t, np.imag(rho[:, i]), '--', label=r'$\mathrm{Im}(\rho_{%d}(t))$'%(i))
        except:
            print("Failed to plot index: %d"%(i))
    plt.legend(frameon=False)
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot density matrix elements output by a heom calculation.')
    parser.add_argument('fname', type=str)
    parser.add_argument('--inds', nargs='+', default = [0])

    args = parser.parse_args()
    x = [int(i) for i in args.inds]
    plot(args.fname, x)


