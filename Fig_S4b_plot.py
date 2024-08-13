import matplotlib.pyplot as plt
import h5py
import numpy as np
file_path = 'Data\Fig S4 Data\Driven_HEOM_results\driven_heom_b_eps_1.out'

data = []  # to store the parsed data

with open(file_path, 'r') as file:
    for line in file:
        # Split each line into a list of values using space as the delimiter
        values = [float(val) for val in line.split()]
        data.append(values)

# Now 'data' is a list of lists, where each inner list represents a line of values
# You can access specific values using indexing, e.g., data[0] for the first line
# or data[0][1] for the second value in the first line.

# Example: Print the entire data
time=[]
y=[]
y1=[]
y2=[]
for line in data:
    time.append((line[0]))
    y.append((line[1]))
    y1.append((line[7]))
    y2.append((line[3]))
fig, ax = plt.subplots()
ax.set_facecolor((0.95,0.95,0.95))
plt.rc("font", family="serif")
plt.plot(time,y,color='red',ls='dashed',marker='o',markevery=15,label=r'$\rho_{00}$ HEOM')
plt.plot(time,y1,color='red',ls='dashed',marker='x',markevery=15,label=r'$\rho_{11}$ HEOM')
plt.plot(time,y2,color='red',ls='dashed',marker='+',markevery=15,label=r'Re$\rho_{10}$ HEOM')
filename='Data\Fig S4 Data\drivendata/b.h5'
h5 = h5py.File(filename,'r')
#kmax=14
kmax=4
N=200
wantdelt=0.1
tarr = np.linspace(0,wantdelt*(N-1),num=N)
M = h5["A"]  # VSTOXX futures data

plt.plot(tarr,M[0,:],color='#808080',marker='o',ls='dotted',markevery=20)

plt.plot(tarr,M[3,:],color='#808080',marker='x',ls='dotted',markevery=20)
plt.plot(tarr,M[2,:],color='#808080',marker='+',ls='dotted',markevery=20)
filename='Data\Fig S4 Data\drivendata/tb25.h5'
h5 = h5py.File(filename,'r')
#kmax=14
kmax=4
N=200
wantdelt=0.1
tarr = np.linspace(0,wantdelt*(N-1),num=N)
M = h5["A"]  # VSTOXX futures data
UG= h5["UB"]
MM=np.zeros((4,200),dtype='complex')
for i in np.arange(0,N):
    MM[:,i]=M[:,i]
for i in np.arange(kmax+1,N):
    MM[:,i]=np.matmul(UG[:,:,i],np.transpose([1,0,0,0]))
plt.plot(tarr,MM[0,:],color='#808080',marker='o',markevery=20)

plt.plot(tarr,MM[3,:],color='#808080',marker='x',markevery=20)
plt.plot(tarr,MM[2,:],color='#808080',marker='+',markevery=20)




filename='Data\Fig S4 Data\drivendata/tb2.h5'
h5 = h5py.File(filename,'r')
#kmax=14
kmax=4
N=200
wantdelt=0.1
tarr = np.linspace(0,wantdelt*(N-1),num=N)
M = h5["A"]  # VSTOXX futures data
UG= h5["UB"]
MM=np.zeros((4,200),dtype='complex')
for i in np.arange(0,N):
    MM[:,i]=M[:,i]
for i in np.arange(kmax+1,N):
    MM[:,i]=np.matmul(UG[:,:,i],np.transpose([1,0,0,0]))
plt.plot(tarr,MM[0,:],color='black',marker='o',markevery=20)

plt.plot(tarr,MM[3,:],color='black',marker='x',markevery=20)
plt.plot(tarr,MM[2,:],color='black',marker='+',markevery=20)




plt.xticks(fontsize=15, fontname = 'serif')
plt.yticks(fontsize=15, fontname = 'serif')
plt.xlabel('Time',fontsize=15, fontname = 'serif')
plt.ylim([-0.5,1])
plt.xlim([0,20])
plt.legend()
#plt.savefig("Fig S4 Data\driven25.png",dpi=1000)
plt.show()