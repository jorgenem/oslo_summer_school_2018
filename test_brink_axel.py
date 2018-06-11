from __future__ import division
import numpy as np 
import matplotlib.pyplot as plt 
import shellmodelutilities as smutil

# Set bin width and range
bin_width = 0.20
Emin = 4  # Minimum and maximum excitation
Emax = 10 # energy over which to extract strength function
Nbins = int(np.ceil(Emax/bin_width))
Emax_adjusted = bin_width*Nbins # Trick to get an integer number of bins
bins = np.linspace(0,Emax_adjusted,Nbins+1)
bins_middle = (bins[0:-1]+bins[1:])/2 # Array of middle-bin values, to use for plotting gsf

# Define list of calculation input files and corresponding label names
inputfile = ""

# Instantiate figure which we will fill
f_gsf, ax_gsf = plt.subplots(1,1)

# Read energy levels from file
levels = smutil.read_energy_levels(inputfile)

# Read M1 transitions from the same file
transitions = smutil.read_transition_strengths(inputfile, type="M1")

# Select the [spin,parity] combinations which we will include
Jpi_lists = [
             [[0,+1]],         # combination 1,
             [[2,+1],[4,+1]],  # combination 2, etc
            ]

for i in range(len(Jpi_lists)):
    Jpi_list = Jpi_lists[i]
    # Calculate the gsf
    gsf = smutil.strength_function_average(levels, transitions, Jpi_list, bin_width, Emin, Emax, type="M1")
    
    # Plot it
    ax_gsf.plot(bins_middle[0:len(gsf)], gsf, label="combination "+str(i))

# Make the plot nice
ax_gsf.set_yscale('log')
ax_gsf.set_ylabel(r'$f\, \mathrm{(MeV^{-3})}$')
ax_gsf.set_xlabel(r'$E_\gamma\,\mathrm{(MeV)}$')
ax_gsf.legend()

# Show plot
plt.show()