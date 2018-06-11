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

# Select the [2*spin,parity] combinations which we will include
# NB! Note the factor 2 for the spin! This is to represent all spins by integers.
Jpi_list = [[0,+1],[2,+1],[4,+1],[6,+1],[8,+1],[10,+1],[12,+1],[14,+1],[16,+1],[18,+1],[20,+1],[22,+1],[24,+1],[26,+1],[28,+1],
            [0,-1],[2,-1],[4,-1],[6,-1],[8,-1],[10,-1],[12,-1],[14,-1],[16,-1],[18,-1],[20,-1],[22,-1],[24,-1],[26,-1],[28,-1]]
# NB! If your nucleus has odd A, you need to replace Jpi_list by this one:
# Jpi_list = [[0+1,+1],[2+1,+1],[4+1,+1],[6+1,+1],[8+1,+1],[10+1,+1],[12+1,+1],[14+1,+1],[16+1,+1],[18+1,+1],[20+1,+1],[22+1,+1],[24+1,+1],[26+1,+1],[28+1,+1],
#             [0+1,-1],[2+1,-1],[4+1,-1],[6+1,-1],[8+1,-1],[10+1,-1],[12+1,-1],[14+1,-1],[16+1,-1],[18+1,-1],[20+1,-1],[22+1,-1],[24+1,-1],[26+1,-1],[28+1,-1]]

# Calculate the gsf
gsf = smutil.strength_function_average(levels, transitions, Jpi_list, bin_width, Emin, Emax, type="M1")

# Plot it
ax_gsf.plot(bins_middle[0:len(gsf)], gsf, label="Strength function")

# Make the plot nice
ax_gsf.set_yscale('log')
ax_gsf.set_ylabel(r'$f\, \mathrm{(MeV^{-3})}$')
ax_gsf.set_xlabel(r'$E_\gamma\,\mathrm{(MeV)}$')
ax_gsf.legend()

# Show plot
plt.show()