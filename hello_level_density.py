from __future__ import division
import numpy as np 
import matplotlib.pyplot as plt 
import shellmodelutilities as smutil



# Set bin width and range
bin_width = 0.20
Emax = 20
Nbins = int(np.ceil(Emax/bin_width))
Emax_adjusted = bin_width*Nbins # Trick to get an integer number of bins
bins = np.linspace(0,Emax_adjusted,Nbins+1)

# Define list of calculation input files and corresponding label names
inputfile = ""

# Instantiate figure which we will fill
f_rho, ax_rho = plt.subplots(1,1)

# Read energy levels from file
levels = smutil.read_energy_levels(inputfile)

# Calculate level density
rho = smutil.total_level_density(levels, bin_width, Emax)

# Plot it
ax_rho.step(bins, np.append(0,rho), where='pre', label="Level density")

# Make the plot nice
ax_rho.set_yscale('log')
ax_rho.set_xlabel(r'$E_x \, \mathrm{(MeV)}$')
ax_rho.set_ylabel(r'$\rho \, \mathrm{(MeV^{-1})}$')
ax_rho.legend()

# Show plot
plt.show()