from __future__ import division
import numpy as np 
import matplotlib.pyplot as plt 
import shellmodelutilities as smutil

# Set bin width and range
bin_width = 0.20
Emin = 0  # Minimum and maximum excitation
Emax = 20 # energy over which to extract strength function
Nbins = int(np.ceil(Emax/bin_width))
Emax_adjusted = bin_width*Nbins # Trick to get an integer number of bins
bins = np.linspace(0,Emax_adjusted,Nbins+1)
bins_middle = (bins[0:-1]+bins[1:])/2 # Array of middle-bin values, to use for plotting gsf

# Define list of calculation input files and corresponding label names
inputfile = ""

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


# ===== Copied stuff from the strength function function in the shellmodelutilities =====
Nbins = int(np.ceil(Emax/bin_width)) # Make sure the number of bins cover the whole Ex region.
# print "Emax =", Emax, "Nbins*bin_width =", Nbins*bin_width
bin_array = np.linspace(0,bin_width*Nbins,Nbins+1) # Array of lower bin edge energy values
bin_array_middle = (bin_array[0:-1]+bin_array[1:])/2 # Array of middle bin values
# Find index of first and last bin (lower bin edge) where we put counts.
# It's important to not include the other Ex bins in the averaging later, because they contain zeros which will pull the average down.
i_Exmin = int(np.floor(Emin/bin_width)) 
i_Exmax = int(np.floor(Emax/bin_width))  

prefactor = {"M1":  11.5473e-9, "E1": 1.047e-6}

Egs = levels[0,0] # Read out the absolute ground state energy, so we can get relative energies later

# Allocate matrices to store the summed B(M1) values for each pixel, and the number of transitions counted
B_pixel_sum = np.zeros((Nbins,Nbins,len(Jpi_list)))
B_pixel_count = np.zeros((Nbins,Nbins,len(Jpi_list)))


# Loop over all transitions and put in the correct pixel:
for i_tr in range(len(transitions[:,0])):
  Ex = transitions[i_tr,2] - Egs
  # Check if transition is below Emax, skip if not
  if Ex < Emin or Ex >= Emax:
    continue

  # Get bin index for Eg and Ex (initial). Indices are defined with respect to the lower bin edge.
  i_Eg = int(np.floor(transitions[i_tr,6]/bin_width))
  i_Ex = int(np.floor(Ex/bin_width))

  # Read initial spin and parity of level:
  Ji = int(transitions[i_tr,0])
  pi = int(transitions[i_tr,1])
  try:
    i_Jpi = Jpi_list.index([Ji,pi])
  except: 
    continue

  # Add B(M1) value and increment count to pixel, respectively
  B_pixel_sum[i_Ex,i_Eg,i_Jpi] += transitions[i_tr,7]

  B_pixel_count[i_Ex,i_Eg,i_Jpi] += 1 # Original



# Allocate (Ex,Jpi) matrix to store level density
rho_ExJpi = np.zeros((Nbins,len(Jpi_list)))
# Count number of levels for each (Ex, J, pi) pixel.
for i_l in range(len(levels[:,0])):
  E, J, pi = levels[i_l]
  # Skip if level is outside range:
  if E-Egs >= Emax:
    continue
  i_Ex = int(np.floor((E-Egs)/bin_width))
  try:
    i_Jpi = Jpi_list.index([J,pi])
  except:
    continue
  rho_ExJpi[i_Ex,i_Jpi] += 1

rho_ExJpi /= bin_width # Normalize to bin width, to get density in MeV^-1


# Calculate gamma strength functions for each Ex, J, pi individually, using the partial level density for each J, pi.
gSF = np.zeros((Nbins,Nbins,len(Jpi_list)))
a = prefactor["M1"] # conversion constant
for i_Jpi in range(len(Jpi_list)):
  for i_Ex in range(Nbins):
    gSF[i_Ex,:,i_Jpi] = a * smutil.div0(B_pixel_sum[i_Ex,:,i_Jpi], B_pixel_count[i_Ex,:,i_Jpi]) * rho_ExJpi[i_Ex, i_Jpi]
gSF_currentExrange = gSF[i_Exmin:i_Exmax+1,:,:]
gSF_Jpiavg = smutil.div0(gSF_currentExrange.sum(axis=(2)), (gSF_currentExrange!=0).sum(axis=(2)))

# Instantiate figure which we will fill
f_gsf, ax_gsf = plt.subplots(1,1)

# Plot it
from matplotlib.colors import LogNorm # To get log scaling on the z axis
colorbar_object = ax_gsf.pcolormesh(bins, bins, gSF_Jpiavg, norm=LogNorm())
f_gsf.colorbar(colorbar_object) # Add colorbar to plot

# Make the plot nice
plt.gca().set_title('gSF_explot')
ax_gsf.set_xlabel(r"$gSF -- E_\gamma \, \mathrm{(MeV)}$")
ax_gsf.set_ylabel(r'$E_x \, \mathrm{(MeV)}$')

# New Figure: Oslo Method like 1st Gen matrix
f_gsf, ax_gsf = plt.subplots(1,1)
plt.gca().set_title('oslo_matrix')

# To receive the 1st Gen. matrix P, we use P ~ rho(Ex-Eg) * T(Eg); a
# assuming dipol radiation this yiels P ~ rho(Ex-Eg) * gSF(Eg) * Eg^3
oslo_matrix = np.zeros((Nbins,Nbins))
for i_Ex in range(Nbins):
  for i_Eg in range(Nbins):
    i_Ediff = i_Ex-i_Eg
    if i_Ediff>=0: # no gamma's with higher energy then the excitation energy
      oslo_matrix[i_Ex,i_Eg] = gSF_Jpiavg[i_Ex,i_Eg] * pow(i_Eg,3.) * np.sum(rho_ExJpi, axis=1)[i_Ediff]

# Plot it
from matplotlib.colors import LogNorm # To get log scaling on the z axis
colorbar_object = ax_gsf.pcolormesh(bins, bins, oslo_matrix, norm=LogNorm())
f_gsf.colorbar(colorbar_object) # Add colorbar to plot

# Make the plot nice
ax_gsf.set_xlabel(r"$E_\gamma \, \mathrm{(MeV)}$")
ax_gsf.set_ylabel(r'$E_x \, \mathrm{(MeV)}$')

# Show plot
plt.show()