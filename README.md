# pyeom
A python code package accompanying "Unified Framework for Open Quantum Dynamics with Memory" at Nature Communications by Ivander, Lindoy, and Lee.

The main directory contains the code to plot the figures in the main text (and a figure set in the supplementary information). They are named by the figures they represent in the manuscript. Some will need modifying parameters to specify which figure panel the codes will produce. Important information are commented in the early parts of the code.

Inside "Data/" you will find the raw data for each figures and the job files that were required to produce them.

Inside "pyeom/" you will find the engine behind the computations, with codes for performing HEOM, QuaPI, Dyck path, TTM simulations, etc.
