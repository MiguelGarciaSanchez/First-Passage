First-Passage
=============

- Simulates  Brownian motions until absorption in a circunference or on a spherical surface.
- Computes the mean time until absorption for each initial position.
- Computes the mean time until escaping a circular patch on the surface of a sphere for each initial position.

Installation:
-------------
.. code:: bash

    git clone https://github.com/MiguelGarciaSanchez/Clean-First-Passage
    cd Clean-First-Passage
    pip install -r requirements.txt

Example:
--------
Diff_Sphere.py: Shows an example of a Brownian motion on a spherical surface.


Main Programms:
===============

main_sphere.py: 
---------------
	Computes the  mean time until absorption of a brownian particle diffusing on a 	spherical surface.
		|
		| Absorbing region: Represented by a spherical cap [[theta_min,theta_max],delta] with pole 	[theta_min, theta_max] and maximum arc length delta.
		|
		| Reflecting region: Given by min and max angles [theta_min, theta_max], [phi_min, phi_max]
		|
		| main_sphere.py counts with paralelization: You can choose the number of cores to use in the 		n_cores parameter.

main_circumference.py: 
----------------------
	Computes the global mean time until absorption of a brownian particle diffusing in a circumference.
		|
		| Absorbing region: Given by min and max angles [theta_min, theta_max]
		|
		| Also allows paralelization.

Data Storage:
=============
Each program saves the initial positions of the particles and their mean time until absorption in a dataframe. The dataframes are stored in (./Data/Sphere/ or ./Data/Circumference)

