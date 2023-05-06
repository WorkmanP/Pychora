PyChora: Python-Chess Opening Recommendation Algorithm

Libraries used:
	PySimpleGUI:
		Installed using: "pip install PySimpleGUI"
		Used to generate and receive inputs from the GUI
		Not used during the pychora_db.py and pychora_rec.py programs

	tkinter:
		Installed using: "pip install tk"
		(If running on a Unix system, also use: "sudo apt-get install python3-tk")
		Used to generate and receive inputs from the GUI
		Not used during the pychora_db.py and pychora_rec.py programs

	Matplotlib:
		Installed using: "pip install matplot"
		Used to generate cluster graphs and plot data

	Chess:
		Installed using: "pip install chess"
		Used to read to and from PGN files, additionally used in opening data types
		and recommendations for consistency
	
	pandas:	
		Installed using: "pip install pandas"
		Used for plotting cluster graphs and for calculaing BIC scores
		for Gaussian Mixed Models

	SciPy:
		Installed using "pip install scipy"
		Used for quality of life calculations
	
	sklearn:
		installed using "pip install -U scikit-learn"
		Used to generate GMM and clustering algorithms.
