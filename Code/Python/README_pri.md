(20181111: For some reason, the jupytext pairing is not working. Specifically, the BufferStockTheory.py file that is generated does NOT regenerate the ipynb notebook as it is supposed to do. In fact, the *.py file that is generated does not even work on the command line. For the time being, both the jupyter notebook and the generated .py file, which each work individually, are locked until the problem can be diagnosed and repaired.

(Earlier text, from when it was working, is below).

The jupyter notebook BufferStockTheory.ipynb and the ipython script BufferStockTheory.py are an example of a 
[jupytext](https://github.com/mwouts/jupytext) `paired notebook` 

To activate the pairing, from a computer with the jupyter notebook software installed, just execute at the command line the command 

	`jupyter notebook` 
	
Then, assuming you have ipython installed on your computer, from the command line, execute the command 

	`ipython BufferStockTheory.py`

