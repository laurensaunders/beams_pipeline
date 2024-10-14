# beams_pipeline

beams_pipeline is a tool that allows you to make a beam and then analyze the leakage spectra for that beam. 
You can use it to make a completely analytic beam simulation, or to use a real beam (from HFSS), and then 
analyze that beam using the Map Multi Tool code.

To get the Map Multi Tool on your computer, go to your command line or terminal and paste in:

```git clone https://github.com/CMB-S4/map_multi_tool.git```

For the beam systematics pipeline (this repo), paste in:

```git clone https://github.com/laurensaunders/beams_pipeline.git```

You will also need ```numpy```, ```scipy```, ```matplotlib```, ```math```, and ```time```, so make sure that each of those is also installed on your computer. Typically, you can install each of these with ```pip install package_name```.

Once you have installed the packages, open the file ```beams_pipeline/mmt_tools.py```. In line 5 of this code, change ```path_prefix``` to the directory that holds ```map_multi_tool``` on your computer. This will allow you to import ```map_multi_tool``` to use with this module. If ```map_multi_tool``` and ```beams_pipeline``` are not held in the same outer directory, you will also need to change the definition of ```beams_pipeline_prefix``` in line 8 of this code.
