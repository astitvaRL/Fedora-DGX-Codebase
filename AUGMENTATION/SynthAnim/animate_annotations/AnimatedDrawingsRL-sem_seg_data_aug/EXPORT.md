## Documentation for using pygltfgen export with export_view.py
This file keep track of the temporary workflow to use pygltfgen library with export_view before it's consolidated. 

## Steps to setup:
1. Download rust following the instructions: https://rustup.rs/#
2. cd libs/pygltfgen, make sure this submodule is checked out. If not, pull these submodules
- ```git submodule update --init --recursive``` (if hasn't initialized)
- ```git submodule update --remote``` 
3. In the pygltfgen library, in the terminal: ```maturin develop```
The dependencies will be set up automatically.



