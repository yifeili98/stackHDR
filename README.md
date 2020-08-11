# stackHDR

## Usage
put the input image filenames and their exposure time in a textfile 
for example ```inputs.txt```:
```
sample_image/image1.jpg 1/30 
sample_image/image2.jpg 1/15 
sample_image/image2.jpg 1/5 
sample_image/image2.jpg 1/2 
```
Then create a output folder, for example ```/output_folder```
We could generate tone mapped image in the folder by calling
```python ./src/hdr ./inputs.txt output_folder```
  
Note that some MTB may break some pre-aligned image (for example the memorial church by Paul Debevec), which the break is caused by the inappropriate shift for pre-alignment for the images, in this case, we add a random letter as the third argument to the call, for example:  
```python ./src/hdr ./inputs.txt output_folder n```  
Then we could disable image alignment.