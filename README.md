IPID-1: Inferring Phase-Isostable Dynamics of order 1
==========================================================================================================================

This code is an implementation of the IPID-1 algorithm from the paper [Inferring oscillator's phase and amplitude response from a scalar signal exploiting test stimulation](https://arxiv.org/abs/2206.09173).
It infers the [phase response curve](http://www.scholarpedia.org/article/Phase_response_curve) and isostable response curve from observations of a scalar oscillatory signal and a scalar perturbation signal. 
 
Note that this is for inference from observations only. If you have the system equations available there is a different, straightforward algorithm that you should use, see my other [repository on isostable coordinates](https://github.com/rokcestnik/isostable_coordinates_from_equations).

The implementation was originally in Python and then important parts were rewritten in c for efficiency. All the algorithms are in the directory 'code', the rest of the directories are there to organize the input and output. It is ran from a single bash script: [run.bash](run.bash). The code uses [Eigen](http://eigen.tuxfamily.org/) for a short linear algebra part.

### Setup
1. clone/download the repository, 
2. add [Eigen](http://eigen.tuxfamily.org/) library files to the folder 'code/Eigen' (or link them when compiling in the next step), 
3. compile the code by going to the 'code' directory and run 
```
bash compile.bash
```

### Execution

The parameters are in the file [parameters.py](parameters.py). Put the signal observations in 'signal/signal_x.txt' file, and the perturbation signal in the 'forcing/forcing.txt' one value per line (no commas). Run it by executing the bash script [run.sh](run.sh):

```
bash run.sh
```

### Example

The script [run_example.sh](run_example.sh) is already set with an example. It first generates the data by integrating a simple system, and then runs the algorithm to infer the curves, and plots them. 
