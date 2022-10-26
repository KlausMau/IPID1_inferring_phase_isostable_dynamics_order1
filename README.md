IPID-1: Inferring Phase-Isostable Dynamics of order 1
==========================================================================================================================

This code is an implementation of the IPID-1 algorithm from the paper [Inferring oscillator's phase and amplitude response from a scalar signal exploiting test stimulation](https://arxiv.org/abs/2206.09173).
It infers the [phase response curve](http://www.scholarpedia.org/article/Phase_response_curve) and isostable response curve from observations of a scalar oscillatory signal and a scalar perturbation signal. 
 
Note that this is for inference from observations only. If you have the system equations available there is a different, straightforward algorithm that you should use, see [repository on isostable coordinates](https://github.com/rokcestnik/isostable_coordinates_from_equations).

This implementation is in Python and orientates strongly on the [original C implementation](https://github.com/rokcestnik/IPID1_inferring_phase_isostable_dynamics_order1).


### Example

The script [example.py](example.py) is already set with an example. It first generates the data by integrating a simple system, and then runs the algorithm to infer the curves, and plots them together with the ground truth for this system.
