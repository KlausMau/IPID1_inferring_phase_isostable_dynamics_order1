# replace the parameters
scp parameters.py parameters_bck.py
scp parameters_example.py parameters.py

echo "generating forcing"
python code/forcing_gen.py
echo "generating signal"
python code/signal_gen.py
echo "thresholding signal"
python code/thresholding.py
echo "initialize phase"
python code/initialize_phase.py
echo "iterating prc inference"
for i in {1..8}
do
	echo -e "\ti = "$i
	python code/prc_infer.py
	scp PRC/sol.txt PRC/iterations/$i.txt
	scp phase/phase.txt phase/iterations/$i.txt
done
echo "thresholding phase"
python code/phase_thresholding.py
echo "initialize amplitude"
python code/initialize_amplitude.py
echo "iterating arc inference"
for i in {1..8}
do
	echo -e "\ti = "$i
	python code/arc_infer.py
	scp ARC/sol.txt ARC/iterations/$i.txt
	scp amplitude/amplitude.txt amplitude/iterations/$i.txt
done
echo "plot"
python code/plot_example.py

# replace the parameters back
scp parameters_bck.py parameters.py
rm parameters_bck.py
