# Author: Aadil Ahamed
# plot_results.p: plot the results of the nn_experiment

reset
set xtic auto
set ytic auto
set title "Brute Force vs KD Tree Comparison for NN on CPU"
set xlabel "Input Size (records)"
set ylabel "Running Time (ms)"
set key outside
plot "./output/result.txt" using 1:2 with points pt 5 title 'brute force', \
"./output/result.txt" using 1:3 with points pt 7 title 'kd tree'
