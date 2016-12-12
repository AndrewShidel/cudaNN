build:
	if ! cmp -s "nn.cu" ".nn_sav" ; then nvcc -o nn nn.cu ; fi
	cp nn.cu .nn_sav

run:build
	./nn
