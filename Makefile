build:
	@nvcc -o nn nn.cu

buildIfChanged:
	@find . -name '*.cu' -o -name '*.h' -o -name '*.cpp' -type f | xargs cat | md5 > /tmp/._checksums
	@if ! cmp -s "/tmp/._checksums" "/tmp/.checksums" ; then make build ; fi
	@rm -f /tmp/.checksums
	@mv /tmp/._checksums /tmp/.checksums

run:buildIfChanged
	./nn
