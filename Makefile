.PHONY: data

data:
	mkdir -p data
	cd data; curl -L -o GSE273980_RAW.tar "https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE273980&format=file"

run:
	python3 -B separate_functions/main.py

visualizations:
	python3 -B separate_functions/visualization.py

clean:
	rm data/*