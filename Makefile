PY=python3
PDB=-m pdb

main:
	cd src && $(PY) main.py

bert:
	cd src && $(PY) bert.py

debug:
	cd src && $(PY) $(PDB) main.py

clean_csv:
	rm csv/*

clean_txt:
	rm txt/*
