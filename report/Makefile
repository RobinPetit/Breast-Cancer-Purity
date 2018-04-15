report: report.pdf
	evince report.pdf &

report.pdf: report.tex report.bib
	latex report
	bibtex report
	latex report
	latex report
	dvips -o report.ps -t letterSize report.dvi
	ps2pdf report.ps report.pdf

clean:
	# LaTeX
	rm -f report.aux
	rm -f report.log
	#BibTex
	rm -f report.blg
	rm -f report.bbl
	# print
	rm -f report.dvi
	rm -f report.ps
	rm -f report.pdf
