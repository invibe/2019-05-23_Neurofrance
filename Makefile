default: show

SRC=2019-03-25_HDR_RobinBaures

edit:
	atom $(SRC).py

html:
	#python3 $(SRC).py $(SRC).html
	python3 $(SRC).py index.html

figures:
	rsvg-convert figures/boutin-franciosini-ruffier-perrinet-19_figure1_a.svg -f png -d 450 -p 450 -o figures/boutin-franciosini-ruffier-perrinet-19_figure1_a.png
	rsvg-convert figures/boutin-franciosini-ruffier-perrinet-19_figure1_b.svg -f png -d 450 -p 450 -o figures/boutin-franciosini-ruffier-perrinet-19_figure1_b.png
	rsvg-convert figures/boutin-franciosini-ruffier-perrinet-19_figure1_c.svg -f png -d 450 -p 450 -o figures/boutin-franciosini-ruffier-perrinet-19_figure1_c.png
	rsvg-convert figures/boutin-franciosini-ruffier-perrinet-19_figure1.svg -f png -d 450 -p 450 -o figures/boutin-franciosini-ruffier-perrinet-19_figure1.png

page:
	python3 $(SRC).py
	cat /tmp/talk.bib |pbcopy
	atom ~/pool/blog/perrinet_curriculum-vitae_tex/LaurentPerrinet_Presentations.bib
	# academic ...

show: html
#	open -a firefox $(SRC).html
	open /Applications/Safari.app/Contents/MacOS/Safari  index.html

github: html
	git commit --dry-run -am 'Test' | grep -q -v 'nothing to commit' && git commit -am' updating slides'
	git push
	open https://laurentperrinet.github.io/$(SRC)

print: html
	open -a /Applications/Chromium.app https://laurentperrinet.github.io/$(SRC)/?print-pdf&showNotes=true
	#open "https://laurentperrinet.github.io/$(SRC)/?print-pdf&showNotes=true"
