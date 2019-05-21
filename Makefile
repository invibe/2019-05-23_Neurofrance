default: github

SRC=2019-05-23_Neurofrance

edit:
	atom $(SRC).py

html:
	python3 $(SRC).py index.html

page:
	python3 $(SRC).py
	cat /tmp/talk.bib |pbcopy
	#atom ~/pool/blog/perrinet_curriculum-vitae_tex/LaurentPerrinet_Presentations.bib
	# academic ...

figures:
	cp ../AnticipatorySPEM/figures/Result/scatter_velocity_sigmo_*.svg figures

show: html
#	open -a firefox $(SRC).html
	open /Applications/Safari.app/Contents/MacOS/Safari  index.html

github: html
	git commit --dry-run -am 'Test' | grep -q -v 'nothing to commit' && git commit -am' updating slides'
	git push
	# open https://invibe.github.io/$(SRC)

print: html
	#open -a /Applications/Chromium.app https://laurentperrinet.github.io/$(SRC)/?print-pdf&showNotes=true
	#open "https://laurentperrinet.github.io/$(SRC)/?print-pdf&showNotes=true"
	/Applications/Chromium.app/Contents/MacOS/Chromium --headless --disable-gpu --print-to-pdf=$(SRC).pdf "https://invibe.github.io/$(SRC)/?print-pdf"
