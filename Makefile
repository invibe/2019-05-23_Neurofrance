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

get_figures:
	# from the paper
	rsync -a ../PasturelMontagniniPerrinet2019/figures/{1_B_protocol_recording,1_C_protocol_bet}.png figures
	# methods
	rsync -a ../AnticipatorySPEM/figures/Experiment/{Experiment_classique_4_blocks,Experiment_block_0,Experiment_block_0_EM,Experiment_block_0_bet,Experiment_block_0_bet_EM}.svg figures
	rsync -a ../AnticipatorySPEM/figures/raw/{raw_trace,raw_fitted}.svg figures
	rsync -a ../AnticipatorySPEM/figures/Result/{Results_BCP_velocity_sigmo_*,kde_mean_velocity_sigmo_fixed,scatter_velocity_sigmo_fixed,scatter_velocity_sigmo_fixed,scatter_velocity_sigmo_mean}.svg figures
	# KDE
	rsync -a ../AnticipatorySPEM/figures/BCP/KDE_{bet_mean_fixed,bet_fixed,velo_mean_fixed,velo_fixed}.svg figures

show: html
#	open -a firefox $(SRC).html
	open /Applications/Safari.app/Contents/MacOS/Safari  index.html

github: html
	git add figures
	git commit --dry-run -am 'Test' | grep -q -v 'nothing to commit' && git commit -am' updating slides'
	git push
	# open https://invibe.github.io/$(SRC)

print: html
	#open -a /Applications/Chromium.app https://laurentperrinet.github.io/$(SRC)/?print-pdf&showNotes=true
	#open "https://laurentperrinet.github.io/$(SRC)/?print-pdf&showNotes=true"
	/Applications/Chromium.app/Contents/MacOS/Chromium --headless --disable-gpu --print-to-pdf=$(SRC).pdf "https://invibe.github.io/$(SRC)/?print-pdf"
