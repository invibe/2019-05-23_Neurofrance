__author__ = "Laurent Perrinet INT - CNRS"
__licence__ = 'GPL licence'
DEBUG = True
DEBUG = False

fig_width = 12

import os
home = os.environ['HOME']
figpath_talk = 'figures'
figpath_slides = os.path.join(home, 'nextcloud/libs/slides.py/figures/')
#
import sys
print(sys.argv)
tag = sys.argv[0].split('.')[0]
if len(sys.argv)>1:
    slides_filename = sys.argv[1]
else:
    slides_filename = None

from academic import slugify

print('😎 Welcome to the script generating the slides for ', tag)
YYYY = int(tag[:4])
MM = int(tag[5:7])
DD = int(tag[8:10])

# see https://github.com/laurentperrinet/slides.py
from slides import Slides

height_px = 80
height_ratio = .7

meta = dict(
 embed = False,
 draft = DEBUG, # show notes etc
 width= 1600,
 height= 1000,
 # width= 1280, #1600,
 # height= 1024, #1000,
 margin= 0.1618,#
 reveal_path='https://cdnjs.cloudflare.com/ajax/libs/reveal.js/3.7.0/',
 #reveal_path='https://s3.amazonaws.com/hakim-static/reveal-js/',
 theme='simple',
 bgcolor="white",
 author='Laurent Perrinet, Chloé Pasturel & Anna Montagnini',
 author_link=f'<a href="https://laurentperrinet.github.io/talk/{slugify(tag)}/">Laurent Perrinet</a>, Chloé Pasturel & Anna Montagnini',
 short_title='Detecting switches in a volatile environment',
 title='Should I stay or should I go? Humans adapt to the volatility of visual motion properties, and know about it',
 conference_url='https://www.neurosciences.asso.fr/V2/colloques/SN19/',
 short_conference='NeuroFrance 2019',
 conference='Colloque international de la Société des Neurosciences 2019',
 location='Marseille (France)',
 abstract="""Animal behavior must constantly adapt to changes, for example when the state of an environmental context changes unexpectedly. For an agent that interacts with this volatile setting, it is important to react accurately and as quickly as possible. For example, it has already been shown that when a random sequence of directions of motion to the right or left of a visual target is suddenly biased to one direction, human observers adapt to accurately anticipate it with their eye movements. Here, we prove that this ability extends to a volatile environment where probability biases could change at random switching times. In addition, we also recorded the level of confidence reported by human observers. These results were compared to those of a probabilistic agent that is optimal in relation to the event switching generating model. Compared to other models such as the leaky integrator, we found a better match between the behavioral response observed and that given by this agent. Furthermore, we were also able to fit the experimental data with different levels of switching volatility in the model and derive a common marker for the inter-variability of participants, by titrating their level of preference between exploration and exploitation. Such results prove that in such an unstable environment, human observers can still effectively represent an internal belief, and use this representation in their sensory-motor control system and for explicit judgments. This work offers an innovative approach to more generically test human cognitive abilities in uncertain and dynamic environments.""",
 YYYY=YYYY, MM=MM, DD=DD,
 tag=tag,
 projects='pace-itn',
 time_start = '11:00:00',
 time_end = '1:30:00',
 url=f'https://laurentperrinet.github.io/talk/{slugify(tag)}',
 sections=['Motivation: Should I stay or should I go?',
          'Methods: Experimental protocol',
          'Results:The Bayesian Changepoint Detector',
          'Results: Matching Behavioral data',
          'Results: Analyzing inter-individual differences',
          ]
)

# https://pythonhosted.org/PyQRCode/rendering.html
# pip3 install pyqrcode
# pip3 install pypng

import pathlib
pathlib.Path(figpath_talk).mkdir(parents=True, exist_ok=True)

figname_qr = os.path.join(figpath_talk, 'qr.png')
if not os.path.isfile(figname_qr):
    import pyqrcode as pq
    code = pq.create(meta['url'])
    code.png(figname_qr, scale=5)

print(meta['sections'])
s = Slides(meta)

# angelo-franciosini/avatar.jpg   james-a.-bednar/avatar.jpg      kiana-mansour-pour/avatar.jpg   nicole-voges/avatar.gif
# anna-montagnini/avatar.jpg      jean-bernard-damasse/avatar.jpg laurent-madelain/avatar.png     rick-a.-adams/avatar.jpg
# frédéric-chavane/avatar.png     jens-kremkow/avatar.jpg         laurent-u-perrinet/avatar.jpeg  victor-boutin/avatar.jpg
# hugo-ladret/avatar.jpg          karl-friston/avatar.jpg         mina-a-khoei/avatar.jpg         wahiba-taouali/avatar.jpg


# figpath_people = os.path.join(home, 'ownCNRS/2019-01_LACONEU/people')
url_people = 'https://laurentperrinet.github.io/authors/'
Karl = s.content_imagelet(os.path.join(url_people, 'karl-friston/avatar.jpg'), height_px)
Rick = s.content_imagelet(os.path.join(url_people, 'rick-a.-adams/avatar.jpg'), height_px)
Anna = s.content_imagelet(os.path.join(url_people, 'anna-montagnini/avatar.jpg'), height_px)
LM = s.content_imagelet(os.path.join(url_people, 'laurent-madelain/avatar.png'), height_px)
JB = s.content_imagelet(os.path.join(url_people, 'jean-bernard-damasse/avatar.jpg'), height_px)
Fredo = s.content_imagelet(os.path.join(url_people, 'frédéric-chavane/avatar.png'), height_px)
Python = s.content_imagelet('https://www.python.org/static/community_logos/python-powered-h-140x182.png', height_px)
s.meta['Acknowledgements'] =f"""
<small>
<h5>Acknowledgements:</h5>
<ul>
    <li>Rick Adams and Karl Friston @ UCL - Wellcome Trust Centre for Neuroimaging</li>
    <li>Jean-Bernard Damasse and Laurent Madelain - ANR REM</li>
    <li>Frédéric Chavane - INT</li>
</ul>
<BR>
{Rick}{Karl}{JB}{LM}{Anna}{Fredo}<a href="https://github.com/laurentperrinet/slides.py">{Python}</a>
<BR>
    This work was supported by the <a href="https://laurentperrinet.github.io/project/pace-itn/">PACE-ITN Project</a>.
</small>

"""
#############################################################################
# 🏄🏄🏄🏄🏄🏄🏄🏄 Should I stay or should I go? 🏄🏄🏄🏄🏄🏄🏄🏄
#############################################################################
i_section = 0
s.open_section()
intro = """
<h2 class="title">{title}</h2>
<h3>{author_link}</h3>
""".format(**meta)
intro += s.content_imagelet('http://laurentperrinet.github.io/slides.py/figures/troislogos.png', s.meta['height']*.2, embed=False) #bgcolor="black",
intro += """
<h4><a href="{conference_url}">{conference}</a>, {DD}/{MM}/{YYYY} </h4>

{Acknowledgements}
""".format(**meta)
#############################################################################
# s.add_slide(content=intro)
#
# s.add_slide(content=s.content_figures(
#     #[os.path.join(figpath_talk, 'qr.png')], bgcolor="black",
#     [os.path.join(figpath_slides, 'mire.png')], bgcolor=meta['bgcolor'],
#     height=s.meta['height']*1.),
#     #image_fname=os.path.join(figpath_aSPEM, 'mire.png'),
#     notes="""
# Check-list:
# -----------
#
# * (before) bring VGA adaptors, AC plug, remote, pointer
# * (avoid distractions) turn off airport, screen-saver, mobile, sound, ... other running applications...
# * (VP) open monitor preferences / calibrate / title page
# * (timer) start up timer
# * (look) @ audience
#
# http://pne.people.si.umich.edu/PDF/howtotalk.pdf
#
#  """)
#
# s.add_slide(content=s.content_figures([figname_qr], cell_bgcolor=meta['bgcolor'], height=s.meta['height']*height_ratio) + '<BR><a href="{url}"> {url} </a>'.format(url=meta['url']),
#             notes="All the material is available online - please flash this QRcode this leads to a page with links to further references and code ")

#############################################################################
#############################################################################
#############################################################################
#############################################################################
#############################################################################
#############################################################################
#############################################################################
#############################################################################

s.add_slide(content=intro,
            notes="""
* (AUTHOR) Hello, I am Laurent Perrinet from the Institute of Neurosciences of
la Timone in Marseille, a joint unit from the CNRS and the AMU

* (OBJECTIVE)


* Let's me first describe the motivation of this work...




Today I will present some recent results on detecting switches in a sequence

In particular, I will try to give some principles of active inference, framed in a practical example of a dynamic, poissbly volatile environment and how we may use it to anticipate ---- and my main goal in general is to illustrate how this theory may give a creative and efficient tool to do psychophysics.



https://laurentperrinet.github.io/talk/2016-10-13-law/
https://laurentperrinet.github.io/talk/2018-02-01-bcp-invibe-fest/
https://laurentperrinet.github.io/talk/2018-04-05-bcp-talk/
https://laurentperrinet.github.io/talk/2019-01-18-laconeu/
https://laurentperrinet.github.io/talk/2019-04-05-bbcp-causal-kickoff/

this talk= https://laurentperrinet.github.io/talk/2019-05-23-neurofrance/

* (SHOW TITLE)

""")
N_seq = 40
import os
import numpy as np
np.random.seed(42)
def create_movie(seq, T=.5, radius=1/64, fps=50, W=1000, H=600, figname='sequence.mp4'):
    import gizeh as gz
    import moviepy.editor as mpy

    r = W*radius
    duration = T * len(seq)
    # print(seq)
    duration_flash = .05
    s = dict(r=r, fill=(1, 1, 1))

    def make_frame(t):
        i = int(t/T)
        dir = seq[i]
        t_ = t - i*T
        # print(i, dir, t_)

        surface = gz.Surface(W, H, bg_color=(0.5, 0.5, 0.5))

        flash = gz.circle(xy=(W/2., H/2.), **s)

        if t_ < duration_flash:
            flash.draw(surface)

        else:
            x = W/2. * (1 - dir*t_)
            rect = gz.circle(xy=(x, H/2.), **s)
            rect.draw(surface)

        text = gz.text(f"{i+1:02d}/{len(seq):02d}", fontfamily="Impact",  fontsize=W/25,
                            h_align="right", v_align="top",
                          fill=(0, 0, 0), xy=(W/8, H/10), angle=0)
        text.draw(surface)
        # print (surface.get_npimage().shape)
        return surface.get_npimage()

    clip = mpy.VideoClip(make_frame, duration=duration)
    clip.write_videofile(figname, fps=fps)
    return 'ok'

# figname='alternation.mp4'
# seq = [-1, 1] * (N_seq//2)
# figname = os.path.join(figpath_talk, figname)
# if not os.path.isfile(figname): create_movie(seq, figname=figname)
#
# s.add_slide(content="""
#     <video loop="1" autoplay="1" controls width=99%/>
#       <source type="video/mp4" src="{}">
#     </video>
#     <BR>
#     """.format(figname), # s.embed_video(figname)),
# notes="""
#
#
#
# """)

figname='block50.mp4'
seq = -1 + 2 * (np.random.rand(N_seq)>.5)
# print(seq)
figname = os.path.join(figpath_talk, figname)
if not os.path.isfile(figname): create_movie(seq, figname=figname)

s.add_slide(video_fname=figname,
# s.add_slide(content="""
#     <video loop="1" autoplay="1" controls width=99%/>
#       <source type="video/mp4" src="{}">
#     </video>
#     <BR>
#     """.format(figname), # s.embed_video(figname)),
notes="""

* typically tested with a regular sequence

* but, We live in a fundamentally volatile world - Think for instance to a medical doctor

* what if a bias

* block 50%

""")

figname = 'block75.mp4'
seq = -1 + 2 * (np.random.rand(N_seq)<.75)
# print(seq)
figname = os.path.join(figpath_talk, figname)
if not os.path.isfile(figname): create_movie(seq, figname=figname)

s.add_slide(video_fname=figname,
# s.add_slide(content="""
#     <video loop="1" autoplay="1" controls width=99%/>
#       <source type="video/mp4" src="{}">
#     </video>
#     <BR>
#     """.format(figname), # s.embed_video(figname)),
notes="""

* block 75% to the right

""")


title = meta['sections'][i_section]

s.add_slide(content=s.content_figures(
[os.path.join(figpath_talk, 'Experiment_classique_4_blocks.png')],
        title=title + ' - Probability bias', height=s.meta['height']*.825),
notes="""



""")


bib =  'Montagnini A, Souto D, and Masson GS (2010) <a href="http://jov.arvojournals.org/article.aspx?articleid=2138664">J Vis (VSS Abstracts) 10(7):554</a>,<BR> Montagnini A, Perrinet L, and Masson GS (2015) <a href="https://arxiv.org/abs/1611.07831">BICV book chapter</a>'

s.add_slide(content=s.content_figures(
[os.path.join(figpath_talk, '1_B_protocol_recording.png')],
        title=title + ' - Eye Movements', height=s.meta['height']*.825) + bib,
notes="""
* This was shown to happen in the more  experimental setting While recording smooth pursuit eye movements : Anna Montagnini has previously shown that if you use a probabilistic bias in the direction of the movement of the target, the the eye will (uncousciously) anticipate in the direction of this bias.

* this protocol used a random length fixation period then a pause of fixed duration, and then a traget moving at 15 deg / s

* the value p gives the probability of going to the right : at .5 it is unbisaed, and at .75 for instance it goes 75% to the right and 25% to the left
""")

for txt in ['1', '2']:
    s.add_slide(content=s.content_figures(
[os.path.join(figpath_talk, 'image_anna_' + txt + '.png')],
            title=title + ' - Eye Movements', height=s.meta['height']*.825),
   notes="""

* we show in this plot the velocity of the eye that she recorded when tracking the target.

* we observe that as you depart from .5, there is an anticipatory component to the SPEM

* Moreover, she has proved that this behaviour is progressive and increases with the value of p

""")
#
# for tag, notes_ in zip(['Experiment_randomblock_', 'Experiment_classique_', 'Experiment_randomblock_'], ["""
# * our initial goal was to extend these results to a more volatile environment. This Is well described by a three layered architecture decribing the evolution of outcomes (left or right) as a function of the trial number in an experimental block:
#
# - at the bottom we have switches, that is moments were we know that there was a change in context:
#
# - then we have an intermediate layer which describes this context as the probability p which defines the bias towards one direction. as switches happen at random times but with a given hazard rate, blocks are of average length of here 40.
#
# - finally, we draw the directions as a sequences of binary events following this bernouilli trials
#
# If we draw another exmple of this generative model
# ""","""
#
# This has to be put in contrast with a more classical protocol such as that used in the previously described experiment where different blocks of fixed length were drawn, but with different probabilities.
#
# ""","""
#
# Here, we
# """]):
#     for txt in [str(i) for i in range(4)[::-1]]:
#         s.add_slide(content=s.content_figures(
#     [os.path.join(figpath_aSPEM, tag + txt + '.png')],
#                 title=title + ' - Eye Movements', height=s.meta['height']*.775),
#        notes=notes_)

figname='randomblock.mp4'
seq = -1 + 2 * (np.random.rand(N_seq)<.75)
seq[N_seq//2:] = -1 + 2 * (np.random.rand(N_seq//2)>.75)
# print(seq)
figname = os.path.join(figpath_talk, figname)
if not os.path.isfile(figname): create_movie(seq, figname=figname)


s.add_slide(video_fname=figname,
# s.add_slide(content="""
#     <video loop="1" autoplay="1" controls width=99%/>
#       <source type="video/mp4" src="{}">
#     </video>
#     <BR>
#     """.format(figname), # s.embed_video(figname)),
notes="""

* random block first half 75% to the right, then symetric

""")


s.add_slide(content=s.content_figures(
[os.path.join(figpath_talk, 'Experiment_block_0.svg')],
        title=title + ' - Switching model', height=s.meta['height']*.825),
notes="""

In summary, the design of our experimental setting is therefore very similar to the previous experiment but with a more general construct:

- using the same 3-layered generative model, we generated sequences of directions

- and generated 3 blocks of 200 trials

- with an average block length of 40 trials

We anticipated that such an  experiment for which we simply recordedd the eye movements should be more difficult for observers compared to the classical experiments with longer (400 trials), fixed blocks and...


""")

s.close_section()

i_section = 1
#############################################################################
# 🏄🏄🏄🏄🏄🏄🏄🏄     The Experimental protocol             🏄🏄🏄🏄🏄🏄🏄🏄
#############################################################################
#############################################################################

s.open_section()
title = meta['sections'][i_section]
s.add_slide_outline(i_section,
notes="""
Indeed, these raw psycholophysical results are encouraging but since we used a generative model for generating the sequence, let's see if we can build a Bayesian model which would be optimal wrt to this generative model.

Indeed, such a model already exists, the onlin BCP, and we will adapt it for our specific setting.
""")


url =  'full code @ <a href="https://github.com/chloepasturel/AnticipatorySPEM">github.com/chloepasturel/AnticipatorySPEM</a>'

s.add_slide(content=s.content_figures(
[os.path.join(figpath_talk, 'Experiment_block_0.svg')],
        title=title + ' - Switching model', height=s.meta['height']*.825) + url,
notes="""

* the whole experiment was coded by Chloé using :
- python for the generative model,
- the psychopy library for the stimulus display + connection to the eyelink 1000 that we used to record EMs
- numpy, pandas and pylab for the data analysis

* all this code is available : for running the experiments, re-analyzing the data and doing all plots are on github


Let's now have a look at the raw psychophysical results..

""")


s.add_slide(content=s.content_figures(
[os.path.join(figpath_talk, '1_C_protocol_bet.png')],
        title=title + ' - Rating scale', height=s.meta['height']*.825),
notes="""

This is why we added a supplementary experiment for each observer but on a different day for which we asked at every trial to give a subjective, conscious evaluation of the direction of the next trial + a confidence about this inference. Once this information given by the subject, we were showing the actual outcome.

Interestingly, we used exactly the same sequence, allowing to make a direc comparison of the results of both experiments

We called this experiment the bet experiment.


""")

s.add_slide(content=s.content_figures(
[os.path.join(figpath_talk, 'Experiment_block_0_bet.svg')],
        title=title + ' - Rating scale', height=s.meta['height']*.825) + url,
notes="""
First, we overlay the results of the bet result for one of the 12 subjects

We rescaled th value given by the observer so that it fits to 0 (sure it goes left) to 1 (sure it goes right)

We observe a pretty good fit of this trace as a function of trial number


In particular, we see 2 main effects:
- results are more variable when the bias is around .5 than when it is high (close to zero or one)
- switches were detected quite rapidly but with a certain delay of a few trials. Indeed, note that this plot shows the entire sequence but that observers had only access to some internal representation of the memory of the previous observations. When faced with some new observations, the observer has to constantly adapt his response to either exploit this information by considering that this observation belongs to the same sub-block or to explore a novel hypothesis. This compromise is one of the crucial component that we wish to explore.

Let's now have a look at EMs...

""")


# for txt in ['results_pari', 'results_enregistrement']:
#     s.add_slide(content=s.content_figures(
# [os.path.join(figpath_GDR, txt + '.png')],
#             title=title, height=s.meta['height']*.825),
#    notes="""
#
# """)


for txt in ['raw_trace', 'raw_fitted']: # 'raw_fit',
    s.add_slide(content=s.content_figures(
[os.path.join(figpath_talk, txt + '.svg')],
            title=title + ' - Fitting eye movements', height=s.meta['height']*.825) + url,
   notes="""

I show here a typical velocity traces for one subject / 2 trials

- x-axis is time in milliseconds aligned on target onset, and we show respectively from left to right the fixation in gray, the GAP in pink (300ms) and the run in light gray.

- y-axis is the velocity as computed as the gradient of position. Remark that the eyelink provides with the periods of saccades or blinks that we removed from the signal. it is quite noisy and to complement existing signal processing methods, Chloe implemented a robust

- fitting method which allows to extract some key components of the velocity traces: maximum speed, latency, temporal inertia ($\tau$) and most interestingly acceleration before motion onset. We cross-validated that this method was givinfg similar results to other classical methods but in a more robust fashion/

While being sensible to recording errors, this allows us to extract the anticipatory component of SPEMs and..

""")

s.add_slide(content=s.content_figures(
[os.path.join(figpath_talk, 'Experiment_block_0_EM.svg')],
        title=title + ' - Eye Movements', height=s.meta['height']*.825) + url,
notes="""

* I show here the overlay of this variable on the plot of probability biases

* these accelarations values were here scaled according to their extremal values.

* there seems to be a trend with the polarity of the acceleration being negative for p values below .5 and positive for values above .5



""")

s.add_slide(content=s.content_figures(
[os.path.join(figpath_talk, 'Experiment_block_0_bet_EM.svg')],
        title=title + ' - Behaviour', height=s.meta['height']*.825) + url,
notes="""

... to make this clearer, and because we used the same sequence, we can overlay the results of both experiments in one plot:

which qualitatively confirms such an intuition...

""")

for txt in ['scatter_velocity_sigmo_real']:#, 'p_bet--v_a']: # TODO : make a sequence to uncover parts
    s.add_slide(content=s.content_figures(
[os.path.join(figpath_talk, txt + '.svg')],
            title=title, height=s.meta['height']*.75) + url,
   notes="""
* quantitatively, one can now plot the results for all subjects

* the x-axis corresponds to the probability that was coded at the second layer and which is unknown to the observer
* the y -axis shows either the bet or the
* dots represent single responses - the saturation giving the identity of the observer

we notice a quite nice linear correlation (black line) for both experiments; of the order of that found in the classical experiment with fixed blocks and a vartiety of bias values. This is surprising as the blocks are of random length, observer can still adapt to such a volatile environment.

Another visualization, the scatter plot of acceleration  as a function of probability bet shows also that there is a correlation between both  variables.

This allows to make a first point: it is possible to use more genreal models such as hierarchical generative models.

However, while this results seem encouraging, a more finer analysis may be necessary.

""")

s.close_section()

i_section = 2
#############################################################################
# 🏄🏄🏄🏄🏄🏄🏄🏄 The Bayesian Changepoint Detector  🏄🏄🏄🏄🏄🏄🏄🏄
#############################################################################
#############################################################################

s.open_section()
title = meta['sections'][i_section]
s.add_slide_outline(i_section,
notes="""
Indeed, these raw psycholophysical results are encouraging but since we used a generative model for generating the sequence, let's see if we can build a Bayesian model which would be optimal wrt to this generative model.

Indeed, such a model already exists, the onlin BCP, and we will adapt it for our specific setting.
""")


s.add_slide(content=s.content_figures(
[os.path.join(figpath_talk, 'Experiment_block_0.svg')],
        title=title + ' - Switching model', height=s.meta['height']*.825),# + url,
notes="""
Let's remember our hierarchical generative model.

At any given trial, we wish to construct an algorithm which

We will introduce a fundamental component of Bayesian models : a latent variable

this new variable will be used to test different hypothesis which will be evaluated to predict future states. it is called latent because it aims at representing a variable that is latent (or hidden) to the observer

in our case, we will assume that the bayesian model knows about the structure of the generative model and we will set it to the current run-length $r$, that is, at any given trial, the hypothesis that the past r observations belong to the same block. of course a wrong choice of a latent variables (let's say the temperture in the experimental room) may give unexpected results, even is the bayesian model is "optimal" - an essential point to understand in bayesian inference

""")

s.add_slide(content="""
Bayesian Online Changepoint Detector
------------------------------------

* an implementation of
[Adams &amp; MacKay 2007 "Bayesian Online Changepoint Detection"](http://arxiv.org/abs/0710.3742)
in Python.

````
@TECHREPORT{ adams-mackay-2007,
AUTHOR = "Ryan Prescott Adams and David J.C. MacKay",
TITLE  = "Bayesian Online Changepoint Detection",
INSTITUTION = "University of Cambridge",
ADDRESS = "Cambridge, UK",
YEAR = "2007",
NOTE = "arXiv:0710.3742v1 [stat.ML]",
URL = "http://arxiv.org/abs/0710.3742"
}
````

* adapted from https://github.com/JackKelly/bayesianchangepoint by Jack Kelly (2013) for a binomial input.

* This code is based on the  [MATLAB implementation](http://www.inference.phy.cam.ac.uk/rpa23/changepoint.php) provided by Ryan Adam. Was available at http://hips.seas.harvard.edu/content/bayesian-online-changepoint-detection

* full code @ https://github.com/laurentperrinet/bayesianchangepoint

""", notes='<br><a href="http://arxiv.org/abs/0710.3742">Adams &amp; MacKay 2007 "Bayesian Online Changepoint Detection</a>', md=True)

tag = 'bcp_model_layer_' #  'model_bcp_'
blobs = ["""
Initialize $P(r_0=0)=1$ and  $ν^{(0)}_1 = ν_{prior}$ and $χ^{(0)}_1 = χ_{prior}$
""","""
Observe New Datum $x_t$  and   Perform Prediction $P (x_{t+1} | x_{1:t}) =   P (x_{t+1}|x_{1:t} , r_t) \cdot P (r_t|x_{1:t})$
""","""
Evaluate (likelihood) Predictive Probability $π_{1:t} = P(x_t |ν^{(r)}_t,χ^{(r)}_t)$
<br>
Calculate Growth Probabilities $P(r_t=r_{t-1}+1, x_{1:t}) = P(r_{t-1}, x_{1:t-1}) \cdot π^{(r)}_t \cdot (1−h))$
<br>
<font color="FF0000">Calculate Changepoint Probabilities $P(r_t=0, x_{1:t})= \sum_{r_{t-1}} P(r_{t-1}, x_{1:t-1}) \cdot π^{(r)}_t \cdot h$
</font>""","""
Calculate Evidence $P(x_{1:t}) = \sum_{r_{t-1}} P (r_t, x_{1:t})$
<br>
Determine Run Length Distribution $P (r_t | x_{1:t}) = P (r_t, x_{1:t})/P (x_{1:t}) $
""","""
Update Sufficient Statistics :
<br>
$ν^{(r+1)}_{t+1} = ν^{(r)}_{t} +1$, $χ^{(r+1)}_{t+1} = χ^{(r)}_{t} + u(x_t)$
<br>
<font color="FF0000"> $ν^{(0)}_{t+1} = ν_{prior}$, $χ^{(0)}_{t+1} = χ_{prior}$</font>

"""]
for txt, blob, notes_ in zip([str(i) for i in range(1, 6)], blobs, ["""
* in this graph information will be represented at different nodes. each node represent a belief which takes the form of a probability distribution over the set of parameters that we wish to describe.
* it can be the mean and variance of a gaussain, but in general it will be 2 parameters. in our case, we wish to estimate p (between zero and one) - it is characterized by the beta distribution (mathematically it is the conjugate of the bernouilli distribution)
* (mathematically, we will use th family of exponenetial distributions:, gaussians, binomials) among which the beta distribution belongs
First, we initialize the first node to prior values
* at trial zero, there is no information, so we intiialize to the prior values
""","""

""","""

""","""

""","""

"""]):
    s.add_slide(content=s.content_figures(
[os.path.join(figpath_talk, tag + txt + '.png')],
            title=title, height=s.meta['height']*.775)+blob,
   notes=notes_)

# https://raw.githubusercontent.com/laurentperrinet/bayesianchangepoint/master/README.md

s.add_slide(content="""
<h2>Bayesian Changepoint Detector</h2>

<ol>
<li> Initialize
</li>
 <ul>
  <li>
   $P(r_0=0)=1$ and
 </li>
  <li>
   $ν^{(0)}_1 = ν_{prior}$ and $χ^{(0)}_1 = χ_{prior}$
 </li>
  </ul>
 <li>
 Observe New Datum $x_t$
</li>
 <li>
  Evaluate Predictive Probability $π_{1:t} = P(x_t |ν^{(r)}_t,χ^{(r)}_t)$
</li>
 <li>
  Calculate Growth Probabilities $P(r_t=r_{t-1}+1, x_{1:t}) = P(r_{t-1}, x_{1:t-1}) \cdot π^{(r)}_t \cdot (1−H(r^{(r)}_{t-1}))$
</li>
 <li>
  Calculate Changepoint Probabilities $P(r_t=0, x_{1:t})= \sum_{r_{t-1}} P(r_{t-1}, x_{1:t-1}) \cdot π^{(r)}_t \cdot H(r^{(r)}_{t-1})$
</li>
 <li>
  Calculate Evidence $P(x_{1:t}) = \sum_{r_{t-1}} P (r_t, x_{1:t})$
</li>
 <li>
  Determine Run Length Distribution $P (r_t | x_{1:t}) = P (r_t, x_{1:t})/P (x_{1:t}) $
</li>
 <li>
  Update Sufficient Statistics :
</li>
 <ul>
  <li>
   $ν^{(0)}_{t+1} = ν_{prior}$, $χ^{(0)}_{t+1} = χ_{prior}$
 </li>
  <li>
   $ν^{(r+1)}_{t+1} = ν^{(r)}_{t} +1$, $χ^{(r+1)}_{t+1} = χ^{(r)}_{t} + u(x_t)$
 </li>
  </ul>
 <li>
  Perform Prediction $P (x_{t+1} | x_{1:t}) =   P (x_{t+1}|x_{1:t} , r_t) \cdot P (r_t|x_{1:t})$
</li>
 <li>
  go to (2)
</li>
 </ol>
        """, notes='', md=False)

url =  'full code @ <a href="https://github.com/laurentperrinet/bayesianchangepoint">github.com/laurentperrinet/bayesianchangepoint</a>'

s.add_slide(content=s.content_figures(
[os.path.join(figpath_talk, 'github.png')],
        title=title, height=s.meta['height']*.825) + url,
notes="""

""")
#
# modes = ['expectation', 'fixed', ] # 'max',  'expectation',#for mode in ['expectation']:#, 'max']:# for mode in ['expectation', 'max']:
# for mode, mode_txt in zip(['expectation', 'fixed', ], [' - Full model', ' - Fixed window', ]):
#
#     figname = os.path.join(figpath_talk, 'bayesianchangepoint_' + mode + '.png')
#     if not os.path.isfile(figname):
#         print('Doing ', figname)
#         T = 400
#         p_gen = .25 * np.ones(T)
#         p_gen[100:300] = .75
#         np.random.seed(2018)
#         o = 1 * (p_gen > np.random.rand(T))
#
#         p_bar, r_bar, beliefs = bcp.inference(o, h=1/200, p0=.5)
#         # fig, axs = bcp.plot_inference(o, p_gen, p_bar, r_bar, beliefs, max_run_length=250, fixed_window_size=200, mode=mode, fig_width=fig_width)
#
#         if mode == 'fixed':
#             fig, axs = bcp.plot_inference(o, p_gen, p_bar, r_bar, beliefs, max_run_length=250, fixed_window_size=40, mode=mode, fig_width=fig_width)
#         else:
#             fig, axs = bcp.plot_inference(o, p_gen, p_bar, r_bar, beliefs, max_run_length=250, mode=mode, fig_width=fig_width)
#         fig.savefig(figname, dpi=400)
#
#     s.add_slide(content=s.content_figures([figname],
#                 title=title + mode_txt, height=s.meta['height']*.825) + url,
#        notes="""
# Let's now see the application of our model to a simple synthetic example before applying it to the experimental protocol that we used in our two experiments
#
#
# - we show two panels, one below which displays the value of the belief for the different run-length, and one above where we will show the resaulting prediction of the next outcome.
# we obtain for any given sequence different values at the given trial in the form of columns for any possible run-length: the belief,
# and the sufficient statistics for the beta distirbution which allow to provide with an estimate of the current probability
#
# - first, we show the value of probability, low probabilities are blueish while high probabilities. at every trial, the agent evluates the value for the different possible run lengths, generating a column. by showing all columns we generate this image which shows the evaluation along the sequence of trials.
#
# - second we show above the sequence of observations that were shown to the agent in a light black line. the read line gives an evaluation of the most probable a posteriori probability as the probability to hte run-length the maximum a posteriori belief on the differettn beliefs about run-lengths. using the estimate of the precision at this
#
# We remark two main observations:
#
# - first, beliefs grow at the beginning along a linear ridge, as we begin our model by assuming there was a switch at time 0. Then we observe that at a switch (hidden to the model), the model
# such that belief is more stronlgly diffused until the probability
#
# -second, we may use this information to read-out the information the most probable probability and the confidence interval as shown by the red dashed lines (.05, and .95)
#
#
# This is in contrast with a fixed length model, for which
# - the delay will always be similar
# - there is no dynamic upDATE OF THE INFERRD probability
#
#
# as a summary, for any given sequnce, we get an estimate of the probability given by the ideal observer. we will now see how we can apply that to our experiemntal protocol.
# """)
#
# for mode, mode_txt in zip(['expectation', 'fixed'], [' - Full model', ' - Fixed window', ]):
#     import bayesianchangepoint as bcp
#     for i_block in range(3):
#         figname = os.path.join(figpath_talk, 'bayesianchangepoint_exp_' + mode + '_' + str(i_block) + '.png')
#         if not os.path.isfile(figname):
#             print("Doing ", figname)
#
#             seed = 42
#             np.random.seed(seed)
#             #N_time = 1000
#             #N_trials = 4
#
#             tau = 25.
#             N_blocks = 3 # 4 blocks avant
#             seed = 51 #119 #2017
#             N_trials = 200
#             tau = N_trials/5.
#             (trials, p) = bcp.switching_binomial_motion(N_trials=N_trials, N_blocks=N_blocks, tau=tau, seed=seed)
#
#             h = 1./tau # a.exp['tau']
#             print('this experiment uses', N_trials, 'trials and a switch rate of h=', h, '(that is, one switch every', 1/h, 'trials on average)')
#             print('i_block=', i_block)
#             o = p[:, i_block, 0]
#             p_bar, r_bar, beliefs = bcp.inference(o, h=h, p0=.5)
#             if mode == 'fixed':
#                 fig, axs = bcp.plot_inference(p[:, i_block, 0], p[:, i_block, 1], p_bar, r_bar, beliefs, max_run_length=200, fixed_window_size=40, mode=mode, fig_width=fig_width)
#             else:
#                 fig, axs = bcp.plot_inference(p[:, i_block, 0], p[:, i_block, 1], p_bar, r_bar, beliefs, max_run_length=200, mode=mode, fig_width=fig_width)
#             fig.savefig(figname, dpi=400)
#
#         s.add_slide(content=s.content_figures([figname],
#                     title=title +  mode_txt, #' - inference with BCP',
#                     height=s.meta['height']*.825) + url,
#            notes="""
#            Let's use our model on the different sequences that were generated in our experiments in the different blocks.
#
# we the same arrangement of panels, we show below the dynamical evolution of beliefs and above the resulting readout from the model
#
# we see that as in the synthetic example above, there is a correct detection of switch after a short delay of a few trials
#
# in particular, from this correct detection, the value of the inferred probability approaches the true one as the number of observations increase in one subblock.
#
# again, we see that a fixed length model gives a similar output but with the two disadvantages described above
#
# Let's now see how this applies to our experimental results by comparing human observers to our bayesian agent.
# """)


for fname in ['BCP_mean']:
    s.add_slide(content=s.content_figures(
    [os.path.join(figpath_talk, fname + '.svg')],
            title=title + ' - Compiling results', height=s.meta['height']*.825),
    notes="""



    """)


s.close_section()

i_section = 3
#############################################################################
# 🏄🏄🏄🏄🏄🏄🏄🏄 Match with data  🏄🏄🏄🏄🏄🏄🏄🏄
#############################################################################
#############################################################################
s.open_section()
title = meta['sections'][i_section]
s.add_slide_outline(i_section,
notes="""

""")

# tag = 'bayesianchangepoint_'
# s.add_slide(content=s.content_figures(
#     [os.path.join(figpath_aSPEM, txt + '.png') for txt in [tag + 'm', tag + 'e']],
#             title=title, height=s.meta['height']*.825, transpose=True, fragment=True),
#    notes="""
#
# """)

# tag = 'results_bayesianchangepoint_'
# for txt in [tag + 'm', tag + 'e']:
#     s.add_slide(content=s.content_figures(
#         [os.path.join(figpath_ms, txt + '.png')],
#         # [os.path.join(figpath_aSPEM, txt + '.png')],
#                 title=title, height=s.meta['height']*.825),
#        notes="""
#
#     """)

# /Users/laurentperrinet/pool/AME_is_AnalyseModelisationExperimentation_Chloe/AnticipatorySPEM/figures/Result/Results_BCP_velocity_sujet_10.svg


s.add_slide(content=s.content_figures(
[os.path.join(figpath_talk, 'BCP_mean.svg')],
        title=title + ' - Compiling results', height=s.meta['height']*.825),
notes="""



""")


tag = 'Results_BCP_position'
tag = 'Results_BCP_velocity'
tag = 'Results_BCP_velocity_sigmo'
for txt in [str(i) for i in range(2)]:# [6, 10, 5, 2]]:
    s.add_slide(content=s.content_figures(
        [os.path.join(figpath_talk, tag + '_' + txt + '.svg')],
                title=title +  ' - fit with BCP', height=s.meta['height']*.825),
       notes="""
Among our 12 subjects, we show four representative examples. we will use the same figure as in the section with raw results

but we superposed to our 2 variables, the value of the readout inferred probability along with the confidence interval.

compared to the raw results which were using the true (hidden) probability, it seems qualitatively that it follows well the traces observed experimenetally
- first, both have similar delays in ddetercting a switch, reflecting the diuffusion of probability
- second, precisions seems to increase in bigger sub-blocks as a function of the inferred run-length

as a result, the inferred probability as a function of time constitutes a useful regressor
    """)

for txt in ['scatter_velocity_sigmo_real', 'scatter_velocity_sigmo_fixed', 'scatter_velocity_sigmo_mean']: # TODO : make a sequence to uncover parts
    s.add_slide(content=s.content_figures(
[os.path.join(figpath_talk, txt + '.svg')],
            title=title, height=s.meta['height']*.75) + url,
   notes="""

We may finally wrap up results and the model and plot

scatters plots are visually misleading as they do not show well the density of data points



""")

tag = 'KDE'
for mode, mode_txt in zip(['fixed', 'mean_fixed'], [' - Fixed window', ' - Full model']):
# for mode in ['fixed', 'expectation']: #, 'max', modes: #
    s.add_slide(content=s.content_figures(
        [os.path.join(figpath_talk, tag + '_' +  session + '_' + mode + '.svg') for session in ['bet', 'velo']],
                title=title + mode_txt, height=s.meta['height']*.7, transpose=False, fragment=False),
       notes="""
we therefore used a kernel density estimation which clearly show the relationship between the agent probability and that reported by human observers
- on the right, we

to summarize, we have shown that
- there is a correlation in the anticiapatory response of eye movements in a volatile environment that is captured if we know the true probability
- that a fixed length models captures some of this correlation, but that
- our online bayesian changep[oint model better captures this correlation and that this may hint at the neural mechanisms used to anticipate in a dynamic environment

the brain is not strongly a bayesian machine, but weakly


    """)


s.close_section()

i_section = 4
#############################################################################
# 🏄🏄🏄🏄🏄🏄🏄🏄 Application to interindividual differences? 🏄🏄🏄🏄🏄🏄🏄🏄
#############################################################################
s.open_section()
title = meta['sections'][i_section]
s.add_slide_outline(i_section,
notes="""


""")

s.add_slide(content=s.content_figures(
    [os.path.join(figpath_talk, '5_inter-individual_differences_fit.png')],
            title=title + 'interindividual differences', height=s.meta['height']*.85, transpose=False, fragment=True),
   notes="""

- interindividual differences

""")

s.close_section()

#############################################################################
# 🏄🏄🏄🏄🏄🏄🏄🏄 OUTRO - 5''  🏄🏄🏄🏄🏄🏄🏄🏄
#############################################################################
#############################################################################
s.open_section()
s.add_slide(content=intro,
            notes="""
perspectives:
- active inference
- RL

* Thanks for your attention!
""")


s.add_slide(content=s.content_figures([figname_qr], cell_bgcolor=meta['bgcolor'], height=s.meta['height']*height_ratio) + '<BR><a href="{url}"> {url} </a>'.format(url=meta['url']),
            notes="All the material is available online - please flash this code this leads to a page with links to further references and code ")

s.close_section()

#############################################################################
#############################################################################
#############################################################################
#############################################################################
#############################################################################
#############################################################################
#############################################################################
if slides_filename is None:
    with open("README.md", "w") as text_file:
        text_file.write("""\
# {title}

* What:: talk @ [conference](conference_url)
* Who:: {author}
* Where: {location}, see {url}
* When: {DD:02d}/{MM:02d}/{YYYY}, time: {time_start}-{time_end}

* What:
  * Slides @ https://laurentperrinet.github.io/{tag}
  * Code for slides @ https://github.com/laurentperrinet/{tag}/
  * Abstract: {abstract}

""".format(**meta))

    with open("/tmp/talk.bib", "w") as text_file:
        text_file.write("""\
@inproceedings{{{tag},
    Author = "{author}",
    Booktitle = "{conference}",
    Title = "{title}",
    Abstract = "{abstract}",
    Url = "{url}",
    Year = "{YYYY}",
    Date = "{YYYY}-{MM:02d}-{DD:02d}",
    location = "{location}",
    projects = "{projects}",
    time_start = "{YYYY}-{MM:02d}-{DD:02d}T{time_start}",
    time_start = "{YYYY}-{MM:02d}-{DD:02d}T{time_end}",
    url = "{url}",
    url_slides = "https://laurentperrinet.github.io/{tag}",
    url_code = "https://github.com/laurentperrinet/{tag}/",
}}

""".format(**meta))

else:
    s.compile(filename=slides_filename)

# Check-list:
# -----------
#
# * (before) bring miniDVI adaptors, AC plug, remote, pointer
# * (avoid distractions) turn off airport, screen-saver, mobile, sound, ... other running applications...
# * (VP) open monitor preferences / calibrate / title page
# * (timer) start up timer
# * (look) @ audience
#
# Preparing Effective Presentations
# ---------------------------------
#
# Clear Purpose - An effective image should have a main point and not be just a collection of available data. If the central theme of the image isn't identified readily, improve the paper by revising or deleting the image.
#
# Readily Understood - The main point should catch the attention of the audience immediately. When trying to figure out the image, audience members aren't fully paying attention to the speaker - try to minimize this.
#
# Simple Format - With a simple, uncluttered format, the image is easy to design and directs audience attention to the main point.
#
# Free of Nonessential Information - If information doesn't directly support the main point of the image, reserve this content for questions.
#
# Digestible - Excess information can confuse the audience. With an average of seven images in a 10-minute paper, roughly one minute is available per image. Restrict information to what is extemporaneously explainable to the uninitiated in the allowed length of time - reading prepared text quickly is a poor substitute for editing.
#
# Unified - An image is most effective when information is organized around a single central theme and tells a unified story.
#
# Graphic Format - In graphs, qualitative relationships are emphasized at the expense of precise numerical values, while in tables, the reverse is true. If a qualitative statement, such as "Flow rate increased markedly immediately after stimulation," is the main point of the image, the purpose is better served with a graphic format. A good place for detailed, tabular data is in an image or two held in reserve in case of questions.
#
# Designed for the Current Oral Paper - Avoid complex data tables irrelevant to the current paper. The audience cares about evidence and conclusions directly related to the subject of the paper - not how much work was done.
#
# Experimental - There is no time in a 10-minute paper to teach standard technology. Unless the paper directly examines this technology, only mention what is necessary to develop the theme.
#
# Visual Contrast - Contrasts in brightness and tone between illustrations and backgrounds improves legibility. The best color combinations include white letters on medium blue, or black on yellow. Never use black letters on a dark background. Many people are red/green color blind - avoid using red and green next to each other.
#
# Integrated with Verbal Text - Images should support the verbal text and not merely display numbers. Conversely, verbal text should lay a proper foundation for each image. As each image is shown, give the audience a brief opportunity to become oriented before proceeding. If you will refer to the same image several times during your presentation, duplicate images.
#
# Clear Train of Thought - Ideas developed in the paper and supported by the images should flow smoothly in a logical sequence, without wandering to irrelevant asides or bogging down in detail. Everything presented verbally or visually should have a clear role supporting the paper's central thesis.
#
# Rights to Use Material - Before using any text, image, or other material, make sure that you have the rights to use it. Complex laws and social rules govern how much of someone's work you can reproduce in a presentation. Ignorance is no defense. Check that you are not infringing on copyright or other laws or on the customs of academic discourse when using material.
#
# http://pne.people.si.umich.edu/PDF/howtotalk.pdf
#
