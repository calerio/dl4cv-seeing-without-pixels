# FAQ

Simple answers first; links go to the detail. Words in the [glossary](glossary.md) are not explained again here.

## The paper

**What does the paper do, in one sentence?**
It shows that the camera's movement alone, with no video pixels, says a lot about what a person is doing: a small
model reading only the camera path can pick the right description of the activity well above chance.

**How can camera movement tell you what someone is doing?**
Each activity moves the camera in its own way. Dribbling a basketball shakes a worn camera rhythmically; cooking
keeps it fairly still and pointed down at the counter; climbing moves it up in bursts. The model learns these
patterns from many examples.

**How good is it?**
On the Ego-Exo4D test it picks the right caption out of five 44.8% of the time (chance is 20%). Sports and dance are
easier (about 51%) than step-by-step tasks like cooking (about 34%).

## Our project

**What are we adding?**
Three experiments, each aimed at a weakness the authors name themselves:
1. Training with harder wrong answers, to do better on step-by-step tasks.
2. One model for both first-person and third-person video, instead of two separate ones.
3. Adding where the person is looking (eye gaze) as a second input.

**Did we reproduce the paper?**
Yes, exactly: 44.81%, the same as their Table 2, from their released model and our code
([`results/reproduction.md`](../../results/reproduction.md)).

**Why don't we train from scratch?**
It would take days per run, and the course advises fine-tuning a trained model. Every experiment starts from the
authors' model and trains a little more.

## Experiment 1: harder wrong answers

**What was the idea?**
In the test, the five options all come from the same video, so the model must tell apart, say, "cuts an onion" from
"stirs the pot" in the same kitchen. In training, the wrong answers were random clips from other videos, which are
much easier to rule out. We trained it on same-video wrong answers too.

**Did it work?**
Yes. On step-by-step tasks accuracy went from 33.4% to 36.9%, and overall from 45.0% to 47.3%. It held in all three
training runs, and sports/dance improved a little too ([status](status.md)).

**Why train it three times?**
Training involves randomness. One good run could be luck; the same gain in three runs is not.

**Why did the first attempt do nothing?**
Simply putting same-video clips in a batch made them about 3% of what the model learns from, too little to matter. An
extra loss term that focuses on them (the within-take loss) is what made the difference.

**Why does "a wrong answer must have a different main verb" matter?**
A video can contain the same step twice ("picks up the knife"). Without this rule we would punish the model for
matching a clip to a caption that is actually also correct. The paper's test uses the same rule.

## Experiment 2: one model for both viewpoints

**What are we testing?**
Whether one model can learn first-person and third-person camera paths together and do as well as the two separate
specialist models, and whether telling it which kind of video it sees (the domain token) helps.

**Why is that not trivial?**
The two kinds of data differ: different frame rates, different camera conventions (which axis points up), different
scales, and first-person data knows which way gravity points while third-person does not. We measured and aligned
each of these before training ([`docs/ego_exo.md`](../ego_exo.md)).

**What do the run names mean?**
- **S-ego / S-exo:** the two specialists, each trained only on its own kind of video.
- **U0 / U1:** one joint model, without / with the domain token.
- **E→X:** the first-person model trained further on third-person data only; it isolates the effect of sharing.

## Method and rigour

**Why all the rules (pull requests, reviews, regression check)?**
Research code rarely crashes when it is wrong; it quietly gives a slightly wrong number. The rules catch that before
it ends up in the report: every change is reviewed, and every code change must still reproduce the paper's 44.81%.

**What does "the rule was fixed before the results" mean, and why does it matter?**
Before a run, we write down what result would count as success. Otherwise it is too easy to look at whatever came out
and call it a success. A professor will ask; the commit times in git prove the order.

**What is a confidence interval, quickly?**
A range for the true effect. "+3.5 points, 95% interval +2.4 to +4.8" means we are confident the real gain is
somewhere between those, and in particular above zero.

## Practical

**Can I run things myself?**
The public parts, yes: the DynPose data, the released models and the paper's code run on a laptop for small things
like plots. Training and anything using Ego-Exo4D runs on Valerio's cluster account, because of the dataset licence.

**Where do I start reading?**
This wiki, then the repo's `README.md`, then `LOG.md` (what happened each day), then `docs/` for the designs.

**Can I ask questions about the code?**
Yes: write `@claude` followed by your question in a comment on any pull request or issue on GitHub.
