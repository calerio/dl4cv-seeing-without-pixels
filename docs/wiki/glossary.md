# Glossary

In alphabetical order. A word in *italics* has its own entry.

**Batch.** The group of training examples the model looks at in one step: here 1,024 clip–caption pairs.

**Bootstrap.** A way to put a range around a result: re-draw the test questions (or training runs) at random many
times, recompute the result each time, and see how much it moves. Our *confidence interval*s come from it.

**CamFormer.** The paper's model. A small *transformer* (0.3 million numbers) that reads only how the camera moved,
never the video's pixels, and turns it into a vector that can be compared with a sentence.

**Checkpoint.** A saved copy of a model's weights. "The released checkpoint" means the authors' trained model.

**CLIP.** A model by OpenAI that turns sentences (and images) into vectors. CamFormer uses CLIP's text part, frozen
(never changed), as the target its trajectory vectors must match.

**Confidence interval (95%).** A range that, if we repeated the whole experiment, would contain the true value 95%
of the time. If the range for "B minus A" is entirely above zero, B is better than A beyond reasonable doubt.

**Contrastive learning.** Training by comparison: the model must pick the right caption for a clip among many wrong
ones (*negatives*). It is rewarded for scoring the right pair higher than the wrong ones.

**DynPose-100K.** A dataset of ordinary internet videos (filmed by an observer, *exo*) with camera paths estimated
from the video. Public.

**Domain token.** One extra input we add that tells the model "this trajectory is first-person" or "third-person".
Used in experiment 2.

**Ego / egocentric.** Filmed by a camera the person wears, e.g. smart glasses. The camera moves with their head.

**Ego-Exo4D.** A large dataset from Meta with ego and exo video of people cooking, repairing bikes, playing sports,
etc. Its licence is personal, which is why only Valerio's account uses it.

**Epoch.** One pass through the whole training set. Ego-Exo4D: 155 *batch*es, about 9 minutes on one GPU.

**Exo / exocentric.** Filmed by someone or something watching, e.g. a phone or a tripod.

**Fine-tuning.** Continuing to train an already trained model a little, instead of training from scratch. The course
recommends it, and every experiment here starts from the authors' checkpoints.

**Gate.** A pass/fail rule written down before seeing results, deciding whether the next, more expensive step runs.

**Hard negative.** A wrong answer that is hard to tell apart from the right one, e.g. a clip from the same video, same
person, same kitchen, but a different step.

**HPC.** Bocconi's computing cluster with GPUs, where all training runs. Jobs are queued with *SLURM*.

**InfoNCE.** The name of the *contrastive* loss the paper uses.

**Loss.** The number training tries to make small: how wrong the model currently is.

**MCQ (5-way).** The test: for each clip, five captions from the same video, one right. Chance is 20%.

**Motion-to-text (m2t).** Given the camera motion, pick the right caption. Our main metric. (Text-to-motion is the
reverse.)

**Negative.** A wrong answer used during *contrastive learning*. Normally the other clips in the *batch*.

**Pilot.** A first, single run used to check that everything works and roughly where results land, before paying for
the full set of runs.

**Procedural vs physical.** The paper's two activity groups. Procedural: step-by-step tasks (cooking, bike repair,
health). Physical: sports and dance (basketball, soccer, dance, music, climbing). Procedural is harder because the
head moves less.

**Reproduction.** Getting the paper's own numbers with our setup. We match their Table 2 exactly (44.81%).

**Regression check.** A test we rerun after every code change: it must still give 44.81%. If not, something broke.

**Seed.** The number that fixes the random choices in a training run. Different seeds give slightly different models,
so we train each setup with 3 seeds to tell real effects from luck.

**SLURM.** The queue system on the *HPC*: you submit a job, it waits for a free GPU, then runs.

**Take.** One recording in Ego-Exo4D. A take is cut into many short clips, each with a caption.

**Trajectory / pose.** The camera's path: where it is and which way it points, many times per second.

**Transformer.** The neural-network design behind most modern AI models. CamFormer is a very small one.

**Validation set.** Data held back from training, used to watch progress. In this dataset the test questions are
built from validation recordings, so we never pick models using validation results.

**Within-take loss.** Our addition in experiment 1: an extra *loss* that asks the model to tell apart clips from the
same recording.
