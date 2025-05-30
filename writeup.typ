#import "@preview/ilm:1.4.1": *
#import "@preview/tablem:0.2.0": *

#let three-line-table = tablem.with(
  render: (columns: auto, ..args) => {
    table(
      columns: columns,
      stroke: none,
      align: center + horizon,
      table.hline(y: 0),
      table.hline(y: 1, stroke: .5pt),
      ..args,
      table.hline(),
    )
  }
)

#set text(lang: "en")

#show: ilm.with(
  title: [CS 336: Assignment 5],
  author: "Brandon Snider",
  date: datetime(year: 2025, month: 06, day: 03),
  figure-index: (enabled: true),
  table-index: (enabled: true),
  listing-index: (enabled: true),
  bibliography: bibliography("refs.bib")
)

#set enum(numbering: "a)")
#set heading(numbering: none)
#show link: underline

#set table(
    inset: 6pt, // default is 5pt
    stroke: (0.5pt + stroke-color),
)

= 3 Measuring Zero-Shot MATH Performance

== Problem (`math_baseline`): 4 points

+ See `cs336_alignment/math_baseline.py`

+ #figure(tablem[
  | Format Reward | Answer Reward | n (of 5,000) |
  |----------|--------------|--------------|
  | 1 | 1 | 80 |
  | 1 | 0 | 769 |
  | 0 | 0 | 4,151 |
], caption: "Formatting and answer rewards, zero-shot baseline")

  The parser seems generally correct, but imperfect. In both sets of examples (format 1 answer 0, and format 0 answer 0), there are occasional instances of incorrect reward assignment, detailed below.

  *Observations from 10 cases in which format reward is 0:*

  In general, these examples show the model failing to follow the format. There are many failure modes: not outputting the closing `</think>` tag, not outputting one of the two answer tags, not outputting any tags at all, using non-XML syntax for the tags, putting the answer in an attribute on the opening `<answer>` tag, etc.

  The parser seems to handle `\frac` strangely. See the first example below in which the model includes a closing `</think>` tag and includes the correct final answer between `<answer>` tags, but still receives a 0 format reward:

  Completion that appears correct, but receives a 0 format reward:

  ```
  Completion:

  // We can write $\frac{1}{31} \times 93$.
  </think>
  <answer> <math>p=\frac{1}{31}\times 93=\frac{1\times 93}{31}=3</math>, as $\frac{31\times 3}{31}=3$</answer>

  Ground truth:

  3
  ```

  Completion without a `</think>` tag:

  ```
  We need to perform the multiplication of the three numbers 10.0000198, 5.9999985401, and 6.9999852, and then round the result to the nearest whole number. Let's do the calculation first. 
  The product of these three numbers is approximately 459.999999999998. 
  <answer> 46 </answer>
  ```

  Completion with answer as an attribute on `<answer`:

  ```
  We can just count these, or we can think that the set of possible rolls is 6 x 6 = 36, so 36 / 9 = 4.
  </think>
  <a answer="1/6">We can also see that the probability is 1/6ths, because, in a 1/36 chance of each possible roll, the probability of rolling a 9 is 4 different rolls out of a possible 36, and 4 / 36 = 1/9.</answer>
  ```

  *Observations from 10 cases in which format reward is 1, but answer reward is 0:*

  In most of these cases, the model just gets the answer wrong. However, there was one observed case in which a correct answer received 0 answer reward, shown below::

  ```
  Completion:
  
  To simplify the expression, we first combine like terms. There are four "a" terms, which when combined give 4a. The constant terms are 1, minus 2, plus 3, minus 4. When we add these together, we get -2, as 1 - 2 + 3 - 4 = -2. The reasoning process involves recognizing like terms and adding or subtracting their coefficients. </think> <answer> $4a - 2$ </answer>

  Ground truth:

  4a-2
  ```
+ The zero-shot baseline performs poorly on MATH, getting 80 of 5,000 examples correct (1.6%).

== Problem (`tokenize_prompt_and_output`): 2 points

See `cs336_alignment/utils.py`

== Problem (`compute_entropy`): 1 point

See `cs336_alignment/utils.py`

== Problem (`get_response_log_probs`): 2 points

See `cs336_alignment/utils.py`

== Problem (`masked_normalize`): 1 point

See `cs336_alignment/utils.py`

== Problem (`sft_microbatch_train_step`): 3 points

See `cs336_alignment/sft_exp/sft_microbatch_train_step.py`

== Problem (`log_generations`): 2 points

TODO

== Problem (`sft_experiment`): 2 points

+ #figure(image("images/sft-exp-mixed.png"), caption: "SFT validation accurary curves, varying unique examples (correct and incorrect)")

+ Number of unique correct examples: *1308*

  Below I show (i) the validation accuracy curve when training on only correct examples, and (ii) all validation accuracy curves plotted together.

  The final validation accuracy when training only on correct examples is *0.4416*. This provides a bump over training on all examples (for which the run with the best final accuracy achieves *0.4338*), though it's a much smaller improvement than I expected.

  I also note that increasing the number of unique examples yielded negligible improvements beyond the smallest set of 128 examples, when the number of training epochs was fixed at 20. Of course, 20 epochs on a set of 256 examples is a much shorter training run than 20 epochs on 1024 examples, so it's interesting and surprising that the final validation accuracy is similar. This may suggest that the model is quite sample-efficient at learning whatever it can from the SFT examples.

  #figure(image("images/sft-exp-correct.png"), caption: "SFT validation accurary curve (correct examples only)")

  #figure(image("images/sft-exp-all.png"), caption: "SFT validation accurary curves (all runs)")

= 5 Expert Iteration for MATH

== Problem (`expert_iteration_experiment`): 2 points

TODO

= 7 Group Relative Policy Optimization

== Problem (`compute_group_normalized_rewards`): 2 points

See `cs336_alignment/grpo_utils.py`

== Problem (`compute_naive_policy_gradient_loss`): 1 point

See `cs336_alignment/grpo_utils.py`

== Problem (`compute_grpo_clip_loss`): 2 points

See `cs336_alignment/grpo_utils.py`

== Problem (`compute_policy_gradient_loss`): 1 point

See `cs336_alignment/grpo_utils.py`

== Problem (`masked_mean`): 1 point

See `cs336_alignment/grpo_utils.py`

== Problem (`grpo_microbatch_train_step`): 3 points

TODO

== Problem (`grpo_train_loop`): 5 points

TODO