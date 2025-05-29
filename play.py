import json
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn


def main():
    with open("out/math_baseline.jsonl", "r") as f:
        examples = []
        for line in f:
            ex = json.loads(line)
            ex["reward"] = r1_zero_reward_fn(ex["completion"], ex["ground_truth"])
            examples.append(ex)

    print(examples[0])


if __name__ == "__main__":
    main()
