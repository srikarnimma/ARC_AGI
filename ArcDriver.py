import json
import os
import os.path
import time
import argparse

import numpy as np

from ArcData import ArcData
from ArcProblem import ArcProblem
from ArcSet import ArcSet
from ArcAgent import ArcAgent

def run_training_data(agent: ArcAgent, arc_problems: list[ArcProblem]) -> dict[ArcProblem, tuple[bool, list]]:
    """
    Run each training problem with the test output included so the agent can
    test if they are getting the correct response.
    """
    train_ans_dict: dict[ArcProblem, tuple[bool, list]] = dict()
    total_problems = len(arc_problems)
    for i, trn_problem in enumerate(arc_problems, start=1):
        start_time = time.perf_counter()
        preds: list[np.ndarray] = agent.make_predictions(trn_problem)
        correct = False

        if len(preds) <= 3:
            for prediction in preds:
                answer = trn_problem.test_set().get_output_data().data()
                correct = np.array_equal(answer, prediction)
                if correct: break

        # # store the problem_set and whether it was correctly solved
        train_ans_dict[trn_problem] = (correct, preds)

        elapsed_seconds = time.perf_counter() - start_time
        print(
            f"[{i}/{total_problems}] {trn_problem.problem_name()} "
            f"| correct={correct} | time={elapsed_seconds:.3f}s"
        )

    return train_ans_dict

def load_arc_problems(path: str, problem_data: list[str]) -> list[ArcProblem]:
    problems: list[ArcProblem] = list()
    for problem_name in problem_data:
        with open(os.path.join(path, problem_name)) as p:
            flat_data: dict[str, dict] = json.load(p)
            # convert the data into ArcData (i.e. numpy.ndarray data)
            trn_data: list[ArcSet] = list()
            for dt in flat_data['train']:
                d_input = ArcData(np.array(dt['input']))
                d_output = ArcData(np.array(dt['output']))
                trn_set: ArcSet = ArcSet(arc_input=d_input, arc_output=d_output)
                trn_data.append(trn_set)

            tst_data: list[ArcSet] = list()
            for tst in flat_data['test']:
                t_input = ArcData(np.array(tst['input']))
                t_output = ArcData(np.array(tst['output']))
                tst_set: ArcSet = ArcSet(arc_input=t_input, arc_output=t_output)
                tst_data.append(tst_set)

            arc_problem = ArcProblem(problem_name[:-5], trn_data, tst_data[0])

            # # there should only be one test in the test data
            problems.append(arc_problem)

    return problems


def write_results_csv(file_path: str, milestone_data_set: dict[ArcProblem, tuple[bool, list]]) -> None:
    with open(file_path, 'w') as milestone_file:
        milestone_file.write("Problem Name, Correct, Correct Answer, Prediction 1, Prediction 2, Prediction 3\n")
        for m_answer_set in milestone_data_set.keys():
            m_correct, predictions = milestone_data_set[m_answer_set]
            m_cor_ans = m_answer_set.test_set().get_output_data().data().tolist()
            milestone_file.write(f'{m_answer_set.problem_name()},'
                                 f'{m_correct},'
                                 f'"{m_cor_ans}",')
            if len(predictions) == 0:
                milestone_file.write("empty\n")
                continue
            for i, pred in enumerate(predictions, 1):
                if len(predictions) == i:
                    milestone_file.write(f'"{pred.tolist()}"\n')
                else:
                    milestone_file.write(f'"{pred.tolist()}",')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ARC milestone problems")
    parser.add_argument(
        "-m",
        "--milestone",
        type=str,
        default=None,
        help="Optional milestone folder to run (e.g. B, C, D)",
    )
    parser.add_argument(
        "-p",
        "--problem",
        type=str,
        default=None,
        help="Optional problem id or filename (e.g. 7b6016b9 or 7b6016b9.json)",
    )
    parser.add_argument(
        "-d",
        "--debug",
        action="store_true",
        help="Enable debug printing from ArcAgent",
    )
    args = parser.parse_args()
    milestones_root = 'Milestones'

    if args.milestone:
        selected_milestones = [args.milestone.upper()]
    else:
        selected_milestones = sorted(
            folder_name for folder_name in os.listdir(milestones_root)
            if os.path.isdir(os.path.join(milestones_root, folder_name))
        )

    if args.problem and len(selected_milestones) > 1:
        raise ValueError("When using --problem, please also provide --milestone.")

    # instantiate the agent once
    arc_agent: ArcAgent = ArcAgent(debug=args.debug)

    overall_total = 0
    overall_correct = 0
    milestone_summaries: list[tuple[str, int, int, float]] = []

    for milestone_name in selected_milestones:
        milestone_path = os.path.join(milestones_root, milestone_name)
        if not os.path.isdir(milestone_path):
            raise FileNotFoundError(f"Milestone folder not found: {milestone_path}")

        milestone_data: list[str] = sorted(
            file_name for file_name in os.listdir(milestone_path)
            if file_name.endswith('.json')
        )

        if args.problem:
            problem_file = args.problem if args.problem.endswith('.json') else f"{args.problem}.json"
            if problem_file not in milestone_data:
                raise FileNotFoundError(
                    f"Problem '{args.problem}' not found in {milestone_path}. "
                    f"Expected file: {problem_file}"
                )
            milestone_data = [problem_file]

        print(f"\n=== Running Milestone {milestone_name} ({len(milestone_data)} problem(s)) ===")
        milestone_start = time.perf_counter()
        arc_milestone_problems = load_arc_problems(milestone_path, milestone_data)
        milestone_data_set = run_training_data(arc_agent, arc_milestone_problems)
        milestone_elapsed = time.perf_counter() - milestone_start

        milestone_correct = sum(1 for is_correct, _ in milestone_data_set.values() if is_correct)
        milestone_total = len(milestone_data_set)
        milestone_acc = (100.0 * milestone_correct / milestone_total) if milestone_total > 0 else 0.0

        overall_correct += milestone_correct
        overall_total += milestone_total
        milestone_summaries.append((milestone_name, milestone_correct, milestone_total, milestone_elapsed))

        csv_name = f"Milestone_{milestone_name}_Results.csv"
        write_results_csv(csv_name, milestone_data_set)
        print(
            f"Milestone {milestone_name} summary: {milestone_correct}/{milestone_total} "
            f"({milestone_acc:.1f}%) in {milestone_elapsed:.2f}s | results: {csv_name}"
        )

    print("\n=== Final Summary ===")
    for milestone_name, milestone_correct, milestone_total, milestone_elapsed in milestone_summaries:
        milestone_acc = (100.0 * milestone_correct / milestone_total) if milestone_total > 0 else 0.0
        print(
            f"Milestone {milestone_name}: {milestone_correct}/{milestone_total} "
            f"({milestone_acc:.1f}%) in {milestone_elapsed:.2f}s"
        )

    overall_acc = (100.0 * overall_correct / overall_total) if overall_total > 0 else 0.0
    print(f"Overall: {overall_correct}/{overall_total} ({overall_acc:.1f}%)")
